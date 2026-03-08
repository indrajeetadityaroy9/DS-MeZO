"""vLLM backend: all inference-engine-specific code isolated here.

The controller calls backend methods that return plain Python types,
keeping the algorithm completely vLLM-free.
"""

import json
import os
import shutil

import torch
from safetensors.torch import save_file
from vllm import SamplingParams
from vllm.lora.request import LoRARequest


ADAPTER_STAGING_DIR = "/dev/shm/ds_mezo"

# vLLM merges these projections into fused modules at runtime.
_VLLM_MERGES = {
    "q_proj": "qkv_proj", "k_proj": "qkv_proj", "v_proj": "qkv_proj",
    "gate_proj": "gate_up_proj", "up_proj": "gate_up_proj",
}


# ---------------------------------------------------------------------------
# Module-level functions (picklable for collective_rpc)
# ---------------------------------------------------------------------------

def _register_activation_hooks(worker, hook_map):
    """Register forward hooks on vLLM's worker model.

    hook_map: {vllm_suffix: [original_module_names]}
    e.g. {"qkv_proj": ["q_proj", "v_proj"]}
    """
    model = worker.get_model()
    worker._ds_mezo_hooks = []
    worker._ds_mezo_activations = {}

    for name, module in model.named_modules():
        if "base_layer" in name:
            continue
        # Match by name suffix first (fast filter)
        suffix = name.rsplit(".", 1)[-1]
        if suffix not in hook_map:
            continue
        parts = name.split(".")
        if "layers" not in parts:
            continue
        layers_pos = parts.index("layers")
        if layers_pos + 1 >= len(parts) or not parts[layers_pos + 1].isdigit():
            continue
        layer_idx = int(parts[layers_pos + 1])
        keys = [(layer_idx, mod) for mod in hook_map[suffix]]

        def hook_fn(mod, inp, out, ks=keys):
            act = inp[0].detach().float().cpu()
            for k in ks:
                worker._ds_mezo_activations[k] = act

        worker._ds_mezo_hooks.append(module.register_forward_hook(hook_fn))
    return len(worker._ds_mezo_hooks)


def _collect_and_remove_hooks(worker):
    """Collect activations and remove hooks."""
    activations = worker._ds_mezo_activations
    for h in worker._ds_mezo_hooks:
        h.remove()
    worker._ds_mezo_hooks = []
    worker._ds_mezo_activations = {}
    return activations


def _extract_prompt_logprobs(output, prompt_token_ids):
    """Extract per-token logprobs from vLLM output."""
    logprobs = []
    for i, token_lp in enumerate(output.prompt_logprobs[1:], 1):
        tok_id = prompt_token_ids[i]
        logprobs.append(token_lp[tok_id].logprob)
    return logprobs


def _write_adapter_config(adapter_dir, rank, target_modules):
    """Write PEFT adapter config JSON."""
    config = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": rank,
        "target_modules": target_modules,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f)


def _save_peft_adapter(A_list, B_list, adapter_dir, layers):
    """Serialize A, B matrices as PEFT-compatible LoRA adapter (BF16).

    PiSSA convention: W = W_res + A @ B where A:(d_out, r), B:(r, d_in)
    PEFT convention:  W = W0 + lora_B @ lora_A where lora_A:(r, d_in), lora_B:(d_out, r)
    Therefore: lora_A.weight = B, lora_B.weight = A
    """
    tensors = {}
    for layer_idx, (A_l, B_l) in enumerate(zip(A_list, B_list)):
        prefix = layers[layer_idx]["peft_prefix"]
        tensors[f"{prefix}.lora_A.weight"] = B_l.bfloat16()
        tensors[f"{prefix}.lora_B.weight"] = A_l.bfloat16()
    save_file(tensors, os.path.join(adapter_dir, "adapter_model.safetensors"))


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class VLLMBackend:
    def __init__(self, engine, layer_specs, rank):
        self.engine = engine

        # Build hook_map from target modules + vLLM merge knowledge
        hook_map = {}
        for ls in layer_specs:
            vllm_name = _VLLM_MERGES.get(ls.module_name, ls.module_name)
            hook_map.setdefault(vllm_name, set()).add(ls.module_name)
        self.hook_map = {k: sorted(v) for k, v in hook_map.items()}

        # Adapter staging on /dev/shm
        shutil.rmtree(ADAPTER_STAGING_DIR, ignore_errors=True)
        self.adapter_dir_pos = os.path.join(ADAPTER_STAGING_DIR, "adapter_pos")
        self.adapter_dir_neg = os.path.join(ADAPTER_STAGING_DIR, "adapter_neg")
        self.checkpoint_dir = os.path.join(ADAPTER_STAGING_DIR, "checkpoints")
        os.makedirs(self.adapter_dir_pos)
        os.makedirs(self.adapter_dir_neg)
        os.makedirs(self.checkpoint_dir)

        unique_modules = list({ls.module_name for ls in layer_specs})
        _write_adapter_config(self.adapter_dir_pos, rank, unique_modules)
        _write_adapter_config(self.adapter_dir_neg, rank, unique_modules)

        self.lora_pos = LoRARequest("adapter_pos", 1, self.adapter_dir_pos, load_inplace=True)
        self.lora_neg = LoRARequest("adapter_neg", 2, self.adapter_dir_neg, load_inplace=True)

        self.score_params = SamplingParams(
            max_tokens=1, prompt_logprobs=1, temperature=0.0
        )

    def sync_adapters(self, pos_overrides, neg_overrides, layers):
        """Serialize PiSSA adapters to /dev/shm for vLLM."""
        def get_AB(overrides):
            A_list, B_list = [], []
            for layer in layers:
                k = (layer["layer_idx"], layer["module_name"])
                if k in overrides:
                    A_l, B_l = overrides[k]
                else:
                    A_l, B_l = layer["A"], layer["B"]
                A_list.append(A_l)
                B_list.append(B_l)
            return A_list, B_list

        A_pos, B_pos = get_AB(pos_overrides)
        _save_peft_adapter(A_pos, B_pos, self.adapter_dir_pos, layers)

        A_neg, B_neg = get_AB(neg_overrides)
        _save_peft_adapter(A_neg, B_neg, self.adapter_dir_neg, layers)

    def generate(self, batch, temperature, n):
        """Generate N candidates. Returns (prompt_token_ids, scored_outputs)."""
        gen_params = SamplingParams(n=n, temperature=temperature)
        request_outputs = self.engine.generate(
            batch, sampling_params=gen_params, lora_request=self.lora_pos
        )
        return request_outputs

    def score(self, token_sequences, lora_request):
        """Prefill-only logprob extraction. Returns list[list[float]]."""
        prompts = [{"prompt_token_ids": seq} for seq in token_sequences]
        outputs = self.engine.generate(
            prompts, sampling_params=self.score_params, lora_request=lora_request,
        )
        return [
            _extract_prompt_logprobs(out, seq)
            for out, seq in zip(outputs, token_sequences)
        ]

    def extract_activations(self, input_data):
        """Hook-based activation capture. Returns {(layer_idx, module_name): Tensor}."""
        self.engine.llm_engine.collective_rpc(
            _register_activation_hooks, args=(self.hook_map,)
        )
        self.engine.generate(
            input_data,
            SamplingParams(max_tokens=1, temperature=0.0),
        )
        results = self.engine.llm_engine.collective_rpc(_collect_and_remove_hooks)
        return results[0]
