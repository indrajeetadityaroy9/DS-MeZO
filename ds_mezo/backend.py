"""vLLM backend: adapter hot-swap, scoring, and activation extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from ds_mezo.adapter_io import save_peft_adapter, write_adapter_config

def create_engine(model_path: str | Path, rank: int) -> LLM:
    return LLM(
        model=str(model_path),
        dtype="bfloat16",
        gpu_memory_utilization=0.95,
        enable_lora=True,
        max_lora_rank=max(64, rank),
        enforce_eager=True,
    )


_VLLM_MERGES: dict[str, str] = {
    "q_proj": "qkv_proj", "k_proj": "qkv_proj", "v_proj": "qkv_proj",
    "gate_proj": "gate_up_proj", "up_proj": "gate_up_proj",
}


def _register_activation_hooks(
    worker: Any, hook_map: dict[str, list[str]],
) -> int:
    model = worker.get_model()
    worker._ds_mezo_hooks = []
    worker._ds_mezo_activations = {}

    for name, module in model.named_modules():
        suffix = name.rsplit(".", 1)[-1]
        if suffix not in hook_map:
            continue
        parts = name.split(".")
        layers_pos = parts.index("layers")
        layer_idx = int(parts[layers_pos + 1])
        keys = [(layer_idx, mod) for mod in hook_map[suffix]]

        def hook_fn(
            mod: Any, inp: tuple[torch.Tensor, ...], out: Any,
            ks: list[tuple[int, str]] = keys,
        ) -> None:
            act = inp[0].detach().float()
            for k in ks:
                worker._ds_mezo_activations[k] = act

        worker._ds_mezo_hooks.append(module.register_forward_hook(hook_fn))
    return len(worker._ds_mezo_hooks)


def _collect_and_remove_hooks(
    worker: Any,
) -> dict[tuple[int, str], torch.Tensor]:
    seen: dict[int, torch.Tensor] = {}
    activations: dict[tuple[int, str], torch.Tensor] = {}
    for k, v in worker._ds_mezo_activations.items():
        tid = id(v)
        if tid not in seen:
            cpu_t = torch.empty(v.shape, dtype=v.dtype, pin_memory=True)
            cpu_t.copy_(v, non_blocking=True)
            seen[tid] = cpu_t
        activations[k] = seen[tid]
    torch.cuda.current_stream().synchronize()
    for h in worker._ds_mezo_hooks:
        h.remove()
    worker._ds_mezo_hooks = []
    worker._ds_mezo_activations = {}
    return activations


def _extract_prompt_logprobs(
    output: Any, prompt_token_ids: list[int],
) -> list[float]:
    """Extract per-token logprobs from vLLM output."""
    logprobs = []
    for i, token_lp in enumerate(output.prompt_logprobs[1:], 1):
        tok_id = prompt_token_ids[i]
        logprobs.append(token_lp[tok_id].logprob)
    return logprobs


class VLLMBackend:
    def __init__(
        self, engine: Any, layer_specs: list[Any], rank: int,
        staging_dir: Path | str = Path("/dev/shm/ds_mezo"),
    ) -> None:
        self.engine = engine
        self.rank = rank

        hook_map: dict[str, set[str]] = {}
        for ls in layer_specs:
            vllm_name = _VLLM_MERGES.get(ls.module_name, ls.module_name)
            hook_map.setdefault(vllm_name, set()).add(ls.module_name)
        self.hook_map: dict[str, list[str]] = {
            k: sorted(v) for k, v in hook_map.items()
        }

        unique_modules = sorted({ls.module_name for ls in layer_specs})
        staging = Path(staging_dir)
        (staging / "adapter_pos").mkdir(parents=True, exist_ok=True)
        (staging / "adapter_neg").mkdir(parents=True, exist_ok=True)
        self.adapter_dir_pos = staging / "adapter_pos"
        self.adapter_dir_neg = staging / "adapter_neg"

        write_adapter_config(self.adapter_dir_pos, rank, unique_modules)
        write_adapter_config(self.adapter_dir_neg, rank, unique_modules)

        self.lora_pos = LoRARequest("adapter_pos", 1, str(self.adapter_dir_pos), load_inplace=True)
        self.lora_neg = LoRARequest("adapter_neg", 2, str(self.adapter_dir_neg), load_inplace=True)

        self.score_params = SamplingParams(
            max_tokens=1, prompt_logprobs=1, temperature=0.0
        )

        self._bf16_pos: dict[str, torch.Tensor] = {}
        self._bf16_neg: dict[str, torch.Tensor] = {}
        self.query_count = 0

    def sync_adapters(
        self,
        pos_overrides: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]],
        neg_overrides: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]],
        layers: list[Any],
    ) -> None:
        def get_AB(
            overrides: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]],
        ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
            A_list, B_list = [], []
            for layer in layers:
                k = layer.key
                if k in overrides:
                    A_l, B_l = overrides[k]
                else:
                    A_l, B_l = layer.A, layer.B
                A_list.append(A_l)
                B_list.append(B_l)
            return A_list, B_list

        A_pos, B_pos = get_AB(pos_overrides)
        A_neg, B_neg = get_AB(neg_overrides)

        save_peft_adapter(A_pos, B_pos, self.adapter_dir_pos, layers, self._bf16_pos)
        save_peft_adapter(A_neg, B_neg, self.adapter_dir_neg, layers, self._bf16_neg)

        self.engine.llm_engine.add_lora(self.lora_pos)
        self.engine.llm_engine.add_lora(self.lora_neg)

    def generate(
        self, batch: list[str], temperature: float, n: int,
    ) -> list[Any]:
        self.query_count += len(batch) * n
        gen_params = SamplingParams(n=n, temperature=temperature)
        return self.engine.generate(
            batch, sampling_params=gen_params, lora_request=self.lora_pos
        )

    def score(
        self, token_sequences: list[list[int]], lora_request: LoRARequest,
    ) -> list[list[float]]:
        self.query_count += len(token_sequences)
        prompts = [{"prompt_token_ids": seq} for seq in token_sequences]
        outputs = self.engine.generate(
            prompts, sampling_params=self.score_params, lora_request=lora_request,
        )
        return [
            _extract_prompt_logprobs(out, seq)
            for out, seq in zip(outputs, token_sequences)
        ]

    def extract_activations(
        self, input_data: list[str],
    ) -> dict[tuple[int, str], torch.Tensor]:
        self.query_count += len(input_data)
        self.engine.collective_rpc(
            _register_activation_hooks, args=(self.hook_map,)
        )
        self.engine.generate(
            input_data,
            SamplingParams(max_tokens=1, temperature=0.0),
        )
        results = self.engine.collective_rpc(_collect_and_remove_hooks)
        return results[0]
