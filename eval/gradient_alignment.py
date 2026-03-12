"""Gradient alignment diagnostic: cosine similarity between ZO gradient
estimates and true backprop gradients at selected training steps."""

from __future__ import annotations

import argparse
import json
import math
import time
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ds_mezo.model_config import discover_layers
from ds_mezo.backend import VLLMBackend, create_engine
from ds_mezo.controller import DSMeZO_Controller
from eval.benchmarks import load_mbpp_train
from eval.utils import make_exec_reward


def _compute_bp_gradient(
    model_path: Path,
    adapter_path: Path,
    current_weights: dict[str, tuple[torch.Tensor, torch.Tensor]],
    input_text: str,
    layer_specs: list[Any],
) -> dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]]:
    """Compute backprop gradient on CPU for adapter parameters.

    Returns dict mapping layer key → (grad_A, grad_B) in FP32.
    """
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.float32, device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))

    # Overwrite adapter weights with current training state (PiSSA↔PEFT swap)
    state = model.state_dict()
    for spec in layer_specs:
        key = (spec.layer_idx, spec.module_name)
        A, B = current_weights[key]
        lora_a_key = f"{spec.peft_prefix}.lora_A.weight"
        lora_b_key = f"{spec.peft_prefix}.lora_B.weight"
        state[lora_a_key] = B.cpu().float()
        state[lora_b_key] = A.cpu().float()
    model.load_state_dict(state)

    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

    inputs = tokenizer(input_text, return_tensors="pt")
    labels = inputs["input_ids"].clone()
    outputs = model(**inputs, labels=labels)
    outputs.loss.backward()

    gradients: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
    for spec in layer_specs:
        key = (spec.layer_idx, spec.module_name)
        lora_a_key = f"{spec.peft_prefix}.lora_A.weight"
        lora_b_key = f"{spec.peft_prefix}.lora_B.weight"
        grad_B_peft = None
        grad_A_peft = None
        for name, param in model.named_parameters():
            if name == lora_a_key and param.grad is not None:
                grad_B_peft = param.grad.clone()  # lora_A = B (PiSSA swap)
            elif name == lora_b_key and param.grad is not None:
                grad_A_peft = param.grad.clone()  # lora_B = A (PiSSA swap)
        if grad_A_peft is not None and grad_B_peft is not None:
            gradients[key] = (grad_A_peft, grad_B_peft)

    del model
    return gradients


def main() -> None:
    parser = argparse.ArgumentParser(description="ZO vs BP gradient alignment diagnostic")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--probe-steps", type=int, nargs="+",
                        default=[1, 50, 200, 500, 1000])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    peft_config = PeftConfig.from_pretrained(str(args.adapter_path))
    rank = peft_config.r
    target_modules = list(peft_config.target_modules)

    train_data = load_mbpp_train()
    probe_steps = set(args.probe_steps)

    print("=" * 70)
    print("GRADIENT ALIGNMENT DIAGNOSTIC")
    print(f"Probe steps: {sorted(probe_steps)}")
    print("=" * 70)

    print("\nLoading vLLM engine...")
    llm = create_engine(args.model_path, rank)

    reward, set_problem = make_exec_reward()
    layer_specs = discover_layers(args.model_path, target_modules)
    backend = VLLMBackend(llm, layer_specs, rank)
    controller = DSMeZO_Controller(backend, layer_specs, {
        "output_dir": str(args.output_dir),
        "adapter_path": str(args.adapter_path),
        "score_fn": reward,
        "total_steps": args.total_steps,
    })
    controller._calibrate_activation_bases_full([train_data[0]["prompt"]])

    # Storage for captured ZO gradients at probe steps
    captured: dict[str, Any] = {}

    original_step = controller.step

    def capturing_step(self, batch):
        """Wrapper that captures dd and perturbations before the update."""
        self.step_count += 1
        step = self.step_count

        trajectories, advantages, prompt_len = self._explore(batch)
        batch_activations = self.backend.extract_activations(batch)
        self._update_activation_bases(batch_activations)

        perturbations = {}
        for layer in self.layers:
            perturbations[layer.key] = self._get_perturbation(layer)

        from ds_mezo.kernels import fused_perturb_dual
        pos_layers, neg_layers = {}, {}
        for layer in self.layers:
            z_A, z_B = perturbations[layer.key]
            key_A = (layer.key, "A")
            key_B = (layer.key, "B")
            fused_perturb_dual(layer.A, z_A, self.pos_buffers[key_A], self.neg_buffers[key_A])
            fused_perturb_dual(layer.B, z_B, self.pos_buffers[key_B], self.neg_buffers[key_B])
            pos_layers[layer.key] = (self.pos_buffers[key_A], self.pos_buffers[key_B])
            neg_layers[layer.key] = (self.neg_buffers[key_A], self.neg_buffers[key_B])

        self.backend.sync_adapters(pos_layers, neg_layers, self.layers)
        loss_pos, loss_neg = self._score_contrastive(trajectories, advantages, prompt_len)
        dd = float(loss_pos - loss_neg) / (2.0 * self.eps)

        if step in probe_steps:
            captured[step] = {
                "dd": dd,
                "perturbations": {k: (zA.clone(), zB.clone()) for k, (zA, zB) in perturbations.items()},
                "prompt": batch[0],
            }

        # Continue with the actual update
        from ds_mezo.kernels import zo_muon_update
        max_window = int(math.sqrt(self.total_steps))
        momentum = 1.0 - 1.0 / min(self.step_count, max_window)
        for layer in self.layers:
            z_A, z_B = perturbations[layer.key]
            key_A = (layer.key, "A")
            key_B = (layer.key, "B")
            zo_muon_update(layer.A, self.momentum_buffers[key_A],
                           z_A, self.scratch_buffers[key_A],
                           dd, momentum, self.eta, self.norm_floor)
            zo_muon_update(layer.B, self.momentum_buffers[key_B],
                           z_B, self.scratch_buffers[key_B],
                           dd, momentum, self.eta, self.norm_floor)

        self.lr_scheduler.step()
        self.eta = self._lr_opt.param_groups[0]["lr"]

    controller.step = types.MethodType(capturing_step, controller)

    # Training loop
    total_steps = args.total_steps
    print(f"\n--- Training ({total_steps} steps) ---")
    t_start = time.time()
    for step_idx in range(total_steps):
        problem = train_data[step_idx % len(train_data)]
        set_problem(problem["test_list"], problem["test_imports"])
        controller.step([problem["prompt"]])

        if (step_idx + 1) % 100 == 0:
            print(f"  step {step_idx+1}/{total_steps} | lr={controller.eta:.2e}")

    train_time = time.time() - t_start
    print(f"\nTraining complete: {train_time:.1f}s")

    # Compute BP gradients at each probed step's captured state
    print("\n--- Computing BP gradients (CPU) ---")
    alignment_results = []
    for step in sorted(captured.keys()):
        data = captured[step]
        dd = data["dd"]
        prompt = data["prompt"]

        print(f"  Step {step}: computing backprop gradient on CPU...")
        current_weights = {layer.key: (layer.A, layer.B) for layer in controller.layers}
        bp_grads = _compute_bp_gradient(
            args.model_path, args.adapter_path,
            current_weights, prompt, layer_specs,
        )

        step_result: dict[str, Any] = {"step": step, "dd": dd, "cosine_sim": {}}
        for layer in controller.layers:
            key = layer.key
            if key not in bp_grads:
                continue
            z_A, z_B = data["perturbations"][key]
            bp_A, bp_B = bp_grads[key]

            zo_grad_A = (dd * z_A).cpu().float().flatten()
            zo_grad_B = (dd * z_B).cpu().float().flatten()
            bp_grad_A = bp_A.flatten()
            bp_grad_B = bp_B.flatten()

            cos_A = F.cosine_similarity(zo_grad_A.unsqueeze(0), bp_grad_A.unsqueeze(0)).item()
            cos_B = F.cosine_similarity(zo_grad_B.unsqueeze(0), bp_grad_B.unsqueeze(0)).item()

            layer_name = f"layer_{key[0]}_{key[1]}"
            step_result["cosine_sim"][f"{layer_name}_A"] = cos_A
            step_result["cosine_sim"][f"{layer_name}_B"] = cos_B

        # Aggregate across layers
        all_cos = list(step_result["cosine_sim"].values())
        step_result["mean_cosine_sim"] = sum(all_cos) / len(all_cos) if all_cos else 0.0
        alignment_results.append(step_result)

        print(f"    Mean cosine similarity: {step_result['mean_cosine_sim']:.4f}")

    results_path = args.output_dir / "gradient_alignment.json"
    with open(results_path, "w") as f:
        json.dump({
            "total_steps": total_steps,
            "probe_steps": sorted(probe_steps),
            "train_time": train_time,
            "alignment": alignment_results,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
