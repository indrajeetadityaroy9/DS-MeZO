"""Gradient alignment diagnostic: cosine similarity between ZO gradient
estimates and true backprop gradients at selected training steps."""

from __future__ import annotations

import argparse
import json
import time
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ds_mezo.controller import DSMeZO_Controller
from ds_mezo import build_controller
from eval.data import load_mbpp_train
from eval.rewards import make_exec_reward


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

    param_map = {name: param for name, param in model.named_parameters()}
    gradients: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
    for spec in layer_specs:
        key = (spec.layer_idx, spec.module_name)
        # PiSSA swap: lora_A = B, lora_B = A
        grad_B = param_map[f"{spec.peft_prefix}.lora_A.weight"].grad.clone()
        grad_A = param_map[f"{spec.peft_prefix}.lora_B.weight"].grad.clone()
        gradients[key] = (grad_A, grad_B)

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

    train_data = load_mbpp_train()
    probe_steps = set(args.probe_steps)

    reward, set_problem = make_exec_reward()
    llm, backend, controller, rank, layer_specs = build_controller(
        args.model_path, args.adapter_path, args.output_dir, args.total_steps,
        score_fn=reward, calibration_prompt=train_data[0]["prompt"],
    )

    print("=" * 70)
    print("GRADIENT ALIGNMENT DIAGNOSTIC")
    print(f"Probe steps: {sorted(probe_steps)}")
    print("=" * 70)

    # Storage for captured ZO gradients at probe steps
    captured: dict[str, Any] = {}

    def capturing_step(self, batch):
        """Wrapper that captures dd and perturbations before the update."""
        self.step_count += 1
        trajectories, advantages, prompt_len = self._explore(batch)

        # Dynamic sampling: skip when advantages are below SPSA truncation floor
        if max(abs(a) for a in advantages) < self.eps:
            self._step_lr()
            return

        perturbations = self._perturb_and_sync(batch)
        loss_pos, loss_neg = self._score_contrastive(trajectories, advantages, prompt_len)
        dd = float(loss_pos - loss_neg) / (2.0 * self.eps)

        if self.step_count in probe_steps:
            captured[self.step_count] = {
                "dd": dd,
                "perturbations": {k: (zA.clone(), zB.clone()) for k, (zA, zB) in perturbations.items()},
                "prompt": batch[0],
            }

        self._update_weights(perturbations, dd)

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
        all_cos = step_result["cosine_sim"].values()
        step_result["mean_cosine_sim"] = sum(all_cos) / len(all_cos)
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
