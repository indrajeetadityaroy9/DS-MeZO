"""DS-MeZO ablation evaluation: 8 experiments proving spec claims.

Experiment 0: Memory measurement (§1 near-inference cost)
Experiments 1-7: Controlled A/B ablations, each disabling one component.

Training uses MBPP train split with execution-based reward.
Evaluation uses MBPP test split pass@1 (greedy, execution-based).

Usage: python -m eval.ablations --model-path <path> --adapter-path <path>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import types
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from vllm import LLM
from ds_mezo.model_config import LayerSpec, discover_layers
from ds_mezo.backend import VLLMBackend
from ds_mezo.controller import DSMeZO_Controller, LayerState
from ds_mezo.kernels import fused_perturb_dual
from eval.utils import ExecReward
from eval.benchmarks import eval_mbpp, load_mbpp_train

TOTAL_STEPS = 200


def run_experiment(
    llm: Any,
    layer_specs: list[LayerSpec],
    name: str,
    patch_fn: Callable[[DSMeZO_Controller], None] | None,
    adapter_path: Path,
    output_dir: Path,
    train_data: list[dict[str, Any]],
    rank: int,
) -> dict[str, Any]:
    """Run a single experiment: init controller, train, eval, return results."""
    torch.manual_seed(42)
    experiment_dir = output_dir / name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    reward = ExecReward()
    config = {
        "output_dir": str(experiment_dir),
        "adapter_path": str(adapter_path),
        "total_steps": TOTAL_STEPS,
        "score_fn": reward,
    }

    backend = VLLMBackend(llm, layer_specs, rank)
    controller = DSMeZO_Controller(backend, layer_specs, config)

    # Memory measurement via nvidia-smi (vLLM loads model in subprocess)
    nvsmi = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    mem_after_init = float(nvsmi.stdout.strip().split("\n")[0]) / 1024

    controller._calibrate_activation_bases_full([train_data[0]["prompt"]])

    if patch_fn:
        patch_fn(controller)

    t0 = time.time()
    for step_idx in range(TOTAL_STEPS):
        problem = train_data[step_idx % len(train_data)]
        reward.set_problem(problem["test_list"], problem["test_imports"])
        controller.step([problem["prompt"]])
        if (step_idx + 1) % 50 == 0:
            print(f"    [{name}] step {step_idx+1}/{TOTAL_STEPS} "
                  f"loss_ema={controller.loss_ema:.2e} lr={controller.eta:.2e}")
    train_time = time.time() - t0

    mem_peak = torch.cuda.max_memory_allocated() / 1e9

    # Post-training eval: MBPP pass@1 (greedy, execution-based)
    controller.backend.sync_adapters({}, {}, controller.layers)
    mbpp_result = eval_mbpp(llm, lora_request=backend.lora_pos, n_samples=1, temperature=0.0)

    result = {
        "name": name,
        "pass@1": mbpp_result["pass@1"],
        "loss_ema": controller.loss_ema,
        "dd_ema": controller.dd_ema,
        "train_time": train_time,
        "mem_after_init": mem_after_init,
        "mem_peak": mem_peak,
    }

    del controller
    del backend
    torch.cuda.reset_peak_memory_stats()

    return result


# ── Monkey-patch functions ──────────────────────────────────────────────────

def patch_sgd_momentum(controller: DSMeZO_Controller) -> None:
    """Replace ZO-Muon (N-S orthogonalization) with plain SGD+momentum."""

    def patched_step(self: DSMeZO_Controller, batch: list[str]) -> None:
        self.step_count += 1

        trajectories, advantages, prompt_len, reward_var = self._explore(batch)

        batch_activations = self.backend.extract_activations(batch)
        needs_recalib = self._update_activation_bases_power_iter(batch_activations)
        if needs_recalib:
            self._calibrate_activation_bases_full(batch)

        perturbations: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in self.layers:
            perturbations[layer.key] = self._get_perturbation(layer)

        pos_layers: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        neg_layers: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in self.layers:
            z_A, z_B = perturbations[layer.key]
            pos_A, neg_A = torch.empty_like(layer.A), torch.empty_like(layer.A)
            pos_B, neg_B = torch.empty_like(layer.B), torch.empty_like(layer.B)
            fused_perturb_dual(layer.A, z_A, pos_A, neg_A)
            fused_perturb_dual(layer.B, z_B, pos_B, neg_B)
            pos_layers[layer.key] = (pos_A, pos_B)
            neg_layers[layer.key] = (neg_A, neg_B)

        self.backend.sync_adapters(pos_layers, neg_layers, self.layers)
        loss_pos, loss_neg = self._score_contrastive(trajectories, advantages, prompt_len)

        if self._check_health(loss_pos, loss_neg):
            raw_dd = float(loss_pos - loss_neg) / (2.0 * self.eps)
            dd = self._clip_dd(raw_dd)

            effective_eta = self._apply_kl_constraint(loss_pos, loss_neg)

            for layer in self.layers:
                z_A, z_B = perturbations[layer.key]
                key_A = (layer.key, "A")
                key_B = (layer.key, "B")

                # SGD+momentum with continuous masking (no Newton-Schulz)
                grad_A = dd * z_A
                if self.step_count > self.mask_warmup:
                    cossim = F.cosine_similarity(
                        grad_A.reshape(1, -1),
                        self.momentum_buffers[key_A].reshape(1, -1),
                    ).item()
                    s = (cossim + 1.0) / 2.0
                    self.mask_scale_ema[key_A] = (
                        self.momentum * self.mask_scale_ema[key_A]
                        + (1.0 - self.momentum) * s
                    )
                    grad_A = grad_A * self.mask_scale_ema[key_A]
                self.momentum_buffers[key_A].mul_(self.momentum).add_((1 - self.momentum) * grad_A)
                layer.A.sub_(effective_eta * self.momentum_buffers[key_A])

                grad_B = dd * z_B
                self.momentum_buffers[key_B].mul_(self.momentum).add_((1 - self.momentum) * grad_B)
                layer.B.sub_(effective_eta * self.momentum_buffers[key_B])

        self._update_lr()
        self._update_eps()
        self._update_temperature()

        # Reward-variance exploration monitoring
        alpha = self._ema_alpha()
        if self.initial_reward_var == 0.0 and reward_var > 0:
            self.reward_var_ema = reward_var
            self.initial_reward_var = reward_var
        elif self.initial_reward_var > 0:
            self.reward_var_ema += alpha * (reward_var - self.reward_var_ema)
            if self.reward_var_ema < self.initial_reward_var * 0.25:
                self.explore_temperature = self.T_max

    controller.step = types.MethodType(patched_step, controller)


def patch_random_perturbation(controller: DSMeZO_Controller) -> None:
    """Replace AGZO subspace perturbation with random orthonormal basis."""
    def patched_get_perturbation(
        self: DSMeZO_Controller, layer: LayerState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        A, B = layer.A, layer.B
        r_calib = self.r_calib

        d_in = B.shape[1]
        V_random = torch.linalg.qr(torch.randn(d_in, r_calib, device="cuda"))[0]

        z_coeff_B = torch.randn(B.shape[0], r_calib, device="cuda")
        z_B = z_coeff_B @ V_random.T

        BV = B @ V_random
        Q, _ = torch.linalg.qr(BV)
        z_coeff_A = torch.randn(A.shape[0], Q.shape[1], device="cuda")
        z_A = z_coeff_A @ Q.T

        return z_A * self.eps, z_B * self.eps

    controller._get_perturbation = types.MethodType(patched_get_perturbation, controller)


def patch_static_bases(controller: DSMeZO_Controller) -> None:
    """Freeze activation bases at initial SVD — no per-step power iteration."""
    controller._update_activation_bases_power_iter = types.MethodType(
        lambda self, activations: False, controller
    )


def patch_fixed_eps(controller: DSMeZO_Controller) -> None:
    """Disable adaptive epsilon — keep at initial value."""
    controller._update_eps = types.MethodType(lambda self: None, controller)


def patch_no_masking(controller: DSMeZO_Controller) -> None:
    """Disable continuous masking by setting warmup beyond total steps."""
    controller.mask_warmup = controller.total_steps + 1


def patch_no_gr(controller: DSMeZO_Controller) -> None:
    """Disable gradient regularization."""
    controller.lambda_gr = 0.0


def patch_no_kl(controller: DSMeZO_Controller) -> None:
    """Disable KL constraint."""
    controller.delta_kl = float('inf')


# ── Experiment definitions ──────────────────────────────────────────────────

EXPERIMENTS: list[dict[str, Any]] = [
    {"name": "control",       "patch": None},
    {"name": "no_zomuon",     "patch": patch_sgd_momentum},
    {"name": "no_agzo",       "patch": patch_random_perturbation},
    {"name": "no_masking",    "patch": patch_no_masking},
    {"name": "no_gr",         "patch": patch_no_gr},
    {"name": "no_kl",         "patch": patch_no_kl},
    {"name": "static_bases",  "patch": patch_static_bases},
    {"name": "fixed_eps",     "patch": patch_fixed_eps},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="DS-MeZO ablation evaluation")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Read rank and target modules from adapter config
    adapter_config = json.loads((args.adapter_path / "adapter_config.json").read_text())
    rank = adapter_config["r"]
    target_modules = adapter_config["target_modules"]

    # Load MBPP train split for training data
    train_data = load_mbpp_train()

    print("=" * 70)
    print("DS-MeZO ABLATION EVALUATION")
    print(f"Steps per experiment: {TOTAL_STEPS} | Experiments: {len(EXPERIMENTS)}")
    print(f"Training data: {len(train_data)} MBPP problems")
    print("=" * 70)

    print("\nLoading vLLM engine...")
    t0 = time.time()
    llm = LLM(
        model=str(args.model_path),
        dtype="bfloat16",
        gpu_memory_utilization=0.92,
        enable_lora=True,
        max_lora_rank=max(64, rank),
        max_num_seqs=8,
        enforce_eager=True,
    )
    print(f"Engine loaded in {time.time()-t0:.1f}s")

    # Discover layers once
    layer_specs = discover_layers(args.model_path, target_modules)

    # Experiment 0: inference-only memory baseline
    nvsmi = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    mem_inference = float(nvsmi.stdout.strip().split("\n")[0]) / 1024
    print(f"\n=== §1 Memory Baseline ===")
    print(f"  Inference VRAM (nvidia-smi): {mem_inference:.3f} GB")

    torch.cuda.reset_peak_memory_stats()

    # Run all experiments
    results: list[dict[str, Any]] = []
    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n--- Experiment {i+1}/{len(EXPERIMENTS)}: {exp['name']} ---")
        result = run_experiment(
            llm, layer_specs, exp["name"], exp["patch"],
            args.adapter_path, args.output_dir, train_data, rank,
        )
        results.append(result)
        print(f"  pass@1: {result['pass@1']:.1%} | "
              f"Loss EMA: {result['loss_ema']:.2e} | "
              f"Time: {result['train_time']:.1f}s")

    # Print results
    print("\n" + "=" * 70)
    print("§1 MEMORY: NEAR-INFERENCE COST")
    print("=" * 70)
    control = results[0]
    overhead_gb = control["mem_after_init"] - mem_inference
    overhead_mb = overhead_gb * 1024
    overhead_pct = (overhead_gb / mem_inference) * 100
    print(f"  Inference VRAM:    {mem_inference:.3f} GB")
    print(f"  Training VRAM:     {control['mem_after_init']:.3f} GB")
    print(f"  Training overhead: {overhead_mb:.1f} MB ({overhead_pct:.1f}% of inference)")
    print(f"  Peak during step:  {control['mem_peak']:.3f} GB")

    print("\n" + "=" * 70)
    print(f"ABLATION RESULTS ({TOTAL_STEPS} steps)")
    print("=" * 70)
    print(f"{'Experiment':<16} | {'pass@1':>7} | {'Loss EMA':>10} | {'dd EMA':>10} | {'Time':>6}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<16} | {r['pass@1']:6.1%} | {r['loss_ema']:>10.2e} | {r['dd_ema']:>10.4f} | {r['train_time']:5.1f}s")

    # Delta from control
    ctrl_pass = results[0]["pass@1"]
    print(f"\n{'Experiment':<16} | {'Delta vs control':>16}")
    print("-" * 40)
    for r in results[1:]:
        delta = r["pass@1"] - ctrl_pass
        print(f"{r['name']:<16} | {delta:>+15.1%}")


if __name__ == "__main__":
    main()
