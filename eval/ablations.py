"""DS-MeZO ablation evaluation: 8 experiments proving spec claims.

Experiment 0: Memory measurement (§1 near-inference cost)
Experiments 1-7: Controlled A/B ablations, each disabling one component.

Usage: python eval/ablations.py
"""

from __future__ import annotations

import os
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import subprocess
import sys
import time
import types
from typing import Any, Callable

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vllm import LLM, SamplingParams
from ds_mezo.model_config import LayerSpec, discover_layers
from ds_mezo.backend import VLLMBackend
from ds_mezo.controller import DSMeZO_Controller, LayerState
from ds_mezo.kernels import fused_perturb_dual
from eval.utils import extract_code

TOTAL_STEPS = 200

PROMPTS = [
    "Write a Python function that sorts a list using merge sort.",
    "Write a Python class that implements a binary search tree with insert and search.",
    "Write a Python function that finds the longest common subsequence of two strings.",
    "Write a Python function that implements Dijkstra's shortest path algorithm.",
    "Write a Python function that solves the N-queens problem using backtracking.",
    "Write a Python generator that yields prime numbers using the Sieve of Eratosthenes.",
    "Write a Python function that computes the edit distance between two strings.",
    "Write a Python class implementing a min-heap with push and pop operations.",
    "Write a Python function that finds all permutations of a string.",
    "Write a Python function that implements quickselect to find the kth smallest element.",
    "Write a Python function that checks if a directed graph has a cycle.",
    "Write a Python function that performs topological sort on a DAG.",
    "Write a Python function that implements the KMP string matching algorithm.",
    "Write a Python class that implements an LRU cache with O(1) get and put.",
    "Write a Python function that finds the maximum subarray sum using Kadane's algorithm.",
    "Write a Python function that computes the convex hull of a set of 2D points.",
    "Write a Python function that implements matrix multiplication without numpy.",
    "Write a Python function that solves a Sudoku puzzle using constraint propagation.",
    "Write a Python function that implements a trie for prefix searching.",
    "Write a Python function that computes all strongly connected components of a graph.",
]

EVAL_PROMPTS = PROMPTS[:5]


def score_fn(text: str) -> float:
    """Score = length of syntactically valid Python code.
    Returns 0 for non-compilable output, len(code) otherwise."""
    code = extract_code(text)
    try:
        compile(code, "<eval>", "exec")
        return float(len(code))
    except SyntaxError:
        return 0.0


def run_experiment(
    llm: Any,
    layer_specs: list[LayerSpec],
    name: str,
    config_overrides: dict[str, Any],
    patch_fn: Callable[[DSMeZO_Controller], None] | None,
) -> dict[str, Any]:
    """Run a single experiment: init controller, train, eval, return results."""
    torch.manual_seed(42)
    config = {
        "adapter_path": "/dev/shm/pissa_prep/adapter",
        "total_steps": TOTAL_STEPS,
        "score_fn": score_fn,
        **config_overrides,
    }

    backend = VLLMBackend(llm, layer_specs, config.get("rank", 16))
    controller = DSMeZO_Controller(backend, layer_specs, config)

    # Memory measurement via nvidia-smi (vLLM loads model in subprocess)
    nvsmi = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    mem_after_init = float(nvsmi.stdout.strip().split("\n")[0]) / 1024

    controller._calibrate_activation_bases_full([PROMPTS[0]])

    if patch_fn:
        patch_fn(controller)

    t0 = time.time()
    for step_idx in range(TOTAL_STEPS):
        batch = [PROMPTS[step_idx % len(PROMPTS)]]
        controller.step(batch)
        if (step_idx + 1) % 50 == 0:
            loss_str = f"{controller.loss_ema:.2e}"
            print(f"    [{name}] step {step_idx+1}/{TOTAL_STEPS} "
                  f"loss_ema={loss_str} lr={controller.eta:.2e}")
    train_time = time.time() - t0

    mem_peak = torch.cuda.max_memory_allocated() / 1e9

    # Post-training eval
    controller._sync_adapters({}, {})
    outputs = llm.generate(
        EVAL_PROMPTS,
        SamplingParams(max_tokens=256, temperature=0.7),
        lora_request=backend.lora_pos,
    )
    scores = [score_fn(out.outputs[0].text) for out in outputs]
    avg_score = sum(scores) / max(len(scores), 1)

    result = {
        "name": name,
        "avg_score": avg_score,
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

        trajectories, advantages, prompt_len = self._explore(batch)

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

        self._sync_adapters(pos_layers, neg_layers)
        loss_pos, loss_neg = self._score_contrastive(trajectories, advantages, prompt_len)

        if self._check_health(loss_pos, loss_neg):
            raw_dd = float(loss_pos - loss_neg) / (2.0 * self.eps)
            dd = self._clip_dd(raw_dd)

            effective_eta = self._apply_kl_constraint(loss_pos, loss_neg)
            saved_eta = self.eta
            self.eta = effective_eta

            for layer in self.layers:
                z_A, z_B = perturbations[layer.key]
                key_A = (layer.key, "A")
                key_B = (layer.key, "B")

                # SGD+momentum: no Newton-Schulz orthogonalization
                grad_A = dd * z_A
                do_mask = self.step_count > self.mask_warmup
                if do_mask:
                    mask = (torch.sign(grad_A) == torch.sign(self.momentum_buffers[key_A])).float()
                    grad_A = grad_A * mask
                self.momentum_buffers[key_A].mul_(self.momentum).add_((1 - self.momentum) * grad_A)
                layer.A.sub_(self.eta * self.momentum_buffers[key_A])

                grad_B = dd * z_B
                self.momentum_buffers[key_B].mul_(self.momentum).add_((1 - self.momentum) * grad_B)
                layer.B.sub_(self.eta * self.momentum_buffers[key_B])

            self.eta = saved_eta

        self._update_lr()
        self._update_eps()
        self._update_temperature()

        reward_range = self._last_reward_range
        if self.initial_entropy == 0 and reward_range > 0:
            self.initial_entropy = reward_range
        if (self.initial_entropy > 0 and reward_range > 0
                and reward_range < 0.5 * self.initial_entropy):
            self.explore_temperature = min(
                self.explore_temperature * 1.5, self.T_max
            )

    controller.step = types.MethodType(patched_step, controller)


def patch_random_perturbation(controller: DSMeZO_Controller) -> None:
    """Replace AGZO subspace perturbation with random orthonormal basis."""
    def patched_get_perturbation(
        self: DSMeZO_Controller, layer: LayerState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        A, B = layer.A, layer.B
        r_calib = self.r_calib

        d_in = B.shape[1]
        V_random = torch.linalg.qr(torch.randn(d_in, r_calib, device=B.device))[0]

        z_coeff_B = torch.randn(B.shape[0], r_calib, device=B.device)
        z_B = z_coeff_B @ V_random.T

        BV = B @ V_random
        Q, _ = torch.linalg.qr(BV)
        z_coeff_A = torch.randn(A.shape[0], Q.shape[1], device=A.device)
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


# ── Experiment definitions ──────────────────────────────────────────────────

EXPERIMENTS: list[dict[str, Any]] = [
    {"name": "control",       "overrides": {},                                     "patch": None},
    {"name": "no_zomuon",     "overrides": {},                                     "patch": patch_sgd_momentum},
    {"name": "no_agzo",       "overrides": {},                                     "patch": patch_random_perturbation},
    {"name": "no_masking",    "overrides": {"mask_warmup": TOTAL_STEPS + 1},       "patch": None},
    {"name": "no_gr",         "overrides": {"lambda_gr": 0.0},                     "patch": None},
    {"name": "no_kl",         "overrides": {"delta_kl": 100.0},                    "patch": None},
    {"name": "static_bases",  "overrides": {},                                     "patch": patch_static_bases},
    {"name": "fixed_eps",     "overrides": {},                                     "patch": patch_fixed_eps},
]


def main() -> None:
    print("=" * 70)
    print("DS-MeZO ABLATION EVALUATION")
    print(f"Steps per experiment: {TOTAL_STEPS} | Experiments: {len(EXPERIMENTS)}")
    print("=" * 70)

    print("\nLoading vLLM engine...")
    t0 = time.time()
    llm = LLM(
        model="/dev/shm/pissa_prep/residual",
        dtype="bfloat16",
        gpu_memory_utilization=0.92,
        enable_lora=True,
        max_lora_rank=64,
        max_num_seqs=8,
        enforce_eager=True,
    )
    print(f"Engine loaded in {time.time()-t0:.1f}s")

    # Discover layers once
    layer_specs = discover_layers("/dev/shm/pissa_prep/residual", ["q_proj", "v_proj"])

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
        result = run_experiment(llm, layer_specs, exp["name"], exp["overrides"], exp["patch"])
        results.append(result)
        print(f"  Score: {result['avg_score']:.1f} | "
              f"Loss EMA: {result['loss_ema']:.2e} | "
              f"Time: {result['train_time']:.1f}s")

    # Print results
    print("\n" + "=" * 70)
    print("§1 MEMORY: NEAR-INFERENCE COST")
    print("=" * 70)
    control = results[0]
    overhead_gb = control["mem_after_init"] - mem_inference
    overhead_mb = overhead_gb * 1024
    overhead_pct = (overhead_gb / mem_inference) * 100 if mem_inference > 0 else 0
    print(f"  Inference VRAM:    {mem_inference:.3f} GB")
    print(f"  Training VRAM:     {control['mem_after_init']:.3f} GB")
    print(f"  Training overhead: {overhead_mb:.1f} MB ({overhead_pct:.1f}% of inference)")
    print(f"  Peak during step:  {control['mem_peak']:.3f} GB")

    print("\n" + "=" * 70)
    print(f"ABLATION RESULTS ({TOTAL_STEPS} steps)")
    print("=" * 70)
    print(f"{'Experiment':<16} | {'Score':>7} | {'Loss EMA':>10} | {'dd EMA':>10} | {'Time':>6}")
    print("-" * 70)
    for r in results:
        loss_str = f"{r['loss_ema']:.2e}"
        dd_str = f"{r['dd_ema']:.4f}"
        print(f"{r['name']:<16} | {r['avg_score']:7.1f} | {loss_str:>10} | {dd_str:>10} | {r['train_time']:5.1f}s")

    # Delta from control
    ctrl_score = results[0]["avg_score"]
    print(f"\n{'Experiment':<16} | {'Delta vs control':>16}")
    print("-" * 40)
    for r in results[1:]:
        delta = r["avg_score"] - ctrl_score
        print(f"{r['name']:<16} | {delta:>+16.1f}")


if __name__ == "__main__":
    main()
