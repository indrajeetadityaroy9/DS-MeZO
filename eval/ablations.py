"""DS-MeZO ablation evaluation: memory measurement + controlled A/B ablations
including MeZO baseline (R1) and LoRA init comparison (R15)."""

from __future__ import annotations

import argparse
import json
import math
import time
import types
from pathlib import Path
from typing import Any, Callable

import torch
from ds_mezo.model_config import LayerSpec
from ds_mezo.controller import DSMeZO_Controller, LayerState
from eval.benchmarks import (
    make_exec_reward, eval_mbpp, load_mbpp_train, load_apps_train,
    load_adapter_config, setup_controller,
)


def run_experiment(
    llm: Any,
    layer_specs: list[LayerSpec],
    name: str,
    patch_fn: Callable[[DSMeZO_Controller], None],
    adapter_path: Path,
    output_dir: Path,
    train_data: list[dict[str, Any]],
    rank: int,
    total_steps: int,
    n_samples: int,
    temperature: float,
) -> dict[str, Any]:
    torch.manual_seed(42)
    experiment_dir = output_dir / name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    reward, set_problem = make_exec_reward()

    _, backend, controller, _, _ = setup_controller(
        model_path=None, adapter_path=adapter_path,
        output_dir=experiment_dir, total_steps=total_steps,
        score_fn=reward, engine=llm, layer_specs=layer_specs, rank=rank,
    )

    free, total = torch.cuda.mem_get_info()
    mem_after_init = (total - free) / (1024 ** 3)

    controller._calibrate_activation_bases_full([train_data[0]["prompt"]])
    patch_fn(controller)

    log_interval = max(1, total_steps // 20)
    t0 = time.time()
    for step_idx in range(total_steps):
        problem = train_data[step_idx % len(train_data)]
        set_problem(problem["test_list"], problem["test_imports"])
        controller.step([problem["prompt"]])
        if (step_idx + 1) % log_interval == 0:
            print(f"    [{name}] step {step_idx+1}/{total_steps} "
                  f"lr={controller.eta:.2e}")
    train_time = time.time() - t0

    mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)

    controller.backend.sync_adapters({}, {}, controller.layers)
    mbpp_result = eval_mbpp(
        llm, lora_request=backend.lora_pos,
        n_samples=n_samples, temperature=temperature,
    )

    result = {
        "name": name,
        "pass@1": mbpp_result["pass@1"],
        "pass@1_ci95": mbpp_result["pass@1_ci95"],
        "train_time": train_time,
        "mem_after_init": mem_after_init,
        "mem_peak": mem_peak,
        "total_steps": total_steps,
        "query_count": backend.query_count,
    }

    torch.cuda.reset_peak_memory_stats()
    return result


# ── Monkey-patch functions ──────────────────────────────────────────────────

def patch_sgd_momentum(controller: DSMeZO_Controller) -> None:
    """Replace BSCO (Kalman + N-S) with plain SGD+momentum."""

    def sgd_update(self, perturbations, dd):
        max_window = int(math.sqrt(self.total_steps))
        self._momentum = 1.0 - 1.0 / min(self.step_count, max_window)
        mom = self._momentum
        for layer in self.layers:
            z_A, z_B = perturbations[layer.key]
            for z_mat, suffix, param in [(z_A, "A", layer.A), (z_B, "B", layer.B)]:
                key = (layer.key, suffix)
                self.momentum_buffers[key].mul_(mom).add_((1 - mom) * dd * z_mat)
                param.sub_(self.eta * self.momentum_buffers[key])
        self.lr_scheduler.step()
        self.eta = self._lr_opt.param_groups[0]["lr"]

    controller._update_weights = types.MethodType(sgd_update, controller)


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
    controller._update_activation_bases = types.MethodType(
        lambda self, activations: None, controller
    )


def patch_no_kalman(controller: DSMeZO_Controller) -> None:
    """Replace Kalman with EMA momentum + N-S kernel (pre-BSCO behavior)."""
    from ds_mezo.kernels import zo_muon_update

    def ema_update(self, perturbations, dd):
        max_window = int(math.sqrt(self.total_steps))
        self._momentum = 1.0 - 1.0 / min(self.step_count, max_window)
        mom = self._momentum
        for layer in self.layers:
            z_A, z_B = perturbations[layer.key]
            for z_mat, suffix, param in [(z_A, "A", layer.A), (z_B, "B", layer.B)]:
                key = (layer.key, suffix)
                buf = self.momentum_buffers[key]
                buf.mul_(mom).add_((1 - mom) * dd * z_mat)
                zo_muon_update(param, buf, self.scratch_buffers[key], self.eta, self.norm_floor)
        self.lr_scheduler.step()
        self.eta = self._lr_opt.param_groups[0]["lr"]

    controller._update_weights = types.MethodType(ema_update, controller)


def patch_no_shrinkage(controller: DSMeZO_Controller) -> None:
    """Replace shrinkage RLOO with vanilla RLOO (pre-BSCO behavior)."""

    def vanilla_explore(self, batch):
        self.backend.sync_adapters({}, {}, self.layers)
        request_outputs = self.backend.generate(
            batch, self.temperature, self.num_candidates
        )
        prompt_token_ids = request_outputs[0].prompt_token_ids

        scored = [
            (out, self.score_fn(out.text))
            for out in request_outputs[0].outputs
        ]

        rewards = torch.tensor([r for _, r in scored])
        N = len(scored)
        baselines = (rewards.sum() - rewards) / (N - 1)
        advantages = [float(a) for a in (rewards - baselines)]

        trajectories = [
            list(prompt_token_ids) + list(out.token_ids) for out, _ in scored
        ]
        prompt_len = len(prompt_token_ids)

        return trajectories, advantages, prompt_len

    controller._explore = types.MethodType(vanilla_explore, controller)


def patch_mezo(controller: DSMeZO_Controller) -> None:
    """MeZO baseline: vanilla ZO-SGD on PiSSA. No AGZO, no ZO-Muon, no activation tracking."""
    patch_sgd_momentum(controller)
    patch_random_perturbation(controller)
    patch_static_bases(controller)


def patch_lora_init(controller: DSMeZO_Controller) -> None:
    """Replace PiSSA adapter weights with standard LoRA initialization."""
    eps_dtype = torch.finfo(torch.float32).eps
    norms = []
    for layer in controller.layers:
        torch.nn.init.kaiming_uniform_(layer.A, a=math.sqrt(5))
        torch.nn.init.zeros_(layer.B)
        norms.append(torch.linalg.norm(layer.A).item())
        for key in [(layer.key, "A"), (layer.key, "B")]:
            controller.momentum_buffers[key].zero_()
            controller.variance_buffers[key].fill_(controller.eps ** 2)
    controller.eps = float(torch.tensor(norms).median()) * eps_dtype ** (1.0 / 3.0)


# ── Experiment definitions ──────────────────────────────────────────────────

def _noop(controller: DSMeZO_Controller) -> None:
    """No-op patch for control experiment."""


EXPERIMENTS: list[dict[str, Any]] = [
    {"name": "control",       "patch": _noop},
    {"name": "mezo",          "patch": patch_mezo},
    {"name": "no_zomuon",     "patch": patch_sgd_momentum},
    {"name": "no_agzo",       "patch": patch_random_perturbation},
    {"name": "static_bases",  "patch": patch_static_bases},
    {"name": "lora_init",     "patch": patch_lora_init},
    {"name": "no_kalman",     "patch": patch_no_kalman},
    {"name": "no_shrinkage",  "patch": patch_no_shrinkage},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="DS-MeZO ablation evaluation")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--train-data", choices=["mbpp", "apps"], default="mbpp")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Run subset of experiments by name")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rank, target_modules = load_adapter_config(args.adapter_path)

    if args.train_data == "apps":
        train_data = load_apps_train()
    else:
        train_data = load_mbpp_train()

    experiments = EXPERIMENTS
    if args.experiments:
        exp_names = set(args.experiments)
        experiments = [e for e in EXPERIMENTS if e["name"] in exp_names]

    from ds_mezo.backend import create_engine
    from ds_mezo.model_config import discover_layers

    print("=" * 70)
    print("DS-MeZO ABLATION EVALUATION")
    print(f"Steps per experiment: {args.total_steps} | Experiments: {len(experiments)}")
    print(f"Training data: {len(train_data)} problems ({args.train_data})")
    print(f"Eval: MBPP pass@k (n={args.n_samples}, T={args.temperature})")
    print("=" * 70)

    print("\nLoading vLLM engine...")
    t0 = time.time()
    llm = create_engine(args.model_path, rank)
    print(f"Engine loaded in {time.time()-t0:.1f}s")

    layer_specs = discover_layers(args.model_path, target_modules)

    free, total = torch.cuda.mem_get_info()
    mem_inference = (total - free) / (1024 ** 3)
    print(f"\n=== §1 Memory Baseline ===")
    print(f"  Inference VRAM: {mem_inference:.3f} GB")

    torch.cuda.reset_peak_memory_stats()

    results: list[dict[str, Any]] = []
    for i, exp in enumerate(experiments):
        print(f"\n--- Experiment {i+1}/{len(experiments)}: {exp['name']} ---")
        result = run_experiment(
            llm, layer_specs, exp["name"], exp["patch"],
            args.adapter_path, args.output_dir, train_data, rank,
            total_steps=args.total_steps,
            n_samples=args.n_samples,
            temperature=args.temperature,
        )
        results.append(result)
        print(f"  pass@1: {result['pass@1']:.1%} | "
              f"Time: {result['train_time']:.1f}s | "
              f"Queries: {result['query_count']}")

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
    print(f"ABLATION RESULTS ({args.total_steps} steps)")
    print("=" * 70)
    print(f"{'Experiment':<16} | {'pass@1':>7} | {'95% CI':>17} | {'Time':>6} | {'Queries':>8}")
    print("-" * 70)
    for r in results:
        ci = r['pass@1_ci95']
        print(f"{r['name']:<16} | {r['pass@1']:6.1%} | {ci[0]:6.1%}–{ci[1]:6.1%} | "
              f"{r['train_time']:5.1f}s | {r['query_count']:>8}")

    ctrl_pass = results[0]["pass@1"]
    print(f"\n{'Experiment':<16} | {'Delta vs control':>16}")
    print("-" * 40)
    for r in results[1:]:
        delta = r["pass@1"] - ctrl_pass
        print(f"{r['name']:<16} | {delta:>+15.1%}")

    results_path = args.output_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "mem_inference_gb": mem_inference,
            "total_steps": args.total_steps,
            "n_samples": args.n_samples,
            "temperature": args.temperature,
            "train_data": args.train_data,
            "experiments": results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
