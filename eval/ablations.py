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
from peft import PeftConfig

from ds_mezo.model_config import LayerSpec, discover_layers
from ds_mezo.backend import VLLMBackend, create_engine
from ds_mezo.controller import DSMeZO_Controller, LayerState
from eval.utils import make_exec_reward
from eval.benchmarks import eval_mbpp, load_mbpp_train, load_apps_train


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
    config = {
        "output_dir": str(experiment_dir),
        "adapter_path": str(adapter_path),
        "total_steps": total_steps,
        "score_fn": reward,
    }

    backend = VLLMBackend(llm, layer_specs, rank)
    controller = DSMeZO_Controller(backend, layer_specs, config)

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
    """Replace ZO-Muon (N-S orthogonalization) with plain SGD+momentum."""

    def patched_step(self: DSMeZO_Controller, batch: list[str]) -> None:
        self.step_count += 1

        trajectories, advantages, prompt_len = self._explore(batch)

        batch_activations = self.backend.extract_activations(batch)
        self._update_activation_bases(batch_activations)

        perturbations: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in self.layers:
            perturbations[layer.key] = self._get_perturbation(layer)

        pos_layers: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        neg_layers: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in self.layers:
            z_A, z_B = perturbations[layer.key]
            pos_A = layer.A + z_A
            neg_A = layer.A - z_A
            pos_B = layer.B + z_B
            neg_B = layer.B - z_B
            pos_layers[layer.key] = (pos_A, pos_B)
            neg_layers[layer.key] = (neg_A, neg_B)

        self.backend.sync_adapters(pos_layers, neg_layers, self.layers)
        loss_pos, loss_neg = self._score_contrastive(trajectories, advantages, prompt_len)

        dd = float(loss_pos - loss_neg) / (2.0 * self.eps)

        max_window = int(math.sqrt(self.total_steps))
        mom = 1.0 - 1.0 / min(self.step_count, max_window)

        for layer in self.layers:
            z_A, z_B = perturbations[layer.key]
            key_A = (layer.key, "A")
            key_B = (layer.key, "B")
            grad_A = dd * z_A
            self.momentum_buffers[key_A].mul_(mom).add_((1 - mom) * grad_A)
            layer.A.sub_(self.eta * self.momentum_buffers[key_A])

            grad_B = dd * z_B
            self.momentum_buffers[key_B].mul_(mom).add_((1 - mom) * grad_B)
            layer.B.sub_(self.eta * self.momentum_buffers[key_B])

        self.lr_scheduler.step()
        self.eta = self._lr_opt.param_groups[0]["lr"]

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
    controller._update_activation_bases = types.MethodType(
        lambda self, activations: None, controller
    )


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

    peft_config = PeftConfig.from_pretrained(str(args.adapter_path))
    rank = peft_config.r
    target_modules = list(peft_config.target_modules)

    if args.train_data == "apps":
        train_data = load_apps_train()
    else:
        train_data = load_mbpp_train()

    experiments = EXPERIMENTS
    if args.experiments:
        exp_names = set(args.experiments)
        experiments = [e for e in EXPERIMENTS if e["name"] in exp_names]

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
