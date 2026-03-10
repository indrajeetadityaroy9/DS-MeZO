"""DS-MeZO RL proof-of-concept: train on MBPP, evaluate pass@k."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftConfig
from vllm import LLM

from ds_mezo.model_config import discover_layers
from ds_mezo.backend import VLLMBackend
from ds_mezo.controller import DSMeZO_Controller
from eval.benchmarks import eval_mbpp, load_mbpp_train
from eval.utils import make_exec_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="DS-MeZO RL proof-of-concept")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    model_name = args.model_path.name

    args.output_dir.mkdir(parents=True, exist_ok=True)

    peft_config = PeftConfig.from_pretrained(str(args.adapter_path))
    rank = peft_config.r
    target_modules = list(peft_config.target_modules)

    print("=" * 70)
    print("DS-MeZO RL PROOF-OF-CONCEPT")
    print(f"Model: {model_name} | PiSSA rank-{rank}")
    print(f"Train: MBPP train | Eval: MBPP pass@k (n={args.n_samples}, T={args.temperature})")
    print("=" * 70)

    train_data = load_mbpp_train()

    print("\nLoading vLLM engine...")
    t0 = time.time()
    llm = LLM(
        model=str(args.model_path),
        dtype="bfloat16",
        gpu_memory_utilization=0.95,
        enable_lora=True,
        max_lora_rank=max(64, rank),
        enforce_eager=True,
    )
    print(f"Engine loaded in {time.time()-t0:.1f}s")

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

    print("\n--- Pre-training baseline ---")
    controller.backend.sync_adapters({}, {}, controller.layers)
    pre_mbpp = eval_mbpp(llm, lora_request=backend.lora_pos, n_samples=args.n_samples, temperature=args.temperature)
    ci = pre_mbpp['pass@1_ci95']
    print(f"  MBPP pass@1: {pre_mbpp['pass@1']:.1%} (95% CI: {ci[0]:.1%}–{ci[1]:.1%}, {pre_mbpp['num_tasks']} tasks)")
    ci10 = pre_mbpp['pass@10_ci95']
    print(f"  MBPP pass@10: {pre_mbpp['pass@10']:.1%} (95% CI: {ci10[0]:.1%}–{ci10[1]:.1%})")

    total_steps = controller.total_steps
    print(f"\n--- Training ({total_steps} steps, {len(train_data)} problems) ---")
    t_start = time.time()
    log = []
    for step_idx in range(total_steps):
        problem = train_data[step_idx % len(train_data)]
        set_problem(problem["test_list"], problem["test_imports"])
        controller.step([problem["prompt"]])

        log.append({
            "step": step_idx + 1,
            "eta": controller.eta,
            "eps": controller.eps,
        })

        if (step_idx + 1) % 100 == 0:
            print(f"  step {step_idx+1}/{total_steps} | lr={controller.eta:.2e}")

    train_time = time.time() - t_start
    print(f"\nTraining complete: {train_time:.1f}s ({train_time/total_steps:.1f}s/step)")

    print("\n--- Post-training ---")
    controller.backend.sync_adapters({}, {}, controller.layers)
    post_mbpp = eval_mbpp(llm, lora_request=backend.lora_pos, n_samples=args.n_samples, temperature=args.temperature)
    ci = post_mbpp['pass@1_ci95']
    print(f"  MBPP pass@1: {post_mbpp['pass@1']:.1%} (95% CI: {ci[0]:.1%}–{ci[1]:.1%}, {post_mbpp['num_tasks']} tasks)")
    ci10 = post_mbpp['pass@10_ci95']
    print(f"  MBPP pass@10: {post_mbpp['pass@10']:.1%} (95% CI: {ci10[0]:.1%}–{ci10[1]:.1%})")

    delta_1 = post_mbpp['pass@1'] - pre_mbpp['pass@1']
    print(f"\n  pass@1: {pre_mbpp['pass@1']:.1%} → {post_mbpp['pass@1']:.1%} ({delta_1:+.1%})")
    print(f"  pass@10: {pre_mbpp['pass@10']:.1%} → {post_mbpp['pass@10']:.1%}")
    print(f"  Time: {train_time:.1f}s ({train_time/total_steps:.1f}s/step)")

    results_path = args.output_dir / "rl_bench_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": model_name,
            "rank": rank,
            "train_steps": total_steps,
            "n_samples": args.n_samples,
            "temperature": args.temperature,
            "pre_mbpp": pre_mbpp,
            "post_mbpp": post_mbpp,
            "delta_pass@1": delta_1,
            "train_time": train_time,
            "training_log": log,
        }, f, indent=2, default=str)
    print(f"  Results saved to {results_path}")


if __name__ == "__main__":
    main()
