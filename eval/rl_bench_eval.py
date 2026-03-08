"""DS-MeZO RL proof-of-concept: train on MBPP, evaluate on MBPP pass@k.

Single experiment — DS-MeZO full system. Model-agnostic.
Proves zeroth-order RL post-training improves code generation
on a standard benchmark without backpropagation.

Usage: python -m eval.rl_bench_eval --model-path <path> --adapter-path <path> --output-dir <path>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vllm import LLM

from ds_mezo.model_config import discover_layers
from ds_mezo.backend import VLLMBackend
from ds_mezo.controller import DSMeZO_Controller
from eval.benchmarks import eval_mbpp, load_mbpp_train
from eval.utils import ExecReward


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

    # Read rank and target modules from adapter config
    adapter_config = json.loads((args.adapter_path / "adapter_config.json").read_text())
    rank = adapter_config["r"]
    target_modules = adapter_config["target_modules"]

    print("=" * 70)
    print("DS-MeZO RL PROOF-OF-CONCEPT")
    print(f"Model: {model_name} | PiSSA rank-{rank}")
    print(f"Train: MBPP train | Eval: MBPP pass@k (n={args.n_samples}, T={args.temperature})")
    print("=" * 70)

    # Load training data
    train_data = load_mbpp_train()
    print(f"Training data: {len(train_data)} MBPP problems")

    # Load engine
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

    # Init controller
    reward = ExecReward()
    layer_specs = discover_layers(args.model_path, target_modules)
    backend = VLLMBackend(llm, layer_specs, rank)
    controller = DSMeZO_Controller(backend, layer_specs, {
        "output_dir": str(args.output_dir),
        "adapter_path": str(args.adapter_path),
        "model_path": str(args.model_path),
        "score_fn": reward,
        "total_steps": args.total_steps,
    })
    controller._calibrate_activation_bases_full([train_data[0]["prompt"]])
    print(f"Layers: {len(layer_specs)} | Params: {sum(l.A.numel()+l.B.numel() for l in controller.layers):,}")

    # Pre-training baseline
    print("\n--- Pre-training baseline ---")
    controller.backend.sync_adapters({}, {}, controller.layers)
    pre_mbpp = eval_mbpp(llm, lora_request=backend.lora_pos, n_samples=args.n_samples, temperature=args.temperature)
    print(f"  MBPP pass@1: {pre_mbpp['pass@1']:.1%} ({pre_mbpp['num_tasks']} tasks)")
    if "pass@10" in pre_mbpp:
        print(f"  MBPP pass@10: {pre_mbpp['pass@10']:.1%}")

    # Train
    total_steps = controller.total_steps
    print(f"\n--- Training ({total_steps} steps, {len(train_data)} problems) ---")
    t_start = time.time()
    log = []
    for step_idx in range(total_steps):
        problem = train_data[step_idx % len(train_data)]
        reward.set_problem(problem["test_list"], problem["test_imports"])
        controller.step([problem["prompt"]])

        log.append({
            "step": step_idx + 1,
            "loss_ema": controller.loss_ema,
            "eta": controller.eta,
            "eps": controller.eps,
        })

        if (step_idx + 1) % 100 == 0:
            elapsed = time.time() - t_start
            s_per_step = elapsed / (step_idx + 1)
            print(f"  step {step_idx+1:4d}/{total_steps} | "
                  f"loss_ema={controller.loss_ema:.4f} | "
                  f"lr={controller.eta:.2e} | "
                  f"{s_per_step:.1f}s/step")

    train_time = time.time() - t_start
    print(f"\nTraining complete: {train_time:.1f}s ({train_time/total_steps:.1f}s/step)")

    # Post-training eval
    print("\n--- Post-training evaluation ---")
    controller.backend.sync_adapters({}, {}, controller.layers)
    post_mbpp = eval_mbpp(llm, lora_request=backend.lora_pos, n_samples=args.n_samples, temperature=args.temperature)
    print(f"  MBPP pass@1: {post_mbpp['pass@1']:.1%} ({post_mbpp['num_tasks']} tasks)")
    if "pass@10" in post_mbpp:
        print(f"  MBPP pass@10: {post_mbpp['pass@10']:.1%}")

    # Summary
    delta_1 = post_mbpp['pass@1'] - pre_mbpp['pass@1']
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  MBPP pass@1 pre:    {pre_mbpp['pass@1']:.1%}")
    print(f"  MBPP pass@1 post:   {post_mbpp['pass@1']:.1%}")
    print(f"  Delta pass@1:       {delta_1:+.1%}")
    if "pass@10" in pre_mbpp and "pass@10" in post_mbpp:
        delta_10 = post_mbpp['pass@10'] - pre_mbpp['pass@10']
        print(f"  MBPP pass@10 pre:   {pre_mbpp['pass@10']:.1%}")
        print(f"  MBPP pass@10 post:  {post_mbpp['pass@10']:.1%}")
        print(f"  Delta pass@10:      {delta_10:+.1%}")
    print(f"  Initial loss EMA:   {controller.initial_loss_ema:.4f}")
    print(f"  Final loss EMA:     {controller.loss_ema:.4f}")
    print(f"  Training time:      {train_time:.1f}s ({train_time/total_steps:.1f}s/step)")

    # Save
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
