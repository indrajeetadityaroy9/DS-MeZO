"""DS-MeZO RL proof-of-concept: train on MBPP/APPS, evaluate pass@k on
MBPP + HumanEval + SST-2 + RTE. Supports mid-loop scaling curve evaluation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval.benchmarks import (
    eval_mbpp, eval_humaneval, eval_sst2, eval_rte,
    load_mbpp_train, load_apps_train,
    make_exec_reward, setup_controller,
)


def _run_full_eval(llm, lora_request, n_samples, temperature):
    """Run all benchmarks and return combined results dict."""
    mbpp = eval_mbpp(llm, lora_request=lora_request, n_samples=n_samples, temperature=temperature)
    humaneval = eval_humaneval(llm, lora_request=lora_request, n_samples=n_samples, temperature=temperature)
    sst2 = eval_sst2(llm, lora_request=lora_request, n_shot=8)
    rte = eval_rte(llm, lora_request=lora_request, n_shot=8)
    return {"mbpp": mbpp, "humaneval": humaneval, "sst2": sst2, "rte": rte}


def _print_eval(results, label):
    """Print evaluation results."""
    print(f"\n--- {label} ---")
    mbpp = results["mbpp"]
    ci = mbpp['pass@1_ci95']
    print(f"  MBPP pass@1: {mbpp['pass@1']:.1%} (95% CI: {ci[0]:.1%}–{ci[1]:.1%}, {mbpp['num_tasks']} tasks)")
    if "pass@10" in mbpp:
        ci10 = mbpp['pass@10_ci95']
        print(f"  MBPP pass@10: {mbpp['pass@10']:.1%} (95% CI: {ci10[0]:.1%}–{ci10[1]:.1%})")
    he = results["humaneval"]
    ci = he['pass@1_ci95']
    print(f"  HumanEval pass@1: {he['pass@1']:.1%} (95% CI: {ci[0]:.1%}–{ci[1]:.1%}, {he['num_tasks']} tasks)")
    sst2 = results["sst2"]
    ci = sst2['accuracy_ci95']
    print(f"  SST-2 accuracy: {sst2['accuracy']:.1%} (95% CI: {ci[0]:.1%}–{ci[1]:.1%})")
    rte = results["rte"]
    ci = rte['accuracy_ci95']
    print(f"  RTE accuracy: {rte['accuracy']:.1%} (95% CI: {ci[0]:.1%}–{ci[1]:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="DS-MeZO RL proof-of-concept")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--train-data", choices=["mbpp", "apps"], default="mbpp")
    parser.add_argument("--eval-at-steps", type=int, nargs="+", default=None,
                        help="Evaluate at these training steps for scaling curves")
    args = parser.parse_args()

    model_name = args.model_path.name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.train_data == "apps":
        train_data = load_apps_train()
    else:
        train_data = load_mbpp_train()

    total_steps = args.total_steps
    eval_at_steps = set()
    if args.eval_at_steps:
        eval_at_steps = set(args.eval_at_steps)
        total_steps = max(max(eval_at_steps), total_steps)

    reward, set_problem = make_exec_reward()
    llm, backend, controller, rank, _ = setup_controller(
        args.model_path, args.adapter_path, args.output_dir, total_steps,
        score_fn=reward, calibration_prompt=train_data[0]["prompt"],
    )

    print("=" * 70)
    print("DS-MeZO RL PROOF-OF-CONCEPT")
    print(f"Model: {model_name} | PiSSA rank-{rank}")
    print(f"Train: {len(train_data)} {args.train_data} problems | Steps: {total_steps}")
    print(f"Eval: MBPP + HumanEval + SST-2 + RTE (n={args.n_samples}, T={args.temperature})")
    if eval_at_steps:
        print(f"Scaling checkpoints: {sorted(eval_at_steps)}")
    print("=" * 70)

    controller.backend.sync_adapters({}, {}, controller.layers)
    pre_results = _run_full_eval(llm, backend.lora_pos, args.n_samples, args.temperature)
    _print_eval(pre_results, "Pre-training baseline")

    print(f"\n--- Training ({total_steps} steps, {len(train_data)} problems) ---")
    t_start = time.time()
    log = []
    scaling_checkpoints = []
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

        if (step_idx + 1) in eval_at_steps:
            print(f"\n  [Scaling checkpoint at step {step_idx+1}]")
            controller.backend.sync_adapters({}, {}, controller.layers)
            checkpoint_results = _run_full_eval(
                llm, backend.lora_pos, args.n_samples, args.temperature,
            )
            scaling_checkpoints.append({
                "step": step_idx + 1,
                "results": checkpoint_results,
            })
            _print_eval(checkpoint_results, f"Step {step_idx+1}")

    train_time = time.time() - t_start
    print(f"\nTraining complete: {train_time:.1f}s ({train_time/total_steps:.1f}s/step)")

    controller.backend.sync_adapters({}, {}, controller.layers)
    post_results = _run_full_eval(llm, backend.lora_pos, args.n_samples, args.temperature)
    _print_eval(post_results, "Post-training")

    delta_1 = post_results["mbpp"]['pass@1'] - pre_results["mbpp"]['pass@1']
    print(f"\n  MBPP pass@1: {pre_results['mbpp']['pass@1']:.1%} → {post_results['mbpp']['pass@1']:.1%} ({delta_1:+.1%})")
    print(f"  HumanEval pass@1: {pre_results['humaneval']['pass@1']:.1%} → {post_results['humaneval']['pass@1']:.1%}")
    print(f"  SST-2: {pre_results['sst2']['accuracy']:.1%} → {post_results['sst2']['accuracy']:.1%}")
    print(f"  RTE: {pre_results['rte']['accuracy']:.1%} → {post_results['rte']['accuracy']:.1%}")
    print(f"  Time: {train_time:.1f}s ({train_time/total_steps:.1f}s/step)")
    print(f"  Total forward passes: {backend.query_count}")

    results_path = args.output_dir / "rl_bench_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": model_name,
            "rank": rank,
            "train_steps": total_steps,
            "train_data": args.train_data,
            "train_data_size": len(train_data),
            "n_samples": args.n_samples,
            "temperature": args.temperature,
            "pre": pre_results,
            "post": post_results,
            "delta_mbpp_pass@1": delta_1,
            "train_time": train_time,
            "query_count": backend.query_count,
            "scaling_checkpoints": scaling_checkpoints,
            "training_log": log,
        }, f, indent=2, default=str)
    print(f"  Results saved to {results_path}")


if __name__ == "__main__":
    main()
