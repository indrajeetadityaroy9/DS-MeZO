import os
import subprocess

os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())

import json
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig

from ds_mezo import build_controller
from eval.benchmarks import eval_mbpp, eval_humaneval
from eval.rewards import load_mbpp_train, load_apps_train, make_exec_reward


def _lock_gpu_clocks():
    subprocess.run(["sudo", "nvidia-smi", "-pm", "1"], check=True,
                   capture_output=True)
    max_gfx = subprocess.run(
        ["nvidia-smi", "--query-supported-clocks=graphics",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split("\n")
    max_mem = subprocess.run(
        ["nvidia-smi", "--query-supported-clocks=memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split("\n")
    gfx = max(int(x.strip()) for x in max_gfx)
    mem = max(int(x.strip()) for x in max_mem)
    subprocess.run(["sudo", "nvidia-smi", "-lgc", f"{gfx},{gfx}"],
                   check=True, capture_output=True)
    subprocess.run(["sudo", "nvidia-smi", "-lmc", str(mem)],
                   check=True, capture_output=True)


def _run_full_eval(engine, lora_request, eval_cfg):
    mbpp = eval_mbpp(engine, lora_request=lora_request,
                     n_samples=eval_cfg.n_samples, temperature=eval_cfg.temperature)
    humaneval = eval_humaneval(engine, lora_request=lora_request,
                               n_samples=eval_cfg.n_samples, temperature=eval_cfg.temperature)
    return {"mbpp": mbpp, "humaneval": humaneval}


def _print_eval(results, label):
    print(f"\n--- {label} ---")
    mbpp = results["mbpp"]
    ci = mbpp["pass@1_ci95"]
    print(f"  MBPP pass@1: {mbpp['pass@1']:.1%} (95% CI: {ci[0]:.1%}\u2013{ci[1]:.1%}, {mbpp['num_tasks']} tasks)")
    if "pass@10" in mbpp:
        ci10 = mbpp["pass@10_ci95"]
        print(f"  MBPP pass@10: {mbpp['pass@10']:.1%} (95% CI: {ci10[0]:.1%}\u2013{ci10[1]:.1%})")
    he = results["humaneval"]
    ci = he["pass@1_ci95"]
    print(f"  HumanEval pass@1: {he['pass@1']:.1%} (95% CI: {ci[0]:.1%}\u2013{ci[1]:.1%}, {he['num_tasks']} tasks)")


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    if cfg.system.lock_clocks:
        _lock_gpu_clocks()

    problems = load_apps_train() if cfg.data.train_data == "apps" else load_mbpp_train()

    reward, set_problem = make_exec_reward()
    controller, engine = build_controller(cfg, score_fn=reward)
    backend = controller.backend
    rank = controller.layers[0].B.shape[0]
    total_steps = cfg.training.total_steps
    model_name = Path(cfg.model.path).name

    print("=" * 70)
    print("DS-MeZO RL PROOF-OF-CONCEPT")
    print(f"Model: {model_name} | PiSSA rank-{rank}")
    print(f"Train: {len(problems)} {cfg.data.train_data} problems | Steps: {total_steps}")
    print(f"Eval: MBPP + HumanEval (n={cfg.eval.n_samples}, T={cfg.eval.temperature})")
    if cfg.eval.eval_at_steps:
        print(f"Scaling checkpoints: {sorted(cfg.eval.eval_at_steps)}")
    print("=" * 70)

    backend.sync_adapters({}, {}, controller.layers)
    pre_results = _run_full_eval(engine, backend.lora_pos, cfg.eval)
    _print_eval(pre_results, "Pre-training baseline")

    print(f"\n--- Training ({total_steps} steps, {len(problems)} problems) ---")
    t_start = time.time()
    log = []
    scaling_checkpoints = []
    eval_at = set(cfg.eval.eval_at_steps)
    log_interval = total_steps // 100
    ckpt_interval = total_steps // 10

    for step_idx in range(controller.step_count, total_steps):
        problem = problems[step_idx % len(problems)]
        set_problem(problem["test_list"], problem["test_imports"])
        controller.step([problem["prompt"]])

        log.append({"step": step_idx + 1, "eta": controller.eta, "eps": controller.eps})

        if (step_idx + 1) % log_interval == 0:
            print(f"  step {step_idx+1}/{total_steps} | lr={controller.eta:.2e}")

        if (step_idx + 1) % ckpt_interval == 0:
            controller._save_checkpoint(step_idx + 1)

        if (step_idx + 1) in eval_at:
            print(f"\n  [Scaling checkpoint at step {step_idx+1}]")
            backend.sync_adapters({}, {}, controller.layers)
            checkpoint_results = _run_full_eval(engine, backend.lora_pos, cfg.eval)
            scaling_checkpoints.append({"step": step_idx + 1, "results": checkpoint_results})
            _print_eval(checkpoint_results, f"Step {step_idx+1}")

    train_time = time.time() - t_start
    controller._save_checkpoint(total_steps)
    print(f"\nTraining complete: {train_time:.1f}s ({train_time/total_steps:.1f}s/step)")

    backend.sync_adapters({}, {}, controller.layers)
    post_results = _run_full_eval(engine, backend.lora_pos, cfg.eval)
    _print_eval(post_results, "Post-training")

    delta_1 = post_results["mbpp"]["pass@1"] - pre_results["mbpp"]["pass@1"]
    print(f"\n  MBPP pass@1: {pre_results['mbpp']['pass@1']:.1%} \u2192 {post_results['mbpp']['pass@1']:.1%} ({delta_1:+.1%})")
    print(f"  HumanEval pass@1: {pre_results['humaneval']['pass@1']:.1%} \u2192 {post_results['humaneval']['pass@1']:.1%}")
    print(f"  Time: {train_time:.1f}s ({train_time/total_steps:.1f}s/step)")
    print(f"  Total forward passes: {backend.query_count}")

    results_path = Path(cfg.output_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": model_name,
            "rank": rank,
            "train_steps": total_steps,
            "train_data": cfg.data.train_data,
            "train_data_size": len(problems),
            "n_samples": cfg.eval.n_samples,
            "temperature": cfg.eval.temperature,
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
