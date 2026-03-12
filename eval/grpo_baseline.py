"""GRPO baseline: backprop-based RL post-training via TRL GRPOTrainer."""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from vllm.lora.request import LoRARequest

import evaluate

from ds_mezo.backend import create_engine
from eval.benchmarks import eval_mbpp, load_mbpp_train
from eval.utils import extract_code


# ── Reward function (TRL interface) ─────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _get_code_eval():
    return evaluate.load("code_eval")


def mbpp_exec_reward(completions: list[str], test_list: list,
                     test_imports: list, **kwargs) -> list[float]:
    code_eval = _get_code_eval()
    scores = []
    for completion, tests, imports in zip(completions, test_list, test_imports):
        code = extract_code(completion)
        import_block = "\n".join(imports)
        references = [f"{import_block}\n{test}" for test in tests]
        predictions = [[code]] * len(references)
        _, results_dict = code_eval.compute(
            references=references, predictions=predictions,
            k=[1], num_workers=1, timeout=3.0,
        )
        passed = sum(
            1 for task_results in results_dict.values()
            for _, r in task_results if r["passed"]
        )
        scores.append(passed / len(tests))
    return scores


# ── Memory callback ──────────────────────────────────────────────────────────

class MemoryCallback(TrainerCallback):
    def __init__(self):
        self.peak_vram_mb = 0.0

    def on_step_end(self, args, state, control, **kwargs):
        free, total = torch.cuda.mem_get_info()
        used_mb = (total - free) / (1024 ** 2)
        self.peak_vram_mb = max(self.peak_vram_mb, used_mb)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO baseline (TRL)")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="PiSSA residual model (same as DS-MeZO)")
    parser.add_argument("--adapter-path", type=Path, required=True,
                        help="PiSSA adapter (same as DS-MeZO)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--dsmezo-results", type=Path, required=True,
                        help="Path to DS-MeZO rl_bench_results.json for comparison")
    args = parser.parse_args()

    model_name = args.model_path.name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Read rank and target modules from adapter config via PEFT
    peft_config = PeftConfig.from_pretrained(str(args.adapter_path))
    rank = peft_config.r
    target_modules = list(peft_config.target_modules)

    # Load training data (same as DS-MeZO)
    train_data = load_mbpp_train()
    dataset = Dataset.from_list(train_data)

    print("=" * 70)
    print("GRPO BASELINE (TRL GRPOTrainer)")
    print(f"Model: {model_name} | PiSSA rank-{rank}")
    print(f"Train: MBPP train ({len(train_data)} problems) | Steps: {args.total_steps}")
    print(f"Eval: MBPP pass@k (n={args.n_samples}, T={args.temperature})")
    print("=" * 70)

    # ── Pre-training baseline ────────────────────────────────────────────
    # Evaluate PiSSA init weights via vLLM (identical to DS-MeZO's pre-eval)
    print("\nLoading vLLM engine for pre-training eval...")
    t0 = time.time()

    eval_engine = create_engine(args.model_path, rank)
    print(f"Engine loaded in {time.time()-t0:.1f}s")

    lora_req = LoRARequest("pissa_init", 1, str(args.adapter_path))
    pre_mbpp = eval_mbpp(eval_engine, lora_request=lora_req,
                         n_samples=args.n_samples, temperature=args.temperature)
    print(f"\n--- Pre-training baseline ---")
    ci = pre_mbpp['pass@1_ci95']
    print(f"  MBPP pass@1: {pre_mbpp['pass@1']:.1%} (95% CI: {ci[0]:.1%}–{ci[1]:.1%}, {pre_mbpp['num_tasks']} tasks)")
    ci10 = pre_mbpp['pass@10_ci95']
    print(f"  MBPP pass@10: {pre_mbpp['pass@10']:.1%} (95% CI: {ci10[0]:.1%}–{ci10[1]:.1%})")

    torch.cuda.empty_cache()

    print("\nLoading model for GRPO training...")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    base_model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path), torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, str(args.adapter_path))

    training_args = GRPOConfig(
        output_dir=str(args.output_dir / "grpo_checkpoints"),
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        max_prompt_length=512,
        max_completion_length=512,
        max_steps=args.total_steps,
        num_generations=4,
        per_device_train_batch_size=4,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=100,
        save_strategy="no",
        seed=42,
        report_to="none",
        remove_unused_columns=False,
    )

    mem_callback = MemoryCallback()
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=mbpp_exec_reward,
        args=training_args,
        train_dataset=dataset,
        callbacks=[mem_callback],
    )

    free, total = torch.cuda.mem_get_info()
    mem_before_gb = (total - free) / (1024 ** 3)

    print(f"\n--- Training ({args.total_steps} steps) ---")
    t_start = time.time()
    trainer.train()
    train_time = time.time() - t_start
    peak_vram_gb = mem_callback.peak_vram_mb / 1024

    print(f"\nTraining complete: {train_time:.1f}s ({train_time/args.total_steps:.1f}s/step)")
    print(f"  VRAM before training: {mem_before_gb:.1f} GB")
    print(f"  Peak VRAM:            {peak_vram_gb:.1f} GB")

    adapter_save_path = args.output_dir / "trained_adapter"
    trainer.model.save_pretrained(str(adapter_save_path))
    torch.cuda.empty_cache()

    print("\nLoading vLLM engine for post-training eval...")

    eval_engine = create_engine(args.model_path, rank)
    lora_req = LoRARequest("grpo_trained", 1, str(adapter_save_path))
    post_mbpp = eval_mbpp(eval_engine, lora_request=lora_req,
                          n_samples=args.n_samples, temperature=args.temperature)

    print(f"\n--- Post-training evaluation ---")
    ci = post_mbpp['pass@1_ci95']
    print(f"  MBPP pass@1: {post_mbpp['pass@1']:.1%} (95% CI: {ci[0]:.1%}–{ci[1]:.1%}, {post_mbpp['num_tasks']} tasks)")
    ci10 = post_mbpp['pass@10_ci95']
    print(f"  MBPP pass@10: {post_mbpp['pass@10']:.1%} (95% CI: {ci10[0]:.1%}–{ci10[1]:.1%})")

    delta_1 = post_mbpp["pass@1"] - pre_mbpp["pass@1"]

    delta_10 = post_mbpp["pass@10"] - pre_mbpp["pass@10"]
    print(f"\n  pass@1: {pre_mbpp['pass@1']:.1%} → {post_mbpp['pass@1']:.1%} ({delta_1:+.1%})")
    print(f"  pass@10: {pre_mbpp['pass@10']:.1%} → {post_mbpp['pass@10']:.1%} ({delta_10:+.1%})")
    print(f"  VRAM: {peak_vram_gb:.1f} GB | Time: {train_time:.1f}s")

    results = {
        "method": "grpo",
        "model": model_name,
        "rank": rank,
        "train_steps": args.total_steps,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "pre_mbpp": pre_mbpp,
        "post_mbpp": post_mbpp,
        "delta_pass@1": delta_1,
        "train_time": train_time,
        "peak_vram_gb": peak_vram_gb,
        "mem_before_gb": mem_before_gb,
    }
    results_path = args.output_dir / "grpo_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {results_path}")

    dsmezo = json.loads(args.dsmezo_results.read_text())
    print("\n" + "=" * 70)
    print("COMPARATIVE RESULTS: DS-MeZO vs GRPO")
    print("=" * 70)
    print(f"{'Method':<20} | {'pass@1 pre':>10} | {'pass@1 post':>11} | "
          f"{'Delta':>7} | {'Time':>7} | {'Peak VRAM':>9}")
    print("-" * 78)
    # DS-MeZO row
    dz_pre = dsmezo["pre_mbpp"]["pass@1"]
    dz_post = dsmezo["post_mbpp"]["pass@1"]
    dz_delta = dsmezo["delta_pass@1"]
    dz_time = dsmezo["train_time"]
    print(f"{'DS-MeZO (ZO)':<20} | {dz_pre:>9.1%} | {dz_post:>10.1%} | "
          f"{dz_delta:>+6.1%} | {dz_time:>6.0f}s | {'~17 GB':>9}")
    # GRPO row
    print(f"{'GRPO (backprop)':<20} | {pre_mbpp['pass@1']:>9.1%} | "
          f"{post_mbpp['pass@1']:>10.1%} | {delta_1:>+6.1%} | "
          f"{train_time:>6.0f}s | {peak_vram_gb:>7.1f} GB")


if __name__ == "__main__":
    main()
