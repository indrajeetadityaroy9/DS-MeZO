"""DS-MeZO end-to-end evaluation: train Qwen2-0.5B on code generation.

Proves the core claim: zeroth-order optimization with AGZO + ZO-Muon
improves a real LLM on a measurable task without backpropagation.

Metric: score_fn = compilable Python output length (proxy for code quality).
Success criterion: score_ema increases over 100 training steps.
"""

import os
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import sys
import json
import time
import math

sys.path.insert(0, "/home/ubuntu/DS-MeZO")

from vllm import LLM
from ds_mezo.controller import DSMeZO_Controller


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


def score_fn(text):
    """Score = length of syntactically valid Python code.
    Returns 0 for non-compilable output, len(text) otherwise."""
    try:
        compile(text, "<eval>", "exec")
        return len(text)
    except SyntaxError:
        # Try extracting code block
        if "```python" in text:
            code = text.split("```python")[1].split("```")[0]
            try:
                compile(code, "<eval>", "exec")
                return len(code)
            except SyntaxError:
                pass
        if "```" in text:
            code = text.split("```")[1].split("```")[0]
            try:
                compile(code, "<eval>", "exec")
                return len(code)
            except SyntaxError:
                pass
        # Partial credit: length of text if it contains def/class keywords
        if "def " in text or "class " in text:
            return len(text) // 2
        return 0


def main():
    total_steps = 100

    config = {
        "model_path": "/dev/shm/pissa_prep/residual",
        "adapter_path": "/dev/shm/pissa_prep/adapter",
        "total_steps": total_steps,
        "score_fn": score_fn,
    }

    print("=" * 70)
    print("DS-MeZO EVALUATION: Qwen2-0.5B code generation")
    print(f"Steps: {total_steps} | Prompts: {len(PROMPTS)} | Rank: 16")
    print("=" * 70)

    print("\nLoading vLLM engine...")
    t0 = time.time()
    llm = LLM(
        model=config["model_path"],
        dtype="bfloat16",
        gpu_memory_utilization=0.92,
        enable_lora=True,
        max_lora_rank=64,
        max_num_seqs=8,
        enforce_eager=True,  # Required for activation extraction hooks
    )
    print(f"Engine loaded in {time.time()-t0:.1f}s")

    print("Initializing controller...")
    t0 = time.time()
    controller = DSMeZO_Controller(llm, config)
    print(f"Controller initialized in {time.time()-t0:.1f}s")
    print(f"Layers: {controller.num_layers} | "
          f"Params: {sum(l['A'].numel()+l['B'].numel() for l in controller.layers):,}")

    # Initial activation calibration (normally done in train())
    controller._calibrate_activation_bases_full([PROMPTS[0]])
    print(f"Activation bases calibrated: {len(controller.activation_bases)} layers")

    # --- Pre-training baseline ---
    print("\n--- Pre-training baseline ---")
    from vllm import SamplingParams
    baseline_scores = []
    baseline_outputs = llm.generate(
        PROMPTS[:5],
        SamplingParams(max_tokens=256, temperature=0.7),
    )
    for i, out in enumerate(baseline_outputs):
        text = out.outputs[0].text
        s = score_fn(text)
        baseline_scores.append(s)
        print(f"  [{i}] score={s:4d} | {text[:80]}...")
    baseline_avg = sum(baseline_scores) / len(baseline_scores)
    print(f"  Baseline avg score: {baseline_avg:.1f}")

    # --- Training ---
    print(f"\n--- Training ({total_steps} steps) ---")
    log = []
    t_start = time.time()

    # Override train() to capture per-step metrics
    for step_idx in range(total_steps):
        batch = [PROMPTS[step_idx % len(PROMPTS)]]
        controller.step(batch)

        entry = {
            "step": step_idx + 1,
            "loss_ema": controller.loss_ema,
            "eta": controller.eta,
            "eps": controller.eps,
            "temp": controller.explore_temperature,
            "dd_ema": controller.dd_ema,
        }
        log.append(entry)

        if (step_idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            s_per_step = elapsed / (step_idx + 1)
            eta_remain = s_per_step * (total_steps - step_idx - 1)
            loss_str = f"{controller.loss_ema:.4f}" if controller.loss_ema else "N/A"
            print(f"  step {step_idx+1:3d}/{total_steps} | "
                  f"loss_ema={loss_str} | "
                  f"lr={controller.eta:.2e} | "
                  f"eps={controller.eps:.2e} | "
                  f"temp={controller.explore_temperature:.3f} | "
                  f"{s_per_step:.1f}s/step | "
                  f"ETA {eta_remain:.0f}s")

    train_time = time.time() - t_start
    print(f"\nTraining complete: {train_time:.1f}s total, {train_time/total_steps:.1f}s/step")

    # --- Post-training evaluation ---
    print("\n--- Post-training evaluation ---")
    controller._sync_adapters({}, {})
    post_scores = []
    post_outputs = llm.generate(
        PROMPTS[:5],
        SamplingParams(max_tokens=256, temperature=0.7),
        lora_request=controller.lora_pos,
    )
    for i, out in enumerate(post_outputs):
        text = out.outputs[0].text
        s = score_fn(text)
        post_scores.append(s)
        print(f"  [{i}] score={s:4d} | {text[:80]}...")
    post_avg = sum(post_scores) / len(post_scores)
    print(f"  Post-training avg score: {post_avg:.1f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Baseline avg score:      {baseline_avg:.1f}")
    print(f"  Post-training avg score: {post_avg:.1f}")
    print(f"  Delta:                   {post_avg - baseline_avg:+.1f}")

    if controller.loss_ema and controller.initial_loss_ema:
        print(f"  Initial loss EMA:        {controller.initial_loss_ema:.4f}")
        print(f"  Final loss EMA:          {controller.loss_ema:.4f}")
        loss_delta = controller.loss_ema - controller.initial_loss_ema
        print(f"  Loss EMA delta:          {loss_delta:+.4f}")

    # Loss trajectory (first 10 vs last 10 steps)
    early_loss = [e["loss_ema"] for e in log[:10] if e["loss_ema"] is not None]
    late_loss = [e["loss_ema"] for e in log[-10:] if e["loss_ema"] is not None]
    if early_loss and late_loss:
        print(f"  Avg loss (steps 1-10):   {sum(early_loss)/len(early_loss):.4f}")
        print(f"  Avg loss (steps 91-100): {sum(late_loss)/len(late_loss):.4f}")

    print(f"  Training time:           {train_time:.1f}s ({train_time/total_steps:.1f}s/step)")
    print(f"  Steps completed:         {total_steps}")

    # Save log
    log_path = "/home/ubuntu/DS-MeZO/eval/results.json"
    with open(log_path, "w") as f:
        json.dump({
            "baseline_scores": baseline_scores,
            "post_scores": post_scores,
            "baseline_avg": baseline_avg,
            "post_avg": post_avg,
            "training_log": log,
            "train_time_seconds": train_time,
        }, f, indent=2, default=str)
    print(f"\n  Full log saved to {log_path}")


if __name__ == "__main__":
    main()
