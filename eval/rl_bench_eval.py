"""DS-MeZO RL proof-of-concept: train on MBPP, evaluate on MBPP pass@1.

Single experiment — DS-MeZO full system on Llama-3.1-8B.
Proves zeroth-order RL post-training improves code generation
on a standard benchmark without backpropagation.

Usage: python eval/rl_bench_eval.py
"""

import os
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import sys
import json
import time
import re
import torch

sys.path.insert(0, "/home/ubuntu/DS-MeZO")

from datasets import load_dataset
from vllm import LLM

from ds_mezo.model_config import discover_layers
from ds_mezo.backend import VLLMBackend
from ds_mezo.controller import DSMeZO_Controller
from eval.benchmarks import eval_mbpp

MODEL_PATH = "/dev/shm/pissa_prep/residual"
ADAPTER_PATH = "/dev/shm/pissa_prep/adapter"

# ---------------------------------------------------------------------------
# Execution-based RL reward
# ---------------------------------------------------------------------------

_current_tests = []
_current_imports = []


def extract_code(text):
    """Extract Python code from markdown blocks."""
    if "```python" in text:
        return text.split("```python")[1].split("```")[0]
    if "```" in text:
        return text.split("```")[1].split("```")[0]
    return text


def exec_reward(text):
    """Reward = fraction of test assertions passing."""
    code = extract_code(text)
    imports = "\n".join(_current_imports)
    passed = 0
    for test in _current_tests:
        try:
            exec(f"{imports}\n{code}\n{test}", {})
            passed += 1
        except Exception:
            pass
    return passed / len(_current_tests)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _current_tests, _current_imports

    print("=" * 70)
    print("DS-MeZO RL PROOF-OF-CONCEPT")
    print("Model: Llama-3.1-8B | PiSSA rank-16 | Single H100")
    print("Train: MBPP train | Eval: MBPP test pass@1")
    print("=" * 70)

    # Load training data
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    train_data = []
    for row in dataset:
        func_match = re.search(r'(?:assert\s+(?:not\s+)?(?:\()?\s*)(\w+)\s*\(', row['test_list'][0])
        func_name = func_match.group(1) if func_match else "solution"
        train_data.append({
            "prompt": (f"Write a Python function named `{func_name}`.\n\n"
                       f"{row['prompt']}\n\n```python\n"),
            "test_list": row["test_list"],
            "test_imports": row["test_imports"],
        })
    print(f"Training data: {len(train_data)} MBPP problems")

    # Load engine
    print("\nLoading vLLM engine...")
    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=0.92,
        enable_lora=True,
        max_lora_rank=64,
        max_num_seqs=8,
        enforce_eager=True,
    )
    print(f"Engine loaded in {time.time()-t0:.1f}s")

    # Init controller
    layer_specs = discover_layers(MODEL_PATH, ["q_proj", "v_proj"])
    torch.manual_seed(42)
    backend = VLLMBackend(llm, layer_specs, 16)
    controller = DSMeZO_Controller(backend, layer_specs, {
        "adapter_path": ADAPTER_PATH,
        "score_fn": exec_reward,
    })
    controller._calibrate_activation_bases_full([train_data[0]["prompt"]])
    print(f"Layers: {len(layer_specs)} | Params: {sum(l['A'].numel()+l['B'].numel() for l in controller.layers):,}")

    # Pre-training baseline
    print("\n--- Pre-training baseline ---")
    controller._sync_adapters({}, {})
    pre_mbpp = eval_mbpp(llm, lora_request=backend.lora_pos)
    print(f"  MBPP pass@1: {pre_mbpp['pass@1']:.1%} ({pre_mbpp['num_samples']} samples)")

    # Train (default 1000 steps, cycling through training data)
    total_steps = controller.total_steps
    print(f"\n--- Training ({total_steps} steps, {len(train_data)} problems) ---")
    t_start = time.time()
    log = []
    for step_idx in range(total_steps):
        problem = train_data[step_idx % len(train_data)]
        _current_tests = problem["test_list"]
        _current_imports = problem["test_imports"]
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
            loss_str = f"{controller.loss_ema:.4f}" if controller.loss_ema else "N/A"
            print(f"  step {step_idx+1:4d}/{total_steps} | "
                  f"loss_ema={loss_str} | "
                  f"lr={controller.eta:.2e} | "
                  f"{s_per_step:.1f}s/step")

    train_time = time.time() - t_start
    print(f"\nTraining complete: {train_time:.1f}s ({train_time/total_steps:.1f}s/step)")

    # Post-training eval
    print("\n--- Post-training evaluation ---")
    controller._sync_adapters({}, {})
    post_mbpp = eval_mbpp(llm, lora_request=backend.lora_pos)
    print(f"  MBPP pass@1: {post_mbpp['pass@1']:.1%} ({post_mbpp['num_samples']} samples)")

    # Summary
    delta = post_mbpp['pass@1'] - pre_mbpp['pass@1']
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  MBPP pre-training:  {pre_mbpp['pass@1']:.1%}")
    print(f"  MBPP post-training: {post_mbpp['pass@1']:.1%}")
    print(f"  Delta:              {delta:+.1%}")
    if controller.loss_ema and controller.initial_loss_ema:
        print(f"  Initial loss EMA:   {controller.initial_loss_ema:.4f}")
        print(f"  Final loss EMA:     {controller.loss_ema:.4f}")
    print(f"  Training time:      {train_time:.1f}s ({train_time/total_steps:.1f}s/step)")

    # Save
    log_path = "/home/ubuntu/DS-MeZO/eval/rl_bench_results.json"
    with open(log_path, "w") as f:
        json.dump({
            "model": "Llama-3.1-8B",
            "rank": 16,
            "train_steps": total_steps,
            "pre_mbpp": pre_mbpp,
            "post_mbpp": post_mbpp,
            "delta": delta,
            "train_time": train_time,
            "training_log": log,
        }, f, indent=2, default=str)
    print(f"  Results saved to {log_path}")


if __name__ == "__main__":
    main()
