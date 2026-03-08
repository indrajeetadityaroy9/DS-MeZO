"""DS-MeZO SFT evaluation: train on GSM8K, evaluate via perplexity.

Proves the core claim: zeroth-order SFT with AGZO + ZO-Muon improves
a real LLM's NLL on standard benchmarks without backpropagation.

Primary metric: perplexity on held-out GSM8K samples (direct NLL measurement).
Secondary metric: GSM8K exact-match (only meaningful for larger models).

Usage: python eval/sft_eval.py
"""

import os
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import sys
import json
import time

sys.path.insert(0, "/home/ubuntu/DS-MeZO")

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM

from ds_mezo.model_config import discover_layers
from ds_mezo.backend import VLLMBackend
from ds_mezo.controller import DSMeZO_Controller
from eval.benchmarks import eval_perplexity, eval_gsm8k


MODEL_PATH = "/dev/shm/pissa_prep/residual"
ADAPTER_PATH = "/dev/shm/pissa_prep/adapter"
TOTAL_STEPS = 2000
MAX_SEQ_LEN = 512
HELD_OUT_SAMPLES = 50


def prepare_sft_data(tokenizer, max_seq_len):
    """Load and tokenize GSM8K train split for SFT.

    Returns (train_data, held_out_data) where held_out is used for
    perplexity evaluation.
    """
    dataset = load_dataset("openai/gsm8k", "main", split="train")

    all_data = []
    for row in dataset:
        prompt = f"Question: {row['question']}\nAnswer: Let's solve step by step.\n"
        completion = row["answer"]
        full_text = prompt + completion
        full_ids = tokenizer.encode(full_text, add_special_tokens=True)[:max_seq_len]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_len = min(len(prompt_ids), len(full_ids) - 1)
        all_data.append({
            "token_ids": full_ids,
            "prompt_len": prompt_len,
            "prompt_text": prompt,
        })

    # Split: last HELD_OUT_SAMPLES for evaluation
    train_data = all_data[:-HELD_OUT_SAMPLES]
    held_out = all_data[-HELD_OUT_SAMPLES:]
    return train_data, held_out


def main():
    print("=" * 70)
    print("DS-MeZO SFT EVALUATION: GSM8K")
    print(f"Steps: {TOTAL_STEPS} | Max seq len: {MAX_SEQ_LEN}")
    print("=" * 70)

    # Load tokenizer and prepare data
    print("\nPreparing SFT data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    train_data, held_out = prepare_sft_data(tokenizer, MAX_SEQ_LEN)
    print(f"Training samples: {len(train_data)}")
    print(f"Held-out samples: {len(held_out)}")
    avg_seq = sum(len(s['token_ids']) for s in train_data) / len(train_data)
    avg_comp = sum(len(s['token_ids']) - s['prompt_len'] for s in train_data) / len(train_data)
    print(f"Avg sequence length: {avg_seq:.0f}")
    print(f"Avg completion length: {avg_comp:.0f}")

    # Load vLLM engine
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

    # Initialize controller in SFT mode
    print("Initializing controller (SFT mode)...")
    layer_specs = discover_layers(MODEL_PATH, ["q_proj", "v_proj"])
    backend = VLLMBackend(llm, layer_specs, 16)
    controller = DSMeZO_Controller(backend, layer_specs, {
        "mode": "sft",
        "adapter_path": ADAPTER_PATH,
        "total_steps": TOTAL_STEPS,
        "eta_max": 1e-2,
        "lambda_gr": 0.0,
    })
    print(f"Layers: {controller.num_layers} | Mode: {controller.mode}")

    # Held-out data for perplexity eval
    held_out_ids = [s["token_ids"] for s in held_out]
    held_out_plens = [s["prompt_len"] for s in held_out]

    # Pre-training baselines (with initial PiSSA adapter for fair comparison)
    print("\n--- Pre-training baselines ---")
    controller._sync_adapters({}, {})  # sync initial adapter weights
    baseline_ppl = eval_perplexity(llm, held_out_ids, held_out_plens,
                                   lora_request=backend.lora_pos)
    print(f"  Held-out perplexity: {baseline_ppl['perplexity']:.2f} "
          f"(NLL: {baseline_ppl['avg_nll']:.4f}, {baseline_ppl['total_tokens']} tokens)")

    baseline_gsm8k = eval_gsm8k(llm, lora_request=backend.lora_pos, num_samples=100)
    print(f"  GSM8K exact_match:   {baseline_gsm8k['exact_match']:.1%} "
          f"({baseline_gsm8k['num_parsed']}/{baseline_gsm8k['num_samples']} parsed)")

    # Train
    print(f"\n--- SFT Training ({TOTAL_STEPS} steps) ---")

    t_start = time.time()
    log = []
    for step_idx in range(TOTAL_STEPS):
        sample = train_data[step_idx % len(train_data)]
        controller.step(sample)

        entry = {
            "step": step_idx + 1,
            "loss_ema": controller.loss_ema,
            "eta": controller.eta,
            "eps": controller.eps,
        }
        log.append(entry)

        if (step_idx + 1) % 200 == 0:
            elapsed = time.time() - t_start
            s_per_step = elapsed / (step_idx + 1)
            eta_remain = s_per_step * (TOTAL_STEPS - step_idx - 1)
            loss_str = f"{controller.loss_ema:.4f}" if controller.loss_ema else "N/A"
            print(f"  step {step_idx+1:4d}/{TOTAL_STEPS} | "
                  f"loss_ema={loss_str} | "
                  f"lr={controller.eta:.2e} | "
                  f"{s_per_step:.1f}s/step | "
                  f"ETA {eta_remain:.0f}s")

    train_time = time.time() - t_start
    print(f"\nTraining complete: {train_time:.1f}s total, {train_time/TOTAL_STEPS:.1f}s/step")

    # Post-training evaluation
    print("\n--- Post-training evaluation ---")
    controller._sync_adapters({}, {})

    post_ppl = eval_perplexity(llm, held_out_ids, held_out_plens,
                               lora_request=backend.lora_pos)
    print(f"  Held-out perplexity: {post_ppl['perplexity']:.2f} "
          f"(NLL: {post_ppl['avg_nll']:.4f})")

    post_gsm8k = eval_gsm8k(llm, lora_request=backend.lora_pos, num_samples=100)
    print(f"  GSM8K exact_match:   {post_gsm8k['exact_match']:.1%} "
          f"({post_gsm8k['num_parsed']}/{post_gsm8k['num_samples']} parsed)")

    # Summary
    print("\n" + "=" * 70)
    print("SFT EVALUATION SUMMARY")
    print("=" * 70)
    ppl_delta = post_ppl['perplexity'] - baseline_ppl['perplexity']
    ppl_pct = (ppl_delta / baseline_ppl['perplexity']) * 100
    print(f"  Perplexity baseline:  {baseline_ppl['perplexity']:.2f}")
    print(f"  Perplexity post-SFT:  {post_ppl['perplexity']:.2f}")
    print(f"  Perplexity delta:     {ppl_delta:+.2f} ({ppl_pct:+.1f}%)")
    print(f"  GSM8K baseline:       {baseline_gsm8k['exact_match']:.1%}")
    print(f"  GSM8K post-training:  {post_gsm8k['exact_match']:.1%}")
    if controller.loss_ema and controller.initial_loss_ema:
        print(f"  Initial loss EMA:     {controller.initial_loss_ema:.4f}")
        print(f"  Final loss EMA:       {controller.loss_ema:.4f}")
    print(f"  Training time:        {train_time:.1f}s ({train_time/TOTAL_STEPS:.1f}s/step)")
    print(f"  Steps completed:      {TOTAL_STEPS}")

    # Save results
    results = {
        "baseline_ppl": baseline_ppl,
        "post_ppl": post_ppl,
        "baseline_gsm8k": baseline_gsm8k,
        "post_gsm8k": post_gsm8k,
        "training_log": log,
        "train_time_seconds": train_time,
        "total_steps": TOTAL_STEPS,
        "mode": "sft",
    }
    log_path = "/home/ubuntu/DS-MeZO/eval/sft_results.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {log_path}")


if __name__ == "__main__":
    main()
