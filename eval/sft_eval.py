"""DS-MeZO SFT evaluation: train on GSM8K, evaluate via perplexity.

Pure SFT mode: uses the unified step pipeline (AGZO + ZO-Muon) with NLL loss.
hybrid_switch_step = total_steps ensures no RL exploration.

Primary metric: perplexity on held-out GSM8K samples (direct NLL measurement).
Secondary metric: GSM8K exact-match (only meaningful for larger models).

Usage: python -m eval.sft_eval --model-path <path> --adapter-path <path>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM

from ds_mezo.model_config import discover_layers
from ds_mezo.backend import VLLMBackend
from ds_mezo.controller import DSMeZO_Controller
from eval.benchmarks import eval_perplexity, eval_gsm8k


def prepare_sft_data(
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    held_out_samples: int,
) -> tuple[list[dict], list[dict]]:
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

    train_data = all_data[:-held_out_samples]
    held_out = all_data[-held_out_samples:]
    return train_data, held_out


def main() -> None:
    parser = argparse.ArgumentParser(description="DS-MeZO SFT evaluation")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=2000)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--held-out-samples", type=int, default=50)
    args = parser.parse_args()

    model_name = args.model_path.name

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Read rank and target modules from adapter config
    adapter_config = json.loads((args.adapter_path / "adapter_config.json").read_text())
    rank = adapter_config["r"]
    target_modules = adapter_config["target_modules"]

    print("=" * 70)
    print("DS-MeZO SFT EVALUATION: GSM8K")
    print(f"Model: {model_name} | PiSSA rank-{rank}")
    print(f"Steps: {args.total_steps} | Max seq len: {args.max_seq_len}")
    print("=" * 70)

    # Load tokenizer and prepare data
    print("\nPreparing SFT data...")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    train_data, held_out = prepare_sft_data(
        tokenizer, args.max_seq_len, args.held_out_samples,
    )
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
        model=str(args.model_path),
        dtype="bfloat16",
        gpu_memory_utilization=0.92,
        enable_lora=True,
        max_lora_rank=max(64, rank),
        max_num_seqs=8,
        enforce_eager=True,
    )
    print(f"Engine loaded in {time.time()-t0:.1f}s")

    # Initialize controller in pure SFT mode (hybrid_switch_step = total_steps)
    print("Initializing controller (pure SFT)...")
    layer_specs = discover_layers(args.model_path, target_modules)
    backend = VLLMBackend(llm, layer_specs, rank)
    controller = DSMeZO_Controller(backend, layer_specs, {
        "output_dir": str(args.output_dir),
        "adapter_path": str(args.adapter_path),
        "model_path": str(args.model_path),
        "total_steps": args.total_steps,
        "hybrid_switch_step": args.total_steps,  # Pure SFT: never switch to RL
        "eta_max": 1e-2,
    })
    print(f"Layers: {len(controller.layers)} | hybrid_switch_step: {controller.hybrid_switch_step}")

    # Held-out data for perplexity eval
    held_out_ids = [s["token_ids"] for s in held_out]
    held_out_plens = [s["prompt_len"] for s in held_out]

    # Pre-training baselines (with initial PiSSA adapter for fair comparison)
    print("\n--- Pre-training baselines ---")
    controller.backend.sync_adapters({}, {}, controller.layers)
    baseline_ppl = eval_perplexity(llm, held_out_ids, held_out_plens,
                                   lora_request=backend.lora_pos)
    print(f"  Held-out perplexity: {baseline_ppl['perplexity']:.2f} "
          f"(NLL: {baseline_ppl['avg_nll']:.4f}, {baseline_ppl['total_tokens']} tokens)")

    baseline_gsm8k = eval_gsm8k(llm, lora_request=backend.lora_pos)
    print(f"  GSM8K exact_match:   {baseline_gsm8k['exact_match']:.1%} "
          f"({baseline_gsm8k['num_parsed']}/{baseline_gsm8k['num_samples']} parsed)")

    # Train
    print(f"\n--- SFT Training ({args.total_steps} steps) ---")

    t_start = time.time()
    log = []
    for step_idx in range(args.total_steps):
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
            eta_remain = s_per_step * (args.total_steps - step_idx - 1)
            loss_str = f"{controller.loss_ema:.4f}"
            print(f"  step {step_idx+1:4d}/{args.total_steps} | "
                  f"loss_ema={loss_str} | "
                  f"lr={controller.eta:.2e} | "
                  f"{s_per_step:.1f}s/step | "
                  f"ETA {eta_remain:.0f}s")

    train_time = time.time() - t_start
    print(f"\nTraining complete: {train_time:.1f}s total, {train_time/args.total_steps:.1f}s/step")

    # Post-training evaluation
    print("\n--- Post-training evaluation ---")
    controller.backend.sync_adapters({}, {}, controller.layers)

    post_ppl = eval_perplexity(llm, held_out_ids, held_out_plens,
                               lora_request=backend.lora_pos)
    print(f"  Held-out perplexity: {post_ppl['perplexity']:.2f} "
          f"(NLL: {post_ppl['avg_nll']:.4f})")

    post_gsm8k = eval_gsm8k(llm, lora_request=backend.lora_pos)
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
    print(f"  Initial loss EMA:     {controller.initial_loss_ema:.4f}")
    print(f"  Final loss EMA:       {controller.loss_ema:.4f}")
    print(f"  Training time:        {train_time:.1f}s ({train_time/args.total_steps:.1f}s/step)")
    print(f"  Steps completed:      {args.total_steps}")

    # Save results
    results = {
        "baseline_ppl": baseline_ppl,
        "post_ppl": post_ppl,
        "baseline_gsm8k": baseline_gsm8k,
        "post_gsm8k": post_gsm8k,
        "training_log": log,
        "train_time_seconds": train_time,
        "total_steps": args.total_steps,
        "hybrid_switch_step": args.total_steps,
    }
    results_path = args.output_dir / "sft_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
