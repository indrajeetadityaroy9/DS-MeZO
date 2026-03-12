"""DS-MeZO SFT evaluation: train on GSM8K, evaluate via perplexity and exact match."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftConfig
from transformers import AutoTokenizer

from ds_mezo.model_config import discover_layers
from ds_mezo.backend import VLLMBackend, create_engine
from ds_mezo.controller import DSMeZO_Controller
from ds_mezo.kernels import fused_perturb_dual, zo_muon_update
from eval.benchmarks import eval_perplexity, eval_gsm8k


def sft_step(ctrl: DSMeZO_Controller, sample: dict) -> None:
    """One SFT optimization step using the controller's internals."""
    ctrl.step_count += 1

    acts = ctrl.backend.extract_activations([sample["prompt_text"]])
    ctrl._update_activation_bases(acts)

    perturbations = {l.key: ctrl._get_perturbation(l) for l in ctrl.layers}

    pos_layers, neg_layers = {}, {}
    for layer in ctrl.layers:
        z_A, z_B = perturbations[layer.key]
        key_A, key_B = (layer.key, "A"), (layer.key, "B")
        fused_perturb_dual(layer.A, z_A, ctrl.pos_buffers[key_A], ctrl.neg_buffers[key_A])
        fused_perturb_dual(layer.B, z_B, ctrl.pos_buffers[key_B], ctrl.neg_buffers[key_B])
        pos_layers[layer.key] = (ctrl.pos_buffers[key_A], ctrl.pos_buffers[key_B])
        neg_layers[layer.key] = (ctrl.neg_buffers[key_A], ctrl.neg_buffers[key_B])

    ctrl.backend.sync_adapters(pos_layers, neg_layers, ctrl.layers)

    toks, plen = sample["token_ids"], sample["prompt_len"]
    lp_pos = ctrl.backend.score([toks], ctrl.backend.lora_pos)[0][plen:]
    lp_neg = ctrl.backend.score([toks], ctrl.backend.lora_neg)[0][plen:]
    dd = float((-sum(lp_pos) / len(lp_pos)) - (-sum(lp_neg) / len(lp_neg))) / (2.0 * ctrl.eps)

    max_window = int(math.sqrt(ctrl.total_steps))
    momentum = 1.0 - 1.0 / min(ctrl.step_count, max_window)

    for layer in ctrl.layers:
        z_A, z_B = perturbations[layer.key]
        key_A, key_B = (layer.key, "A"), (layer.key, "B")
        zo_muon_update(layer.A, ctrl.momentum_buffers[key_A], z_A, ctrl.scratch_buffers[key_A],
                       dd, momentum, ctrl.eta, ctrl.norm_floor)
        zo_muon_update(layer.B, ctrl.momentum_buffers[key_B], z_B, ctrl.scratch_buffers[key_B],
                       dd, momentum, ctrl.eta, ctrl.norm_floor)

    ctrl.lr_scheduler.step()
    ctrl.eta = ctrl._lr_opt.param_groups[0]["lr"]


def prepare_sft_data(
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    held_out_samples: int,
) -> tuple[list[dict], list[dict]]:
    dataset = load_dataset("openai/gsm8k", "main", split="train")

    all_data = []
    for row in dataset:
        prompt = f"Question: {row['question']}\nAnswer: Let's solve step by step.\n"
        completion = row["answer"]
        full_text = prompt + completion
        full_ids = tokenizer.encode(full_text, add_special_tokens=True)[:max_seq_len]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_len = len(prompt_ids)
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

    peft_config = PeftConfig.from_pretrained(str(args.adapter_path))
    rank = peft_config.r
    target_modules = list(peft_config.target_modules)

    print("=" * 70)
    print("DS-MeZO SFT EVALUATION: GSM8K")
    print(f"Model: {model_name} | PiSSA rank-{rank}")
    print(f"Steps: {args.total_steps} | Max seq len: {args.max_seq_len}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    train_data, held_out = prepare_sft_data(
        tokenizer, args.max_seq_len, args.held_out_samples,
    )

    print("\nLoading vLLM engine...")
    t0 = time.time()
    llm = create_engine(args.model_path, rank)
    print(f"Engine loaded in {time.time()-t0:.1f}s")

    layer_specs = discover_layers(args.model_path, target_modules)
    backend = VLLMBackend(llm, layer_specs, rank)
    controller = DSMeZO_Controller(backend, layer_specs, {
        "output_dir": str(args.output_dir),
        "adapter_path": str(args.adapter_path),
        "total_steps": args.total_steps,
    })
    controller._calibrate_activation_bases_full([train_data[0]["prompt_text"]])

    held_out_ids = [s["token_ids"] for s in held_out]
    held_out_plens = [s["prompt_len"] for s in held_out]

    # Pre-training baselines (with initial PiSSA adapter for fair comparison)
    print("\n--- Pre-training baselines ---")
    controller.backend.sync_adapters({}, {}, controller.layers)
    baseline_ppl = eval_perplexity(llm, held_out_ids, held_out_plens,
                                   lora_request=backend.lora_pos)
    ppl_ci = baseline_ppl['perplexity_ci95']
    print(f"  Held-out perplexity: {baseline_ppl['perplexity']:.2f} "
          f"(95% CI: {ppl_ci[0]:.2f}–{ppl_ci[1]:.2f}, "
          f"{baseline_ppl['total_tokens']} tokens)")

    baseline_gsm8k = eval_gsm8k(llm, lora_request=backend.lora_pos)
    em_ci = baseline_gsm8k['exact_match_ci95']
    print(f"  GSM8K exact_match:   {baseline_gsm8k['exact_match']:.1%} "
          f"(95% CI: {em_ci[0]:.1%}–{em_ci[1]:.1%}, "
          f"{baseline_gsm8k['num_parsed']}/{baseline_gsm8k['num_samples']} parsed)")

    # Train
    print(f"\n--- SFT Training ({args.total_steps} steps) ---")

    t_start = time.time()
    log = []
    for step_idx in range(args.total_steps):
        sample = train_data[step_idx % len(train_data)]
        sft_step(controller, sample)

        entry = {
            "step": step_idx + 1,
            "eta": controller.eta,
            "eps": controller.eps,
        }
        log.append(entry)

        if (step_idx + 1) % 200 == 0:
            print(f"  step {step_idx+1}/{args.total_steps} | lr={controller.eta:.2e}")

    train_time = time.time() - t_start
    print(f"\nTraining complete: {train_time:.1f}s total, {train_time/args.total_steps:.1f}s/step")

    # Post-training evaluation
    print("\n--- Post-training evaluation ---")
    controller.backend.sync_adapters({}, {}, controller.layers)

    post_ppl = eval_perplexity(llm, held_out_ids, held_out_plens,
                               lora_request=backend.lora_pos)
    ppl_ci = post_ppl['perplexity_ci95']
    print(f"  Held-out perplexity: {post_ppl['perplexity']:.2f} "
          f"(95% CI: {ppl_ci[0]:.2f}–{ppl_ci[1]:.2f})")

    post_gsm8k = eval_gsm8k(llm, lora_request=backend.lora_pos)
    em_ci = post_gsm8k['exact_match_ci95']
    print(f"  GSM8K exact_match:   {post_gsm8k['exact_match']:.1%} "
          f"(95% CI: {em_ci[0]:.1%}–{em_ci[1]:.1%})")

    # Summary
    ppl_delta = post_ppl['perplexity'] - baseline_ppl['perplexity']
    print(f"\n  Perplexity: {baseline_ppl['perplexity']:.2f} → {post_ppl['perplexity']:.2f} ({ppl_delta:+.2f})")
    print(f"  GSM8K EM:   {baseline_gsm8k['exact_match']:.1%} → {post_gsm8k['exact_match']:.1%}")
    print(f"  Time: {train_time:.1f}s ({train_time/args.total_steps:.1f}s/step)")

    # Save results
    results = {
        "baseline_ppl": baseline_ppl,
        "post_ppl": post_ppl,
        "baseline_gsm8k": baseline_gsm8k,
        "post_gsm8k": post_gsm8k,
        "training_log": log,
        "train_time_seconds": train_time,
        "total_steps": args.total_steps,
    }
    results_path = args.output_dir / "sft_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
