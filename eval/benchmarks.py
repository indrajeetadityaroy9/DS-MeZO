"""Standard benchmarks for DS-MeZO evaluation.

Uses evaluate (HuggingFace) for metrics and datasets for data loading.
All benchmarks use vLLM for generation via the engine passed at call time.
"""

from __future__ import annotations

import math
import re
from typing import Any

import evaluate
from datasets import load_dataset
from vllm import SamplingParams


# ---------------------------------------------------------------------------
# Perplexity — direct NLL measurement on held-out data
# ---------------------------------------------------------------------------

def eval_perplexity(
    engine: Any,
    token_sequences: list[list[int]],
    prompt_lens: list[int],
    lora_request: Any = None,
) -> dict[str, float | int]:
    """Per-token perplexity on held-out sequences.

    Measures NLL via prefill logprobs, reports perplexity = exp(avg NLL).
    This is the most direct measure of SFT effectiveness.
    """
    score_params = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0)
    prompts = [{"prompt_token_ids": seq} for seq in token_sequences]
    outputs = engine.generate(
        prompts, sampling_params=score_params, lora_request=lora_request,
    )

    total_nll, total_tokens = 0.0, 0
    num_samples = 0
    for out, seq, plen in zip(outputs, token_sequences, prompt_lens):
        sample_nll = 0.0
        n_completion = 0
        for i, token_lp in enumerate(out.prompt_logprobs[1:], 1):
            if i < plen:
                continue
            tok_id = seq[i]
            lp = token_lp[tok_id].logprob
            sample_nll += -lp
            n_completion += 1
        if n_completion > 0:
            total_nll += sample_nll
            total_tokens += n_completion
            num_samples += 1

    avg_nll = total_nll / max(total_tokens, 1)
    return {
        "perplexity": math.exp(avg_nll),
        "avg_nll": avg_nll,
        "total_tokens": total_tokens,
        "num_samples": num_samples,
    }


# ---------------------------------------------------------------------------
# GSM8K — math reasoning, exact match on final numerical answer
# ---------------------------------------------------------------------------

def eval_gsm8k(
    engine: Any,
    lora_request: Any = None,
    num_samples: int | None = None,
) -> dict[str, Any]:
    """GSM8K exact-match accuracy.

    Generates chain-of-thought solutions, extracts final answer via #### pattern,
    compares to ground truth using evaluate.exact_match.
    """
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    exact_match = evaluate.load("exact_match")

    prompts = [
        f"Question: {row['question']}\nAnswer: Let's solve step by step.\n"
        for row in dataset
    ]
    outputs = engine.generate(
        prompts,
        SamplingParams(max_tokens=512, temperature=0.0),
        lora_request=lora_request,
    )

    predictions, references = [], []
    for out, row in zip(outputs, dataset):
        pred_match = re.search(r"####?\s*(-?[\d,]+)", out.outputs[0].text)
        ref_match = re.search(r"####?\s*(-?[\d,]+)", row["answer"])
        predictions.append(
            pred_match.group(1).replace(",", "") if pred_match else ""
        )
        references.append(ref_match.group(1).replace(",", ""))

    result = exact_match.compute(predictions=predictions, references=references)
    result["num_samples"] = len(dataset)
    result["num_parsed"] = sum(1 for p in predictions if p != "")
    return result


# ---------------------------------------------------------------------------
# MMLU — knowledge, multiple-choice via logprob comparison
# ---------------------------------------------------------------------------

def eval_mmlu(
    engine: Any,
    tokenizer: Any,
    lora_request: Any = None,
    num_samples: int | None = None,
) -> dict[str, Any]:
    """MMLU accuracy via batched logprob comparison for A/B/C/D choices.

    Uses evaluate.accuracy for the metric. Scores each choice by computing
    the logprob of the answer token after the prompt.
    """
    dataset = load_dataset("cais/mmlu", "all", split="test")
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    accuracy = evaluate.load("accuracy")
    score_params = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0)
    choices = ["A", "B", "C", "D"]

    # Build all sequences for all samples in one batch
    all_seqs: list[list[int]] = []
    for row in dataset:
        prompt = f"Question: {row['question']}\n"
        for i, c in enumerate(choices):
            prompt += f"{c}. {row['choices'][i]}\n"
        prompt += "Answer:"

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)

        for c in choices:
            choice_ids = tokenizer.encode(f" {c}", add_special_tokens=False)
            all_seqs.append(prompt_ids + choice_ids)

    # Single batched engine call
    seq_prompts = [{"prompt_token_ids": seq} for seq in all_seqs]
    outputs = engine.generate(
        seq_prompts, sampling_params=score_params, lora_request=lora_request,
    )

    # Collect logprobs per sample
    predictions: list[int] = []
    references: list[int] = []
    num_choices = len(choices)
    for sample_idx, row in enumerate(dataset):
        best_choice, best_lp = 0, float("-inf")
        base = sample_idx * num_choices
        for i in range(num_choices):
            out = outputs[base + i]
            seq = all_seqs[base + i]
            last_lp = out.prompt_logprobs[-1]
            tok_id = seq[-1]
            lp = last_lp[tok_id].logprob
            if lp > best_lp:
                best_lp, best_choice = lp, i

        predictions.append(best_choice)
        references.append(row["answer"])

    result = accuracy.compute(predictions=predictions, references=references)
    result["num_samples"] = len(dataset)
    return result


# ---------------------------------------------------------------------------
# Shared code completion trimming
# ---------------------------------------------------------------------------

def _trim_code_completion(text: str) -> str:
    """Stop at end of code block or first non-indented non-definition line."""
    lines = text.split("\n")
    trimmed: list[str] = []
    for line in lines:
        if line.strip() == "```":
            break
        if trimmed and line.strip() and not line.startswith(" ") and not line.startswith("\t"):
            if not line.startswith("def ") and not line.startswith("class "):
                break
        trimmed.append(line)
    return "\n".join(trimmed)


# ---------------------------------------------------------------------------
# HumanEval — code generation, pass@1 via execution
# ---------------------------------------------------------------------------

def eval_humaneval(
    engine: Any, lora_request: Any = None,
) -> dict[str, Any]:
    """HumanEval pass@1 via evaluate.code_eval.

    Generates function completions, executes with test cases in sandbox.
    """
    dataset = load_dataset("openai/openai_humaneval", split="test")
    code_eval = evaluate.load("code_eval")

    prompts = [row["prompt"] for row in dataset]
    outputs = engine.generate(
        prompts,
        SamplingParams(max_tokens=512, temperature=0.0),
        lora_request=lora_request,
    )

    test_cases: list[str] = []
    predictions: list[list[str]] = []
    for out, row in zip(outputs, dataset):
        predictions.append([_trim_code_completion(out.outputs[0].text)])
        test_cases.append(row["test"] + f"\ncheck({row['entry_point']})")

    metrics, _ = code_eval.compute(
        references=test_cases,
        predictions=predictions,
    )
    metrics["num_samples"] = len(dataset)
    return metrics


# ---------------------------------------------------------------------------
# MBPP — code generation, pass@1 via execution
# ---------------------------------------------------------------------------

def eval_mbpp(
    engine: Any, lora_request: Any = None,
) -> dict[str, Any]:
    """MBPP pass@1 via evaluate.code_eval.

    Generates Python functions from natural language descriptions,
    executes against assertion-based test cases.
    """
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    code_eval = evaluate.load("code_eval")

    prompts: list[str] = []
    for row in dataset:
        func_match = re.search(r'(?:assert\s+(?:not\s+)?(?:\()?\s*)(\w+)\s*\(', row['test_list'][0])
        func_name = func_match.group(1) if func_match else "solution"
        prompts.append(f"Write a Python function named `{func_name}`.\n\n"
                       f"{row['prompt']}\n\n```python\n")

    outputs = engine.generate(
        prompts, SamplingParams(max_tokens=512, temperature=0.0),
        lora_request=lora_request,
    )

    predictions: list[list[str]] = []
    references: list[str] = []
    for out, row in zip(outputs, dataset):
        predictions.append([_trim_code_completion(out.outputs[0].text)])
        imports = "\n".join(row["test_imports"])
        tests = "\n".join(row["test_list"])
        references.append(f"{imports}\n{tests}" if imports else tests)

    metrics, _ = code_eval.compute(references=references, predictions=predictions)
    metrics["num_samples"] = len(dataset)
    return metrics
