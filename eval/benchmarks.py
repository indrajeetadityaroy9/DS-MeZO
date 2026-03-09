"""Standard benchmarks for DS-MeZO evaluation.

Follows canonical evaluation protocols from authoritative harnesses:
- HumanEval: openai/human-eval (Chen et al. 2021, Codex paper §2-3)
- MBPP: bigcode-evaluation-harness (zero-shot docstring format, sanitized split)
- GSM8K: lm-evaluation-harness (8-shot CoT, two-stage answer extraction)
- Perplexity: conditional NLL on held-out completion tokens via prefill logprobs

Protocol parameters (n_samples, temperature) are required arguments —
callers specify them explicitly via CLI args.
"""

from __future__ import annotations

import functools
import math
import re
from typing import Any

import evaluate
from datasets import load_dataset
from vllm import SamplingParams

from eval.utils import extract_code, pass_at_k


# Perplexity — direct NLL measurement on held-out data

def eval_perplexity(
    engine: Any,
    token_sequences: list[list[int]],
    prompt_lens: list[int],
    lora_request: Any = None,
) -> dict[str, float | int]:
    """Per-token perplexity on held-out sequences.

    Measures NLL via prefill logprobs on completion tokens (after prompt_len).
    Reports perplexity = exp(avg NLL).
    """
    score_params = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0)
    prompts = [{"prompt_token_ids": seq} for seq in token_sequences]
    outputs = engine.generate(
        prompts, sampling_params=score_params, lora_request=lora_request,
    )

    total_nll, total_tokens = 0.0, 0
    for out, seq, plen in zip(outputs, token_sequences, prompt_lens):
        # Completion tokens only: skip prompt positions (logprobs[0] is None, so [1:] → index 1..N)
        for i in range(plen, len(out.prompt_logprobs)):
            total_nll += -out.prompt_logprobs[i][seq[i]].logprob
            total_tokens += 1

    avg_nll = total_nll / total_tokens
    return {
        "perplexity": math.exp(avg_nll),
        "avg_nll": avg_nll,
        "total_tokens": total_tokens,
    }


# GSM8K — 8-shot CoT, two-stage answer extraction, exact match

@functools.lru_cache(maxsize=1)
def _get_gsm8k_fewshot(n_shot: int = 8) -> tuple[dict, ...]:
    """Load n_shot demonstrations from GSM8K train split.

    Uses the first n_shot training examples as few-shot demonstrations.
    Standard protocol: 8-shot CoT (Wei et al. 2022, lm-evaluation-harness).
    Returns tuple (hashable for lru_cache).
    """
    train = load_dataset("openai/gsm8k", "main", split="train")
    return tuple(dict(row) for row in train.select(range(n_shot)))


def _build_gsm8k_prompt(question: str, fewshot: tuple[dict, ...]) -> str:
    """Build GSM8K prompt with 8-shot CoT demonstrations.

    Format: Q:/A: with full chain-of-thought gold answers from the train split.
    Follows lm-evaluation-harness canonical format.
    """
    parts = []
    for ex in fewshot:
        parts.append(f"Q: {ex['question']}\nA: {ex['answer']}")
    parts.append(f"Q: {question}\nA:")
    return "\n\n".join(parts)


def _extract_gsm8k_answer(text: str) -> str:
    """Two-stage GSM8K answer extraction (lm-evaluation-harness protocol).

    Stage 1: "The answer is X" — model-generated conclusion format.
    Stage 2: "#### X" — dataset gold format (exactly 4 hashes; 3 hashes is
             a markdown header and must not match).
    Stage 3: Last integer in the text — desperation fallback.
    """
    # Stage 1: "The answer is X"
    m = re.search(r"[Tt]he answer is\s*\$?\s*(-?[\d,]+)", text)
    if m:
        return m.group(1).replace(",", "")

    # Stage 2: exactly 4 hashes (####), not 3 (### = markdown header)
    m = re.search(r"####\s*(-?[\d,]+)", text)
    if m:
        return m.group(1).replace(",", "")

    # Stage 3: last integer
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else ""


def eval_gsm8k(
    engine: Any,
    lora_request: Any = None,
    num_samples: int | None = None,
) -> dict[str, Any]:
    """GSM8K exact-match accuracy with 8-shot CoT prompting.

    Default evaluates full test set (1319 problems). num_samples can be set
    for development but should be None for ICML reporting.

    Protocol:
    - 8-shot CoT demonstrations from train split
    - Greedy decoding (temperature=0)
    - Stop at "Q:" to prevent multi-turn generation
    - Two-stage answer extraction
    - Exact match after comma removal
    """
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if num_samples:
        dataset = dataset.select(range(num_samples))

    exact_match = evaluate.load("exact_match")

    fewshot = _get_gsm8k_fewshot()
    prompts = [_build_gsm8k_prompt(row["question"], fewshot) for row in dataset]

    # Greedy with stop at "Q:" prevents multi-turn generation
    # max_tokens=512: standard CoT generation budget
    outputs = engine.generate(
        prompts,
        SamplingParams(max_tokens=512, temperature=0.0, stop=["Q:", "\n\n\n"]),
        lora_request=lora_request,
    )

    predictions, references = [], []
    for out, row in zip(outputs, dataset):
        predictions.append(_extract_gsm8k_answer(out.outputs[0].text))
        ref_match = re.search(r"####\s*(-?[\d,]+)", row["answer"])
        references.append(ref_match.group(1).replace(",", ""))

    result = exact_match.compute(predictions=predictions, references=references)
    result["num_samples"] = len(dataset)
    result["num_parsed"] = sum(1 for p in predictions if p != "")
    return result


# MBPP — zero-shot docstring format, pass@k via execution

def load_mbpp_train() -> list[dict[str, Any]]:
    """Load MBPP sanitized train split with prompts and test metadata."""
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    return [
        {
            "prompt": build_mbpp_prompt(row),
            "test_list": row["test_list"],
            "test_imports": row["test_imports"],
        }
        for row in dataset
    ]


def build_mbpp_prompt(row: dict) -> str:
    """Build MBPP prompt in bigcode-evaluation-harness docstring format.

    Canonical zero-shot format: task description + first test assertion as docstring.
    """
    return f'"""\n{row["prompt"]}\n{row["test_list"][0]}\n"""\n'


def eval_mbpp(
    engine: Any,
    lora_request: Any,
    n_samples: int,
    temperature: float,
) -> dict[str, Any]:
    """MBPP pass@k via code execution (sanitized split).

    Protocol (bigcode-evaluation-harness):
    - Zero-shot docstring prompt with first test assertion
    - Stop sequences prevent extraneous generation
    - Nucleus sampling (top_p=0.95) when temperature > 0 (Codex paper §3.2)
    - Execution via HuggingFace code_eval with sandboxing
    - Unbiased pass@k estimator (Chen et al. 2021, Eq. 1)
    """
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    code_eval = evaluate.load("code_eval")

    prompts = [build_mbpp_prompt(row) for row in dataset]

    # Stop sequences from bigcode-evaluation-harness
    stop = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif"]

    # Codex paper §3.2: top_p=0.95 nucleus sampling for temperature > 0
    gen_params = SamplingParams(
        max_tokens=512, temperature=temperature, n=n_samples,
        stop=stop, **({"top_p": 0.95} if temperature > 0 else {}),
    )

    outputs = engine.generate(prompts, gen_params, lora_request=lora_request)

    predictions: list[list[str]] = []
    references: list[str] = []
    for out, row in zip(outputs, dataset):
        completions = [extract_code(o.text) for o in out.outputs]
        predictions.append(completions)
        imports = "\n".join(row["test_imports"])
        tests = "\n".join(row["test_list"])
        references.append(f"{imports}\n{tests}")

    _, results_list = code_eval.compute(
        references=references, predictions=predictions,
    )

    per_task_correct = [sum(r[1] for r in task) for task in results_list]
    per_task_n = [len(task) for task in results_list]

    pass_1 = sum(
        pass_at_k(n, c, 1) for n, c in zip(per_task_n, per_task_correct)
    ) / len(dataset)

    result = {
        "pass@1": pass_1,
        "per_task_correct": per_task_correct,
        "per_task_n": per_task_n,
        "num_tasks": len(dataset),
    }

    if n_samples >= 10:
        result["pass@10"] = sum(
            pass_at_k(n, c, 10) for n, c in zip(per_task_n, per_task_correct)
        ) / len(dataset)

    return result


# HumanEval — function completion, pass@k via execution

def eval_humaneval(
    engine: Any,
    lora_request: Any,
    n_samples: int,
    temperature: float,
) -> dict[str, Any]:
    """HumanEval pass@k via code execution.

    Protocol (openai/human-eval, Codex paper §2-3):
    - Prompt = function signature + docstring (canonical, from dataset)
    - Completion concatenated with prompt before execution
    - Stop sequences without trailing spaces (Codex paper + bigcode-harness)
    - Nucleus sampling (top_p=0.95) when temperature > 0
    - Unbiased pass@k estimator (Chen et al. 2021, Eq. 1)
    """
    dataset = load_dataset("openai/openai_humaneval", split="test")
    code_eval = evaluate.load("code_eval")

    prompts = [row["prompt"] for row in dataset]

    # Codex paper stop sequences (no trailing spaces) + bigcode-harness additions
    stop = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]

    gen_params = SamplingParams(
        max_tokens=512, temperature=temperature, n=n_samples,
        stop=stop, **({"top_p": 0.95} if temperature > 0 else {}),
    )

    outputs = engine.generate(prompts, gen_params, lora_request=lora_request)

    predictions: list[list[str]] = []
    references: list[str] = []
    for out, row in zip(outputs, dataset):
        completions = [row["prompt"] + o.text for o in out.outputs]
        predictions.append(completions)
        references.append(row["test"] + f"\ncheck({row['entry_point']})")

    _, results_list = code_eval.compute(
        references=references, predictions=predictions,
    )

    per_task_correct = [sum(r[1] for r in task) for task in results_list]
    per_task_n = [len(task) for task in results_list]

    pass_1 = sum(
        pass_at_k(n, c, 1) for n, c in zip(per_task_n, per_task_correct)
    ) / len(dataset)

    result = {
        "pass@1": pass_1,
        "per_task_correct": per_task_correct,
        "per_task_n": per_task_n,
        "num_tasks": len(dataset),
    }

    if n_samples >= 10:
        result["pass@10"] = sum(
            pass_at_k(n, c, 10) for n, c in zip(per_task_n, per_task_correct)
        ) / len(dataset)

    return result
