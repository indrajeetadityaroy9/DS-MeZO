"""Benchmarks: perplexity, GSM8K exact match, MBPP pass@k, HumanEval pass@k,
SST-2 accuracy, RTE accuracy, LiveCodeBench pass@k, APPS data loading."""

from __future__ import annotations

import functools
import json
import math
import re
from typing import Any

import numpy as np
from scipy import stats

import evaluate
from datasets import load_dataset
from vllm import SamplingParams

from eval.utils import extract_code


@functools.lru_cache(maxsize=1)
def _get_code_eval():
    return evaluate.load("code_eval")


@functools.lru_cache(maxsize=1)
def _get_exact_match():
    return evaluate.load("exact_match")


def _bootstrap_ci(
    samples: np.ndarray,
    confidence_level: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    result = stats.bootstrap(
        (samples,),
        np.mean,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        random_state=np.random.default_rng(seed),
        method="percentile",
    )
    return (
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )


def eval_perplexity(
    engine: Any,
    token_sequences: list[list[int]],
    prompt_lens: list[int],
    lora_request: Any,
) -> dict[str, Any]:
    score_params = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0)
    prompts = [{"prompt_token_ids": seq} for seq in token_sequences]
    outputs = engine.generate(
        prompts, sampling_params=score_params, lora_request=lora_request,
    )

    per_sample_nlls: list[float] = []
    total_nll, total_tokens = 0.0, 0
    for out, seq, plen in zip(outputs, token_sequences, prompt_lens):
        sample_nll = 0.0
        sample_tokens = 0
        for i in range(plen, len(out.prompt_logprobs)):
            sample_nll += -out.prompt_logprobs[i][seq[i]].logprob
            sample_tokens += 1
        total_nll += sample_nll
        total_tokens += sample_tokens
        per_sample_nlls.append(sample_nll / sample_tokens)

    avg_nll = total_nll / total_tokens
    samples = np.array(per_sample_nlls)
    nll_ci = _bootstrap_ci(samples)

    return {
        "perplexity": math.exp(avg_nll),
        "perplexity_ci95": (math.exp(nll_ci[0]), math.exp(nll_ci[1])),
        "avg_nll": avg_nll,
        "avg_nll_ci95": nll_ci,
        "total_tokens": total_tokens,
        "num_samples": len(per_sample_nlls),
    }


@functools.lru_cache(maxsize=1)
def _get_gsm8k_fewshot(n_shot: int = 8) -> tuple[dict, ...]:
    train = load_dataset("openai/gsm8k", "main", split="train")
    return tuple(dict(row) for row in train.select(range(n_shot)))


def _build_gsm8k_prompt(question: str, fewshot: tuple[dict, ...]) -> str:
    parts = []
    for ex in fewshot:
        parts.append(f"Q: {ex['question']}\nA: {ex['answer']}")
    parts.append(f"Q: {question}\nA:")
    return "\n\n".join(parts)


def _extract_gsm8k_answer(text: str) -> str:
    m = re.search(r"[Tt]he answer is\s*\$?\s*(-?[\d,]+)", text)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"####\s*(-?[\d,]+)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else ""


def eval_gsm8k(
    engine: Any,
    lora_request: Any,
) -> dict[str, Any]:
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    exact_match = _get_exact_match()
    fewshot = _get_gsm8k_fewshot()
    prompts = [_build_gsm8k_prompt(row["question"], fewshot) for row in dataset]

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

    per_sample = np.array([
        1.0 if p == r else 0.0
        for p, r in zip(predictions, references)
    ])
    result["exact_match_ci95"] = _bootstrap_ci(per_sample)
    result["num_samples"] = len(dataset)
    result["num_parsed"] = sum(1 for p in predictions if p != "")
    return result


def load_mbpp_train() -> list[dict[str, Any]]:
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
    return f'"""\n{row["prompt"]}\n{row["test_list"][0]}\n"""\n'


def eval_mbpp(
    engine: Any,
    lora_request: Any,
    n_samples: int,
    temperature: float,
) -> dict[str, Any]:
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    code_eval = _get_code_eval()
    prompts = [build_mbpp_prompt(row) for row in dataset]

    stop = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif"]
    gen_params = SamplingParams(
        max_tokens=512, temperature=temperature, n=n_samples,
        stop=stop, top_p=0.95,
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

    pass_at_k_results, results_dict = code_eval.compute(
        references=references, predictions=predictions,
        k=[1, 10] if n_samples >= 10 else [1],
    )

    per_task_c = [
        sum(1 for _, r in task_results if r["passed"])
        for task_results in results_dict.values()
    ]
    per_task_n = [len(task_results) for task_results in results_dict.values()]
    per_task_pass1 = np.array([c / n for c, n in zip(per_task_c, per_task_n)])

    result: dict[str, Any] = {
        "pass@1": pass_at_k_results["pass@1"],
        "pass@1_ci95": _bootstrap_ci(per_task_pass1),
        "num_tasks": len(dataset),
    }

    if n_samples >= 10:
        per_task_pass10 = np.array([
            1.0 - math.comb(n - c, 10) / math.comb(n, 10)
            for c, n in zip(per_task_c, per_task_n)
        ])
        result["pass@10"] = pass_at_k_results["pass@10"]
        result["pass@10_ci95"] = _bootstrap_ci(per_task_pass10)

    return result


# ── HumanEval ─────────────────────────────────────────────────────────────────


def eval_humaneval(
    engine: Any,
    lora_request: Any,
    n_samples: int,
    temperature: float,
) -> dict[str, Any]:
    dataset = load_dataset("openai_humaneval", split="test")
    code_eval = _get_code_eval()

    stop = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif"]
    gen_params = SamplingParams(
        max_tokens=512, temperature=temperature, n=n_samples,
        stop=stop, top_p=0.95,
    )

    prompts = [row["prompt"] for row in dataset]
    outputs = engine.generate(prompts, gen_params, lora_request=lora_request)

    predictions: list[list[str]] = []
    references: list[str] = []
    for out, row in zip(outputs, dataset):
        completions = [row["prompt"] + extract_code(o.text) for o in out.outputs]
        predictions.append(completions)
        references.append(row["test"] + f"\ncheck({row['entry_point']})")

    pass_at_k_results, results_dict = code_eval.compute(
        references=references, predictions=predictions,
        k=[1, 10] if n_samples >= 10 else [1],
    )

    per_task_c = [
        sum(1 for _, r in task_results if r["passed"])
        for task_results in results_dict.values()
    ]
    per_task_n = [len(task_results) for task_results in results_dict.values()]
    per_task_pass1 = np.array([c / n for c, n in zip(per_task_c, per_task_n)])

    result: dict[str, Any] = {
        "pass@1": pass_at_k_results["pass@1"],
        "pass@1_ci95": _bootstrap_ci(per_task_pass1),
        "num_tasks": len(dataset),
    }

    if n_samples >= 10:
        per_task_pass10 = np.array([
            1.0 - math.comb(n - c, 10) / math.comb(n, 10)
            for c, n in zip(per_task_c, per_task_n)
        ])
        result["pass@10"] = pass_at_k_results["pass@10"]
        result["pass@10_ci95"] = _bootstrap_ci(per_task_pass10)

    return result


# ── SuperGLUE: SST-2 + RTE ───────────────────────────────────────────────────


def _build_classification_prompt(
    examples: list[dict], query: str, template_fn: Any,
) -> str:
    parts = [template_fn(ex, with_label=True) for ex in examples]
    parts.append(template_fn(query, with_label=False))
    return "\n\n".join(parts)


def eval_sst2(
    engine: Any,
    lora_request: Any,
    n_shot: int,
) -> dict[str, Any]:
    dataset = load_dataset("nyu-mll/glue", "sst2", split="validation")
    train = load_dataset("nyu-mll/glue", "sst2", split="train")
    label_map = {0: "negative", 1: "positive"}
    demos = [dict(row) for row in train.select(range(n_shot))]

    def template(row: dict, with_label: bool = True) -> str:
        text = f"Sentence: {row['sentence']}\nSentiment:"
        if with_label:
            text += f" {label_map[row['label']]}"
        return text

    prompts = [
        _build_classification_prompt(demos, dict(row), template)
        for row in dataset
    ]

    outputs = engine.generate(
        prompts,
        SamplingParams(max_tokens=10, temperature=0.0, stop=["\n"]),
        lora_request=lora_request,
    )

    per_sample = np.zeros(len(dataset))
    for i, (out, row) in enumerate(zip(outputs, dataset)):
        pred = out.outputs[0].text.strip().lower()
        gold = label_map[row["label"]]
        if pred.startswith(gold):
            per_sample[i] = 1.0

    return {
        "accuracy": float(per_sample.mean()),
        "accuracy_ci95": _bootstrap_ci(per_sample),
        "num_samples": len(dataset),
    }


def eval_rte(
    engine: Any,
    lora_request: Any,
    n_shot: int,
) -> dict[str, Any]:
    dataset = load_dataset("nyu-mll/glue", "rte", split="validation")
    train = load_dataset("nyu-mll/glue", "rte", split="train")
    label_map = {0: "yes", 1: "no"}
    demos = [dict(row) for row in train.select(range(n_shot))]

    def template(row: dict, with_label: bool = True) -> str:
        text = f"Premise: {row['sentence1']}\nHypothesis: {row['sentence2']}\nEntailment:"
        if with_label:
            text += f" {label_map[row['label']]}"
        return text

    prompts = [
        _build_classification_prompt(demos, dict(row), template)
        for row in dataset
    ]

    outputs = engine.generate(
        prompts,
        SamplingParams(max_tokens=10, temperature=0.0, stop=["\n"]),
        lora_request=lora_request,
    )

    per_sample = np.zeros(len(dataset))
    for i, (out, row) in enumerate(zip(outputs, dataset)):
        pred = out.outputs[0].text.strip().lower()
        gold = label_map[row["label"]]
        if pred.startswith(gold):
            per_sample[i] = 1.0

    return {
        "accuracy": float(per_sample.mean()),
        "accuracy_ci95": _bootstrap_ci(per_sample),
        "num_samples": len(dataset),
    }


# ── APPS training data ───────────────────────────────────────────────────────


def load_apps_train(
    difficulty: str = "introductory",
    limit: int = 7000,
) -> list[dict[str, Any]]:
    dataset = load_dataset("codeparrot/apps", split="train")
    result = []
    for row in dataset:
        if row["difficulty"] != difficulty:
            continue
        try:
            io_pairs = json.loads(row["input_output"])
        except (json.JSONDecodeError, TypeError):
            continue
        tests = []
        for inp, out in zip(io_pairs.get("inputs", []), io_pairs.get("outputs", [])):
            tests.append(f"assert solution({repr(inp.strip())}) == {repr(out.strip())}")
        if not tests:
            continue
        result.append({
            "prompt": f'"""\n{row["question"].strip()}\n"""\n',
            "test_list": tests,
            "test_imports": [],
        })
        if len(result) >= limit:
            break
    return result


# ── LiveCodeBench ─────────────────────────────────────────────────────────────


def eval_livecodebench(
    engine: Any,
    lora_request: Any,
    n_samples: int,
    temperature: float,
) -> dict[str, Any]:
    dataset = load_dataset("livecodebench/code_generation_lite", split="test")
    code_eval = _get_code_eval()

    stop = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif"]
    gen_params = SamplingParams(
        max_tokens=512, temperature=temperature, n=n_samples,
        stop=stop, top_p=0.95,
    )

    prompts = []
    for row in dataset:
        prompt = row["question_content"]
        if row.get("starter_code"):
            prompt += f"\n{row['starter_code']}"
        prompts.append(prompt)

    outputs = engine.generate(prompts, gen_params, lora_request=lora_request)

    predictions: list[list[str]] = []
    references: list[str] = []
    for out, row in zip(outputs, dataset):
        completions = [extract_code(o.text) for o in out.outputs]
        predictions.append(completions)
        test_cases = json.loads(row["test"]) if isinstance(row["test"], str) else row["test"]
        test_code = "\n".join(
            f"assert solution({repr(tc['input'])}) == {repr(tc['output'])}"
            for tc in test_cases
        )
        references.append(test_code)

    pass_at_k_results, results_dict = code_eval.compute(
        references=references, predictions=predictions,
        k=[1, 10] if n_samples >= 10 else [1],
    )

    per_task_c = [
        sum(1 for _, r in task_results if r["passed"])
        for task_results in results_dict.values()
    ]
    per_task_n = [len(task_results) for task_results in results_dict.values()]
    per_task_pass1 = np.array([c / n for c, n in zip(per_task_c, per_task_n)])

    result: dict[str, Any] = {
        "pass@1": pass_at_k_results["pass@1"],
        "pass@1_ci95": _bootstrap_ci(per_task_pass1),
        "num_tasks": len(dataset),
    }

    if n_samples >= 10:
        per_task_pass10 = np.array([
            1.0 - math.comb(n - c, 10) / math.comb(n, 10)
            for c, n in zip(per_task_c, per_task_n)
        ])
        result["pass@10"] = pass_at_k_results["pass@10"]
        result["pass@10_ci95"] = _bootstrap_ci(per_task_pass10)

    return result
