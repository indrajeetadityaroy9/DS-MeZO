"""Benchmarks: MBPP pass@k, HumanEval pass@k, SST-2 accuracy, RTE accuracy,
LiveCodeBench pass@k."""

from __future__ import annotations

import json
import math
from typing import Any, Callable

import numpy as np
from scipy import stats

from datasets import load_dataset
from vllm import SamplingParams

from eval.data import build_mbpp_prompt, load_mbpp_train, load_apps_train
from eval.rewards import extract_code, _get_code_eval, _score_code_solution, make_exec_reward


def _bootstrap_ci(samples: np.ndarray) -> tuple[float, float]:
    result = stats.bootstrap(
        (samples,), np.mean,
        confidence_level=0.95, n_resamples=10000,
        random_state=np.random.default_rng(42), method="percentile",
    )
    return (float(result.confidence_interval.low),
            float(result.confidence_interval.high))


_CODE_STOP = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif"]


def _eval_code_gen(
    engine: Any,
    prompts: list[str],
    references: list[str],
    n_samples: int,
    temperature: float,
    lora_request: Any = None,
    stop: list[str] | None = None,
    prefix_fn: Callable[[str, str], str] | None = None,
) -> dict[str, Any]:
    """Shared code-gen evaluation: generate -> extract -> code_eval -> pass@k + CI."""
    code_eval = _get_code_eval()
    gen_params = SamplingParams(
        max_tokens=512, temperature=temperature, n=n_samples,
        stop=stop or _CODE_STOP, top_p=0.95,
    )
    outputs = engine.generate(prompts, gen_params, lora_request=lora_request)

    predictions: list[list[str]] = []
    for out, prompt in zip(outputs, prompts):
        completions = [extract_code(o.text) for o in out.outputs]
        if prefix_fn is not None:
            completions = [prefix_fn(prompt, c) for c in completions]
        predictions.append(completions)

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
        "num_tasks": len(prompts),
    }

    if n_samples >= 10:
        per_task_pass10 = np.array([
            1.0 - math.comb(n - c, 10) / math.comb(n, 10)
            for c, n in zip(per_task_c, per_task_n)
        ])
        result["pass@10"] = pass_at_k_results["pass@10"]
        result["pass@10_ci95"] = _bootstrap_ci(per_task_pass10)

    return result


def eval_mbpp(
    engine: Any, lora_request: Any, n_samples: int, temperature: float,
) -> dict[str, Any]:
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    prompts = [build_mbpp_prompt(row) for row in dataset]
    references = [
        "\n".join(row["test_imports"]) + "\n" + "\n".join(row["test_list"])
        for row in dataset
    ]
    return _eval_code_gen(engine, prompts, references, n_samples, temperature,
                          lora_request=lora_request)


def eval_humaneval(
    engine: Any, lora_request: Any, n_samples: int, temperature: float,
) -> dict[str, Any]:
    dataset = load_dataset("openai_humaneval", split="test")
    prompts = [row["prompt"] for row in dataset]
    references = [row["test"] + f"\ncheck({row['entry_point']})" for row in dataset]
    return _eval_code_gen(engine, prompts, references, n_samples, temperature,
                          lora_request=lora_request,
                          prefix_fn=lambda prompt, code: prompt + code)


# -- SuperGLUE: SST-2 + RTE ------------------------------------------------


def _build_classification_prompt(
    examples: list[dict], query: str, template_fn: Any,
) -> str:
    parts = [template_fn(ex, with_label=True) for ex in examples]
    parts.append(template_fn(query, with_label=False))
    return "\n\n".join(parts)


def _eval_classification(
    engine: Any,
    lora_request: Any,
    n_shot: int,
    dataset_name: str,
    dataset_config: str,
    label_map: dict[int, str],
    template_fn: Callable[..., str],
) -> dict[str, Any]:
    dataset = load_dataset(dataset_name, dataset_config, split="validation")
    train = load_dataset(dataset_name, dataset_config, split="train")
    demos = [dict(row) for row in train.select(range(n_shot))]
    prompts = [
        _build_classification_prompt(demos, dict(row), template_fn)
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


def eval_sst2(
    engine: Any, lora_request: Any, n_shot: int,
) -> dict[str, Any]:
    label_map = {0: "negative", 1: "positive"}

    def template(row: dict, with_label: bool = True) -> str:
        text = f"Sentence: {row['sentence']}\nSentiment:"
        if with_label:
            text += f" {label_map[row['label']]}"
        return text

    return _eval_classification(
        engine, lora_request, n_shot,
        "nyu-mll/glue", "sst2", label_map, template,
    )


def eval_rte(
    engine: Any, lora_request: Any, n_shot: int,
) -> dict[str, Any]:
    label_map = {0: "yes", 1: "no"}

    def template(row: dict, with_label: bool = True) -> str:
        text = f"Premise: {row['sentence1']}\nHypothesis: {row['sentence2']}\nEntailment:"
        if with_label:
            text += f" {label_map[row['label']]}"
        return text

    return _eval_classification(
        engine, lora_request, n_shot,
        "nyu-mll/glue", "rte", label_map, template,
    )


# -- LiveCodeBench ---------------------------------------------------------


def eval_livecodebench(
    engine: Any, lora_request: Any, n_samples: int, temperature: float,
) -> dict[str, Any]:
    dataset = load_dataset("livecodebench/code_generation_lite", split="test")
    prompts = []
    references = []
    for row in dataset:
        prompt = row["question_content"]
        if row.get("starter_code"):
            prompt += f"\n{row['starter_code']}"
        prompts.append(prompt)
        test_cases = json.loads(row["test"])
        references.append("\n".join(
            f"assert solution({repr(tc['input'])}) == {repr(tc['output'])}"
            for tc in test_cases
        ))
    return _eval_code_gen(engine, prompts, references, n_samples, temperature,
                          lora_request=lora_request)
