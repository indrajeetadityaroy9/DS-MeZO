"""Benchmarks: MBPP pass@k, HumanEval pass@k, SST-2 accuracy, RTE accuracy,
LiveCodeBench pass@k, APPS data loading."""

from __future__ import annotations

import functools
import json
import math
import re
from typing import Any, Callable

import numpy as np
from scipy import stats

import evaluate
from datasets import load_dataset
from vllm import SamplingParams

@functools.lru_cache(maxsize=1)
def _get_code_eval():
    return evaluate.load("code_eval")


def extract_code(text: str) -> str:
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return "\n".join(blocks).strip()

    blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return "\n".join(blocks).strip()

    return text.strip()


def make_exec_reward() -> tuple[Callable[[str], float], Callable[[list[str], list[str]], None]]:
    state: dict[str, list[str]] = {"tests": [], "imports": []}
    code_eval = _get_code_eval()

    def set_problem(tests: list[str], imports: list[str]) -> None:
        state["tests"] = tests
        state["imports"] = imports

    def reward(text: str) -> float:
        code = extract_code(text)
        import_block = "\n".join(state["imports"])
        references = [f"{import_block}\n{test}" for test in state["tests"]]
        predictions = [[code]] * len(references)
        _, results_dict = code_eval.compute(
            references=references, predictions=predictions,
            k=[1], num_workers=1, timeout=3.0,
        )
        passed = sum(
            1 for task_results in results_dict.values()
            for _, r in task_results if r["passed"]
        )
        return passed / len(state["tests"])

    return reward, set_problem


def _bootstrap_ci(samples: np.ndarray) -> tuple[float, float]:
    result = stats.bootstrap(
        (samples,), np.mean,
        confidence_level=0.95, n_resamples=10000,
        random_state=np.random.default_rng(42), method="percentile",
    )
    return (float(result.confidence_interval.low),
            float(result.confidence_interval.high))


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
    """Shared code-gen evaluation: generate → extract → code_eval → pass@k + CI."""
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


# ── Setup helpers ────────────────────────────────────────────────────────────

from pathlib import Path

from peft import PeftConfig

from ds_mezo.model_config import discover_layers
from ds_mezo.backend import VLLMBackend, create_engine
from ds_mezo.controller import DSMeZO_Controller


def load_adapter_config(adapter_path: Path | str) -> tuple[int, list[str]]:
    """Read rank and target_modules from PeftConfig."""
    peft_config = PeftConfig.from_pretrained(str(adapter_path))
    return peft_config.r, list(peft_config.target_modules)


def setup_controller(
    model_path: Path | str,
    adapter_path: Path | str,
    output_dir: Path | str,
    total_steps: int,
    score_fn: Callable | None = None,
    calibration_prompt: str | None = None,
    engine: Any = None,
    layer_specs: list | None = None,
    rank: int | None = None,
) -> tuple[Any, VLLMBackend, DSMeZO_Controller, int, list]:
    """Build vLLM engine (or reuse), create backend + controller, calibrate.
    Returns (engine, backend, controller, rank, layer_specs)."""
    if rank is None:
        rank, target_modules = load_adapter_config(adapter_path)
    else:
        _, target_modules = load_adapter_config(adapter_path)
    if engine is None:
        engine = create_engine(model_path, rank)
    if layer_specs is None:
        layer_specs = discover_layers(model_path, target_modules)
    backend = VLLMBackend(engine, layer_specs, rank)
    config: dict[str, Any] = {
        "output_dir": str(output_dir),
        "adapter_path": str(adapter_path),
        "total_steps": total_steps,
    }
    if score_fn is not None:
        config["score_fn"] = score_fn
    controller = DSMeZO_Controller(backend, layer_specs, config)
    if calibration_prompt is not None:
        controller._calibrate_activation_bases_full([calibration_prompt])
    return engine, backend, controller, rank, layer_specs
