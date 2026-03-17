import math

import numpy as np
from scipy import stats

from datasets import load_dataset
from vllm import SamplingParams

from eval.rewards import build_mbpp_prompt, extract_code, _get_code_eval


def _bootstrap_ci(samples):
    result = stats.bootstrap(
        (samples,), np.mean,
        confidence_level=0.95, n_resamples=10000,
        random_state=np.random.default_rng(42), method="percentile",
    )
    return (float(result.confidence_interval.low),
            float(result.confidence_interval.high))


_CODE_STOP = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif"]


def _eval_code_gen(engine, prompts, references, n_samples, temperature,
                   lora_request=None, prefix_fn=None):
    code_eval = _get_code_eval()
    gen_params = SamplingParams(
        max_tokens=512, temperature=temperature, n=n_samples,
        stop=_CODE_STOP, top_p=0.95,
    )
    outputs = engine.generate(prompts, gen_params, lora_request=lora_request)

    predictions = []
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

    result = {
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


def eval_mbpp(engine, lora_request, n_samples, temperature):
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    prompts = [build_mbpp_prompt(row) for row in dataset]
    references = [
        "\n".join(row["test_imports"]) + "\n" + "\n".join(row["test_list"])
        for row in dataset
    ]
    return _eval_code_gen(engine, prompts, references, n_samples, temperature,
                          lora_request=lora_request)


def eval_humaneval(engine, lora_request, n_samples, temperature):
    dataset = load_dataset("openai_humaneval", split="test")
    prompts = [row["prompt"] for row in dataset]
    references = [row["test"] + f"\ncheck({row['entry_point']})" for row in dataset]
    return _eval_code_gen(engine, prompts, references, n_samples, temperature,
                          lora_request=lora_request,
                          prefix_fn=lambda prompt, code: prompt + code)
