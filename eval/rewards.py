import functools
import json
import os
import re

import evaluate
from datasets import load_dataset


@functools.lru_cache(maxsize=1)
def _get_code_eval():
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    return evaluate.load("code_eval")


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return "\n".join(blocks).strip() if blocks else text.strip()


def _score_code_solution(code, tests, imports):
    code_eval = _get_code_eval()
    import_block = "\n".join(imports)
    references = [f"{import_block}\n{test}" for test in tests]
    predictions = [[code]] * len(references)
    _, results_dict = code_eval.compute(
        references=references, predictions=predictions,
        k=[1], num_workers=1, timeout=3.0,
    )
    passed = sum(
        1 for task_results in results_dict.values()
        for _, r in task_results if r["passed"]
    )
    return passed / len(tests)


def make_exec_reward():
    state = {"tests": [], "imports": []}

    def set_problem(tests, imports):
        state["tests"] = tests
        state["imports"] = imports

    def reward(text):
        return _score_code_solution(extract_code(text), state["tests"], state["imports"])

    return reward, set_problem


def build_mbpp_prompt(row):
    return f'"""\n{row["prompt"]}\n{row["test_list"][0]}\n"""\n'


def load_mbpp_train():
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    return [
        {
            "prompt": build_mbpp_prompt(row),
            "test_list": row["test_list"],
            "test_imports": row["test_imports"],
        }
        for row in dataset
    ]


def load_apps_train(difficulty="introductory", limit=7000):
    dataset = load_dataset("codeparrot/apps", split="train")
    result = []
    for row in dataset:
        if row["difficulty"] != difficulty:
            continue
        io_pairs = json.loads(row["input_output"])
        tests = [
            f"assert solution({repr(inp.strip())}) == {repr(out.strip())}"
            for inp, out in zip(io_pairs["inputs"], io_pairs["outputs"])
        ]
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
