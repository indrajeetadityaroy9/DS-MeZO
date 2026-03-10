"""Evaluation utilities: code extraction and execution-based reward."""

from __future__ import annotations

import functools
import re
from typing import Callable

import evaluate


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
