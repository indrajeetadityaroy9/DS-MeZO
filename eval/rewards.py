"""Reward computation: code execution scoring."""

from __future__ import annotations

import functools
import re
from typing import Callable

import evaluate


@functools.lru_cache(maxsize=1)
def _get_code_eval():
    return evaluate.load("code_eval")


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return "\n".join(blocks).strip() if blocks else text.strip()


def _score_code_solution(code: str, tests: list[str], imports: list[str]) -> float:
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


def make_exec_reward() -> tuple[Callable[[str], float], Callable[[list[str], list[str]], None]]:
    state: dict[str, list[str]] = {"tests": [], "imports": []}

    def set_problem(tests: list[str], imports: list[str]) -> None:
        state["tests"] = tests
        state["imports"] = imports

    def reward(text: str) -> float:
        return _score_code_solution(extract_code(text), state["tests"], state["imports"])

    return reward, set_problem
