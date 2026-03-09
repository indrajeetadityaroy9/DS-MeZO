"""Shared utilities for DS-MeZO evaluation scripts."""

from __future__ import annotations

import math
import re
from typing import Callable


def extract_code(text: str) -> str:
    """Extract Python code from markdown-fenced blocks.

    Finds all ```python ... ``` blocks and concatenates them.
    Falls back to unfenced ``` blocks, then raw text.
    """
    # Fenced Python blocks
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return "\n".join(blocks).strip()

    # Unfenced blocks
    blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return "\n".join(blocks).strip()

    return text.strip()


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021, Codex paper Eq. 1).

    Formula: 1 - C(n-c, k) / C(n, k). Caller guarantees n >= k.
    """
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def make_exec_reward() -> tuple[Callable[[str], float], Callable[[list[str], list[str]], None]]:
    """Execution-based reward: fraction of test assertions passing.

    Returns (reward_fn, set_problem_fn).
    reward_fn(text) -> float: score generated code against current tests.
    set_problem_fn(tests, imports): set tests/imports for current problem.
    """
    state: dict[str, list[str]] = {"tests": [], "imports": []}

    def set_problem(tests: list[str], imports: list[str]) -> None:
        state["tests"] = tests
        state["imports"] = imports

    def reward(text: str) -> float:
        code = extract_code(text)
        import_block = "\n".join(state["imports"])
        passed = 0
        for test in state["tests"]:
            try:
                exec(f"{import_block}\n{code}\n{test}", {})
                passed += 1
            except Exception:
                pass
        return passed / len(state["tests"])

    return reward, set_problem
