"""Shared utilities for DS-MeZO evaluation scripts."""

from __future__ import annotations

import math
import re


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


class ExecReward:
    """Execution-based reward: fraction of test assertions passing."""

    def __init__(self) -> None:
        self.tests: list[str] = []
        self.imports: list[str] = []

    def set_problem(self, tests: list[str], imports: list[str]) -> None:
        self.tests = tests
        self.imports = imports

    def __call__(self, text: str) -> float:
        code = extract_code(text)
        import_block = "\n".join(self.imports)
        passed = 0
        for test in self.tests:
            try:
                exec(f"{import_block}\n{code}\n{test}", {})
                passed += 1
            except Exception:
                pass
        return passed / len(self.tests)
