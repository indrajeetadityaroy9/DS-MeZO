"""Training data loaders: MBPP, APPS."""

from __future__ import annotations

import json
from typing import Any

from datasets import load_dataset


def build_mbpp_prompt(row: dict) -> str:
    return f'"""\n{row["prompt"]}\n{row["test_list"][0]}\n"""\n'


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


def load_apps_train(
    difficulty: str = "introductory",
    limit: int = 7000,
) -> list[dict[str, Any]]:
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
