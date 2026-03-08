"""Shared utilities for DS-MeZO evaluation scripts."""


def extract_code(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    if "```python" in text:
        return text.split("```python")[1].split("```")[0]
    if "```" in text:
        return text.split("```")[1].split("```")[0]
    return text
