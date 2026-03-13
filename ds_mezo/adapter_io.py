"""Adapter serialization: save/load PiSSA adapters in PEFT format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig
from safetensors.torch import save_file


def write_adapter_config(
    adapter_dir: Path, rank: int, target_modules: list[str],
) -> None:
    config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    config.save_pretrained(str(adapter_dir))


def save_peft_adapter(
    A_list: list[torch.Tensor],
    B_list: list[torch.Tensor],
    adapter_dir: Path,
    layers: list[Any],
    bf16_cache: dict[str, torch.Tensor],
) -> None:
    """Serialize A, B as PEFT adapter. lora_A.weight = B, lora_B.weight = A (PiSSA↔PEFT swap)."""
    tensors: dict[str, torch.Tensor] = {}
    for layer_idx, (A_l, B_l) in enumerate(zip(A_list, B_list)):
        prefix = layers[layer_idx].peft_prefix
        key_a = f"{prefix}.lora_A.weight"
        key_b = f"{prefix}.lora_B.weight"
        if key_a not in bf16_cache:
            bf16_cache[key_a] = torch.empty_like(B_l, dtype=torch.bfloat16)
            bf16_cache[key_b] = torch.empty_like(A_l, dtype=torch.bfloat16)
        bf16_cache[key_a].copy_(B_l)
        bf16_cache[key_b].copy_(A_l)
        tensors[key_a] = bf16_cache[key_a]
        tensors[key_b] = bf16_cache[key_b]

    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))
