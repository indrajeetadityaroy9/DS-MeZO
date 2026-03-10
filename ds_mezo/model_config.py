"""Layer discovery via meta-device introspection (zero memory)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM


@dataclass
class LayerSpec:
    layer_idx: int
    module_name: str    # "q_proj", "v_proj", etc.
    weight_key: str     # "model.layers.5.self_attn.q_proj.weight"
    peft_prefix: str    # "base_model.model.model.layers.5.self_attn.q_proj"


def discover_layers(
    model_path: str | Path, target_modules: list[str],
) -> list[LayerSpec]:
    config = AutoConfig.from_pretrained(str(model_path))
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    specs: list[LayerSpec] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        suffix = name.rsplit(".", 1)[-1]
        if suffix not in target_modules:
            continue
        parts = name.split(".")
        layer_idx = int(parts[parts.index("layers") + 1])
        specs.append(LayerSpec(
            layer_idx=layer_idx,
            module_name=suffix,
            weight_key=name + ".weight",
            peft_prefix="base_model.model." + name,
        ))

    return sorted(specs, key=lambda s: (s.layer_idx, s.module_name))
