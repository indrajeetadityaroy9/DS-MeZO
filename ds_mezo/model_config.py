"""Model-agnostic layer discovery via PyTorch meta-device introspection.

Uses AutoConfig + AutoModelForCausalLM on torch.device("meta") to discover
trainable linear layers without loading weights (zero memory). Works for any
HuggingFace causal LM architecture.
"""

from dataclasses import dataclass
import torch
from transformers import AutoConfig, AutoModelForCausalLM


@dataclass
class LayerSpec:
    layer_idx: int
    module_name: str    # "q_proj", "v_proj", etc.
    weight_key: str     # "model.layers.5.self_attn.q_proj.weight"
    peft_prefix: str    # "base_model.model.model.layers.5.self_attn.q_proj"


def discover_layers(model_path, target_modules):
    """Discover trainable layers via PyTorch model introspection.

    Instantiates an empty model on meta device (zero memory), iterates
    named_modules() to find nn.Linear layers matching target_modules,
    and returns LayerSpec for each (layer_idx, module) pair.
    """
    config = AutoConfig.from_pretrained(model_path)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    specs = []
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

    del model
    return sorted(specs, key=lambda s: (s.layer_idx, s.module_name))
