"""PiSSA decomposition: pretrained model → W_res base model + A,B PEFT adapter.

Usage: python scripts/prepare_pissa.py <model_path> <output_dir> [target_modules...]

Processes one layer at a time to minimize memory.
Output:
  {output_dir}/residual/  — HuggingFace checkpoint with W_res replacing target weights
  {output_dir}/adapter/   — PEFT LoRA adapter (initial A, B matrices)
"""

import sys
import os
import json
import shutil
import torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ds_mezo.model_config import discover_layers

RANK = 16


def find_safetensor_files(model_path):
    """Return sorted list of safetensor shard files."""
    return sorted(Path(model_path).glob("*.safetensors"))


def build_key_to_file_map(model_path):
    """Map each tensor key to its safetensor shard file."""
    key_map = {}
    for f in find_safetensor_files(model_path):
        with safe_open(str(f), framework="pt") as sf:
            for key in sf.keys():
                key_map[key] = str(f)
    return key_map


def decompose(W0, rank):
    """PiSSA decomposition: W0 = A @ B + W_res (§3.1).
    torch.svd_lowrank returns (U, S, V) where V is n×q, not transposed."""
    U, S, V = torch.svd_lowrank(W0, q=rank, niter=2)
    sqrt_S = torch.sqrt(S[:rank])
    A = U[:, :rank] * sqrt_S.unsqueeze(0)
    B = (sqrt_S.unsqueeze(1) * V.T[:rank, :]).contiguous()
    W_res = W0 - A @ B
    return A, B, W_res


def main():
    model_path = sys.argv[1]
    output_dir = sys.argv[2]
    target_modules = sys.argv[3:] or ["q_proj", "v_proj"]
    residual_dir = os.path.join(output_dir, "residual")
    adapter_dir = os.path.join(output_dir, "adapter")

    print(f"Loading model from {model_path}")
    print(f"Target modules: {target_modules}")

    # Discover layers via meta-device introspection
    layer_specs = discover_layers(model_path, target_modules)
    print(f"Found {len(layer_specs)} target layers")

    # Build weight_key → LayerSpec lookup
    spec_by_key = {ls.weight_key: ls for ls in layer_specs}

    # Create output directories
    os.makedirs(residual_dir)
    os.makedirs(adapter_dir)

    # Copy non-safetensor files (config, tokenizer, etc.)
    for f in Path(model_path).iterdir():
        if f.suffix != ".safetensors" and f.name != ".gitattributes":
            dest = os.path.join(residual_dir, f.name)
            if f.is_file():
                shutil.copy2(str(f), dest)

    # Process shard by shard
    adapter_tensors = {}
    for shard_file in find_safetensor_files(model_path):
        shard_tensors = {}
        with safe_open(str(shard_file), framework="pt") as sf:
            for key in sf.keys():
                tensor = sf.get_tensor(key)

                if key in spec_by_key:
                    ls = spec_by_key[key]
                    print(f"  Decomposing {key} (rank={RANK})")
                    W0 = tensor.float()
                    A, B, W_res = decompose(W0, RANK)

                    shard_tensors[key] = W_res.to(tensor.dtype)

                    # PEFT convention: lora_A = B (r×d_in), lora_B = A (d_out×r)
                    adapter_tensors[f"{ls.peft_prefix}.lora_A.weight"] = B.bfloat16()
                    adapter_tensors[f"{ls.peft_prefix}.lora_B.weight"] = A.bfloat16()
                    del W0
                else:
                    shard_tensors[key] = tensor

        out_shard = os.path.join(residual_dir, Path(shard_file).name)
        save_file(shard_tensors, out_shard)
        del shard_tensors
        print(f"  Saved {out_shard}")

    # Save PEFT adapter
    save_file(adapter_tensors, os.path.join(adapter_dir, "adapter_model.safetensors"))

    adapter_config = {
        "peft_type": "LORA",
        "r": RANK,
        "lora_alpha": RANK,
        "target_modules": target_modules,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    print(f"\nResidual model saved to {residual_dir}")
    print(f"PiSSA adapter saved to {adapter_dir}")
    print(f"Decomposed {len(adapter_tensors) // 2} layers × {len(target_modules)} modules")


if __name__ == "__main__":
    main()
