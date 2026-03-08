"""PiSSA decomposition: pretrained model → W_res base model + A,B PEFT adapter.

Usage: python scripts/prepare_pissa.py <model_path> <output_dir>

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

RANK = 16
TARGET_MODULES = ["q_proj", "v_proj"]


def find_safetensor_files(model_path):
    """Return sorted list of safetensor shard files."""
    p = Path(model_path)
    return sorted(p.glob("*.safetensors"))


def build_key_to_file_map(model_path):
    """Map each tensor key to its safetensor shard file."""
    key_map = {}
    for f in find_safetensor_files(model_path):
        with safe_open(str(f), framework="pt") as sf:
            for key in sf.keys():
                key_map[key] = str(f)
    return key_map


def get_num_layers(key_map):
    """Count transformer layers by finding max layer index in keys."""
    max_idx = -1
    for key in key_map:
        parts = key.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                max_idx = max(max_idx, int(parts[i + 1]))
    return max_idx + 1


def layer_key(layer_idx, module_name):
    """Map (layer_idx, module_name) → safetensors key."""
    return f"model.layers.{layer_idx}.self_attn.{module_name}.weight"


def decompose(W0, rank):
    """PiSSA decomposition: W0 = A @ B + W_res (§3.1).
    torch.svd_lowrank returns (U, S, V) where V is n×q, not transposed."""
    U, S, V = torch.svd_lowrank(W0, q=rank, niter=2)
    sqrt_S = torch.sqrt(S[:rank])
    A = U[:, :rank] * sqrt_S.unsqueeze(0)             # d_out × r
    B = (sqrt_S.unsqueeze(1) * V.T[:rank, :]).contiguous()  # r × d_in
    W_res = W0 - A @ B
    return A, B, W_res


def main():
    model_path = sys.argv[1]
    output_dir = sys.argv[2]
    residual_dir = os.path.join(output_dir, "residual")
    adapter_dir = os.path.join(output_dir, "adapter")

    print(f"Loading model from {model_path}")
    key_map = build_key_to_file_map(model_path)
    num_layers = get_num_layers(key_map)
    print(f"Found {num_layers} layers")

    # Create output directories (fresh — output_dir should not pre-exist)
    os.makedirs(residual_dir)
    os.makedirs(adapter_dir)

    # Copy non-safetensor files (config, tokenizer, etc.)
    for f in Path(model_path).iterdir():
        if f.suffix != ".safetensors" and f.name != ".gitattributes":
            dest = os.path.join(residual_dir, f.name)
            if f.is_file():
                shutil.copy2(str(f), dest)

    # Load all tensors, replace target weights with W_res, save A/B as adapter
    adapter_tensors = {}
    target_modules_list = []

    # Process shard by shard to minimize memory
    for shard_file in find_safetensor_files(model_path):
        shard_tensors = {}
        with safe_open(str(shard_file), framework="pt") as sf:
            for key in sf.keys():
                tensor = sf.get_tensor(key)

                # Check if this is a target module weight
                is_target = False
                for module_name in TARGET_MODULES:
                    for layer_idx in range(num_layers):
                        expected_key = layer_key(layer_idx, module_name)
                        if key == expected_key:
                            print(f"  Decomposing {key} (rank={RANK})")
                            W0 = tensor.float()
                            A, B, W_res = decompose(W0, RANK)

                            # Store W_res in residual model
                            shard_tensors[key] = W_res.to(tensor.dtype)

                            # Store A, B for PEFT adapter
                            prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{module_name}"
                            # PEFT convention: lora_A = B (r×d_in), lora_B = A (d_out×r)
                            adapter_tensors[f"{prefix}.lora_A.weight"] = B.bfloat16()
                            adapter_tensors[f"{prefix}.lora_B.weight"] = A.bfloat16()

                            if module_name not in target_modules_list:
                                target_modules_list.append(module_name)

                            is_target = True
                            del W0
                            break
                    if is_target:
                        break

                if not is_target:
                    shard_tensors[key] = tensor

        # Save shard (modified or original)
        out_shard = os.path.join(residual_dir, Path(shard_file).name)
        save_file(shard_tensors, out_shard)
        del shard_tensors
        print(f"  Saved {out_shard}")

    # Save PEFT adapter
    save_file(adapter_tensors, os.path.join(adapter_dir, "adapter_model.safetensors"))

    # Write adapter config
    adapter_config = {
        "peft_type": "LORA",
        "r": RANK,
        "lora_alpha": RANK,
        "target_modules": target_modules_list,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    print(f"\nResidual model saved to {residual_dir}")
    print(f"PiSSA adapter saved to {adapter_dir}")
    print(f"Decomposed {len(adapter_tensors) // 2} layers × {len(TARGET_MODULES)} modules")


if __name__ == "__main__":
    main()
