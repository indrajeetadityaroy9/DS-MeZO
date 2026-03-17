import argparse
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from ds_mezo.model_config import svd_power_iters

_SVD_NITER = svd_power_iters()


def main():
    parser = argparse.ArgumentParser(description="PiSSA decomposition")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--targets", nargs="+", default=["q_proj", "v_proj"])
    args = parser.parse_args()

    model_path = args.model
    output_dir = args.output
    rank = args.rank
    target_modules = args.targets
    residual_dir = output_dir / "residual"
    adapter_dir = output_dir / "adapter"

    print(f"Loading model from {model_path}")
    print(f"Rank: {rank} | Target modules: {target_modules}")
    print(f"SVD power iterations: {_SVD_NITER}")

    residual_dir.mkdir(parents=True)
    adapter_dir.mkdir(parents=True)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), device_map="cuda",
    )

    pissa_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=target_modules,
        init_lora_weights=f"pissa_niter_{_SVD_NITER}",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, pissa_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total_params:,} parameters")

    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    print(f"PiSSA adapter saved to {adapter_dir}")

    base_model = model.unload()
    base_model.save_pretrained(str(residual_dir), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    tokenizer.save_pretrained(str(residual_dir))

    print(f"Residual model saved to {residual_dir}")
    print(f"Decomposed {trainable // (2 * rank)} layers x {len(target_modules)} modules")


if __name__ == "__main__":
    main()
