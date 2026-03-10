"""DS-MeZO training entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import PeftConfig
from vllm import LLM

from ds_mezo.model_config import discover_layers
from ds_mezo.backend import VLLMBackend
from ds_mezo.controller import DSMeZO_Controller


def main() -> None:
    parser = argparse.ArgumentParser(description="DS-MeZO training")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    parser.add_argument("--prompts", type=Path, required=True, help="Path to prompts JSONL")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.prompts) as f:
        prompts = [json.loads(line)["prompt"] for line in f]

    adapter_path = Path(config["adapter_path"])
    peft_config = PeftConfig.from_pretrained(str(adapter_path))
    rank = peft_config.r
    target_modules = list(peft_config.target_modules)

    llm = LLM(
        model=config["model_path"],
        dtype="bfloat16",
        gpu_memory_utilization=0.95,
        enable_lora=True,
        max_lora_rank=max(64, rank),
        enforce_eager=True,
    )

    layer_specs = discover_layers(config["model_path"], target_modules)
    staging_dir = config.get("staging_dir", "/dev/shm/ds_mezo")
    backend = VLLMBackend(llm, layer_specs, rank, staging_dir=staging_dir)
    controller = DSMeZO_Controller(backend, layer_specs, config)
    controller._calibrate_activation_bases_full([prompts[0]])
    controller.train(prompts)


if __name__ == "__main__":
    main()
