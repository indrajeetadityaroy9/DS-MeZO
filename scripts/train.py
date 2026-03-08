"""DS-MeZO training entry point.

Usage: bash scripts/launch.sh <config.yaml> <prompts.jsonl>

Config YAML only requires model_path and adapter_path.
All other hyperparameters use spec defaults from controller.DEFAULTS.
"""

import os
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import sys
import json
import yaml
from vllm import LLM
from ds_mezo.controller import DSMeZO_Controller


def main():
    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)
    with open(sys.argv[2]) as f:
        prompts = [json.loads(line)["prompt"] for line in f]

    llm = LLM(
        model=config["model_path"],
        dtype="bfloat16",
        gpu_memory_utilization=0.92,
        enable_lora=True,
        max_lora_rank=64,
        max_num_seqs=8,
    )

    controller = DSMeZO_Controller(llm, config)
    controller.train(prompts)


if __name__ == "__main__":
    main()
