"""DS-MeZO training entry point.

Sets GPU environment (clock locking, CUDA allocator, threading) and runs
training. Replaces the former scripts/launch.sh.
"""

from __future__ import annotations

import os
import subprocess

# Environment must be set before importing torch/vLLM.
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
_ncpus = str(os.cpu_count())
os.environ["OMP_NUM_THREADS"] = _ncpus
os.environ["MKL_NUM_THREADS"] = _ncpus

import argparse
import json
from pathlib import Path

import yaml
from ds_mezo import build_controller


def _lock_gpu_clocks() -> None:
    """Lock GPU to max supported clocks for deterministic timing."""
    subprocess.run(["sudo", "nvidia-smi", "-pm", "1"], check=True,
                   capture_output=True)
    max_gfx = subprocess.run(
        ["nvidia-smi", "--query-supported-clocks=graphics",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split("\n")
    max_mem = subprocess.run(
        ["nvidia-smi", "--query-supported-clocks=memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split("\n")
    gfx = max(int(x.strip()) for x in max_gfx)
    mem = max(int(x.strip()) for x in max_mem)
    subprocess.run(["sudo", "nvidia-smi", "-lgc", f"{gfx},{gfx}"],
                   check=True, capture_output=True)
    subprocess.run(["sudo", "nvidia-smi", "-lmc", str(mem)],
                   check=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="DS-MeZO training")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    parser.add_argument("--prompts", type=Path, required=True, help="Path to prompts JSONL")
    parser.add_argument("--lock-clocks", action="store_true",
                        help="Lock GPU to max clocks (requires sudo)")
    args = parser.parse_args()

    if args.lock_clocks:
        _lock_gpu_clocks()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.prompts) as f:
        prompts = [json.loads(line)["prompt"] for line in f]

    calibration = None if config.get("resume_from") else prompts[0]
    _, _, controller, _, _ = build_controller(
        model_path=config["model_path"],
        adapter_path=config["adapter_path"],
        output_dir=config["output_dir"],
        total_steps=config["total_steps"],
        calibration_prompt=calibration,
        extra_config=config,
    )
    controller.train(prompts)


if __name__ == "__main__":
    main()
