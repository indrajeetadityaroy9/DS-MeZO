"""DS-MeZO training entry point.

Sets GPU environment (clock locking, CUDA allocator, threading) and runs
training. Replaces the former scripts/launch.sh.
"""

from __future__ import annotations

import os
import subprocess

# Environment must be set before importing torch/vLLM.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                       "expandable_segments:True,max_split_size_mb:512")
_ncpus = str(os.cpu_count())
os.environ.setdefault("OMP_NUM_THREADS", _ncpus)
os.environ.setdefault("MKL_NUM_THREADS", _ncpus)

import argparse
import json
from pathlib import Path

import yaml
from peft import PeftConfig
from ds_mezo.model_config import discover_layers
from ds_mezo.backend import VLLMBackend, create_engine
from ds_mezo.controller import DSMeZO_Controller


def _lock_gpu_clocks() -> None:
    """Lock GPU to max supported clocks for deterministic timing."""
    try:
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
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass  # No GPU or no sudo — continue without clock locking


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

    adapter_path = Path(config["adapter_path"])
    peft_config = PeftConfig.from_pretrained(str(adapter_path))
    rank = peft_config.r
    target_modules = list(peft_config.target_modules)

    llm = create_engine(config["model_path"], rank)

    layer_specs = discover_layers(config["model_path"], target_modules)
    staging_dir = config.get("staging_dir", "/dev/shm/ds_mezo")
    backend = VLLMBackend(llm, layer_specs, rank, staging_dir=staging_dir)
    controller = DSMeZO_Controller(backend, layer_specs, config)
    if not config.get("resume_from"):
        controller._calibrate_activation_bases_full([prompts[0]])
    controller.train(prompts)


if __name__ == "__main__":
    main()
