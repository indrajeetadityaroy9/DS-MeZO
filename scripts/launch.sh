#!/bin/bash
# DS-MeZO launch script — locks GPU to max clocks for deterministic timing.
set -euo pipefail

# ── GPU clock locks (auto-detect max supported clocks) ────────────────────
sudo nvidia-smi -pm 1
MAX_GFX=$(nvidia-smi --query-supported-clocks=graphics --format=csv,noheader,nounits | sort -rn | head -1)
MAX_MEM=$(nvidia-smi --query-supported-clocks=memory --format=csv,noheader,nounits | sort -rn | head -1)
sudo nvidia-smi -lgc "${MAX_GFX},${MAX_GFX}"
sudo nvidia-smi -lmc "${MAX_MEM}"

# ── Environment ───────────────────────────────────────────────────────────
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
NUM_CPUS=$(python -c "import os; print(os.cpu_count())")
export OMP_NUM_THREADS="${NUM_CPUS}"
export MKL_NUM_THREADS="${NUM_CPUS}"
export TOKENIZERS_PARALLELISM=true

# CUDA allocator: expandable segments avoid fragmentation on 80GB H100.
# max_split_size prevents over-splitting large blocks.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

cd "$(dirname "$(dirname "$(readlink -f "$0")")")"
python -m scripts.train --config "$1" --prompts "$2"
