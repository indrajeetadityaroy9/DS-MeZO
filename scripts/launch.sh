#!/bin/bash
sudo nvidia-smi -lgc 1980,1980 && sudo nvidia-smi -lmc 2619
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(dirname "$(dirname "$(readlink -f "$0")")")"
python scripts/train.py --config "$1" --prompts "$2"
