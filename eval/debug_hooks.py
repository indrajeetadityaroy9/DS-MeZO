"""Debug: check what module names vLLM's worker model exposes."""
import os
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import sys
sys.path.insert(0, "/home/ubuntu/DS-MeZO")

from vllm import LLM, SamplingParams
import re

def list_modules(worker, target_modules):
    model = worker.get_model()
    found = []
    for name, module in model.named_modules():
        if any(t in name for t in target_modules):
            found.append(name)
    return found

llm = LLM(
    model="/dev/shm/pissa_prep/residual",
    dtype="bfloat16",
    gpu_memory_utilization=0.92,
    enable_lora=True,
    max_lora_rank=64,
    max_num_seqs=8,
)

results = llm.llm_engine.collective_rpc(list_modules, args=(["q_proj", "v_proj"],))
for name in sorted(results[0])[:30]:
    print(name)
print(f"\nTotal: {len(results[0])}")
