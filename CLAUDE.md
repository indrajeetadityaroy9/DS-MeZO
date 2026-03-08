# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

DS-MeZO: Zeroth-order RL post-training for LLMs on a single GPU. Uses SPSA gradient estimation on PiSSA (LoRA) adapters with forward passes only — no backpropagation. Targets non-differentiable rewards (code execution, proof verification). Built on vLLM for inference and Triton for fused optimizer kernels.

## Commands

All scripts use `python -m` execution from the project root. No `sys.path` manipulation.

```bash
# PiSSA decomposition (prerequisite — creates residual model + adapter)
python -m scripts.prepare_pissa --model <model_path> --output <output_dir> [--targets q_proj v_proj]

# Training (requires GPU, vLLM — launch.sh sets environment variables)
bash scripts/launch.sh <config.yaml> <prompts.jsonl>

# Evaluation scripts (each requires GPU + PiSSA-prepared model)
python -m eval.rl_bench_eval --model-path <path> --adapter-path <path> --output-dir <path>
python -m eval.sft_eval --model-path <path> --adapter-path <path> --output-dir <path>
python -m eval.ablations --model-path <path> --adapter-path <path> --output-dir <path>

# Tests (require CUDA GPU)
python -m tests.test_evaluation
```

## Architecture

**Two-process design**: The controller (FP32 master weights, optimizer state) and vLLM engine (BF16 inference) communicate via adapter serialization to a configurable staging directory (defaults to `/dev/shm/ds_mezo/`).

```
Controller (controller.py)          Backend (backend.py)
  FP32 master weights                 vLLM engine (BF16)
  SPSA perturbation                   S-LoRA hot-swap
  ZO-Muon optimizer                   Prefill-only scoring
  RLOO advantage computation          Activation hook capture
       |                                    |
       +--- serialize adapters via staging_dir ---+
```

### Core modules (`ds_mezo/`)

- **`controller.py`** — Unified training loop. Single `step()` handles both SFT and RL via `hybrid_switch_step`. Config via `DSMeZOConfig` dataclass. Layer state via `LayerState` dataclass. Algorithm-only — no vLLM imports.
- **`backend.py`** — vLLM isolation layer. Handles adapter serialization (PEFT format), generation, prefill-only logprob scoring, and activation extraction via forward hooks. All vLLM-specific code lives here.
- **`kernels.py`** — Two Triton kernels: `zo_muon_update` (fused gradient + momentum-aligned masking + momentum + 5-iteration Newton-Schulz orthogonalization + param update) and `fused_perturb_dual` (compute θ+ and θ- in one pass). Dispatches tall vs wide kernel based on matrix shape.
- **`model_config.py`** — Model-agnostic layer discovery via `torch.device("meta")` introspection. Returns `LayerSpec` dataclass with PEFT naming conventions.
- **`__init__.py`** — Public API exports: `DSMeZO_Controller`, `DSMeZOConfig`, `VLLMBackend`, `LayerSpec`, `discover_layers`.

### Key conventions

- **PiSSA ↔ PEFT naming**: PiSSA uses `A` (d_out × r) and `B` (r × d_in). PEFT stores `lora_A.weight = B` and `lora_B.weight = A`. This swap appears in `controller.py` init, `backend.py` serialization, and `prepare_pissa.py`.
- **vLLM module merges**: vLLM fuses `q_proj`/`k_proj`/`v_proj` → `qkv_proj` and `gate_proj`/`up_proj` → `gate_up_proj`. The backend's `hook_map` handles this translation for activation capture.
- **Adapter staging**: Adapters serialize to a configurable `staging_dir` (default `/dev/shm/ds_mezo/`, tmpfs) for zero-copy vLLM loading. Two adapter slots: `adapter_pos` (θ+ε) and `adapter_neg` (θ-ε) for SPSA. Staging is ephemeral; persistent artifacts go to `output_dir`.
- **Directory convention** (HuggingFace standard): `output_dir` is the single root for all persistent artifacts (checkpoints under `{output_dir}/checkpoints/`, eval results as `{output_dir}/*.json`). `staging_dir` is ephemeral (tmpfs) for vLLM adapter hot-swap. All paths are required CLI arguments — no hardcoded defaults.
- **Config accepts both dict and DSMeZOConfig**: `DSMeZO_Controller` init accepts either; dicts are converted via `DSMeZOConfig.from_dict()` which silently ignores unknown keys.

### Algorithm pipeline (RL mode, one step)

1. **Explore**: Generate N candidates under unperturbed weights, compute RLOO advantages
2. **Activation tracking**: Power iteration update of per-layer activation bases (full SVD recalibration on drift)
3. **AGZO perturbation**: Project random vectors into activation subspace for B; into B's column space for A
4. **Dual perturbation**: Fused kernel computes θ+ = base + εz and θ- = base - εz
5. **Contrastive scoring**: Score winner/loser trajectories under θ+ and θ- with asymmetric gradient regularization
6. **ZO-Muon update**: Fused kernel applies SPSA gradient → momentum-aligned masking → momentum accumulation → Newton-Schulz orthogonalization → parameter update
7. **Schedule**: Cosine LR, adaptive epsilon, cosine temperature annealing

## Environment

- Requires NVIDIA GPU (H100 target, any CUDA GPU for tests)
- Python 3.12+, PyTorch 2.10, Triton 3.6, vLLM 0.17
- `VLLM_ALLOW_INSECURE_SERIALIZATION=1` required (set by caller — see `scripts/launch.sh`)
- Config files use YAML; `model_path`, `adapter_path`, and `output_dir` are required, all else defaults from `DSMeZOConfig`

## Design docs

- **`objectives.md`** — Authoritative specification and single source of truth. Canonical mechanism-to-code mapping, execution paths, discrepancy log.
- **`DS_MeZO_Combined.md`** — Detailed design rationale. Contains known inaccuracies documented in `objectives.md` §6. Section references (§2, §3, etc.) appear throughout the code as comments linking implementation to spec.
