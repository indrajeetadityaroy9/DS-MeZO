# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

DS-MeZO: Zeroth-order RL post-training for LLMs on a single GPU. Uses SPSA gradient estimation on PiSSA (LoRA) adapters with forward passes only — no backpropagation. Targets non-differentiable rewards (code execution, proof verification). Built on vLLM for inference and Triton for fused optimizer kernels.

## Commands

All scripts use `python -m` execution from the project root. No `sys.path` manipulation.

```bash
# Install (editable)
pip install -e .

# PiSSA decomposition (prerequisite — creates residual model + adapter)
python -m scripts.prepare_pissa --model <model_path> --output <output_dir> [--targets q_proj v_proj]

# Training (requires GPU, vLLM — launch.sh sets environment variables and locks GPU clocks)
bash scripts/launch.sh <config.yaml> <prompts.jsonl>
# Internally runs: python -m scripts.train --config <config.yaml> --prompts <prompts.jsonl>

# Evaluation scripts (each requires GPU + PiSSA-prepared model)
python -m eval.rl_bench_eval --model-path <path> --adapter-path <path> --output-dir <path>
python -m eval.sft_eval --model-path <path> --adapter-path <path> --output-dir <path>
python -m eval.ablations --model-path <path> --adapter-path <path> --output-dir <path>
```

Entry points are also available via `pyproject.toml`: `ds-mezo-train` (→ `scripts.train:main`) and `ds-mezo-pissa` (→ `scripts.prepare_pissa:main`).

A minimal smoke-test config exists at `configs/smoke.yaml` (10 steps, requires PiSSA-prepared model at `/dev/shm/pissa_prep/`). Sample prompts at `prompts/test.jsonl`.

**Note**: There is no `tests/` directory yet — evaluation is done via the `eval/` scripts.

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

- **`controller.py`** — Unified training loop. Single `step()` handles both SFT and RL via `hybrid_switch_step`. Config via `_CONFIG_DEFAULTS` dict with `config.get()`. Layer state via `LayerState` dataclass. Algorithm-only — no vLLM imports.
- **`backend.py`** — vLLM isolation layer. Handles adapter serialization (PEFT format), generation, prefill-only logprob scoring, and activation extraction via forward hooks. All vLLM-specific code lives here.
- **`kernels.py`** — Four Hopper-native Triton kernels: `zo_muon_update` (fused gradient + momentum + 5-iteration Newton-Schulz + param update, dispatches tall vs wide), `fused_power_iter` (fused H.T@H@V + QR for activation subspace tracking), `fused_agzo_perturbation` (fused AGZO perturbation: z_B projection + B@V + in-register QR + z_A projection, eliminates cuSOLVER overhead), `fused_perturb_dual` (θ+=base+z, θ-=base-z in one pass). Requires adapter rank >= 16 (`tl.dot` 16-wide HMMA requirement on Hopper). Uses raw pointer arithmetic only — no block pointers or experimental APIs (Triton 3.6 stability constraint).
- **`model_config.py`** — Model-agnostic layer discovery via `torch.device("meta")` introspection (zero memory — no weights loaded). Returns `LayerSpec` dataclass with PEFT naming conventions.
- **`__init__.py`** — Public API exports: `DSMeZO_Controller`, `VLLMBackend`, `LayerSpec`, `discover_layers`.

### Eval modules (`eval/`)

- **`benchmarks.py`** — Standard benchmark implementations: `eval_perplexity()`, `eval_gsm8k()` (8-shot CoT), `eval_mbpp()` (zero-shot), `eval_humaneval()`.
- **`utils.py`** — Shared utilities: `extract_code()` (parses code blocks from LLM output), `pass_at_k()` (unbiased estimator), `make_exec_reward()` (closure factory returning `(reward_fn, set_problem_fn)`).
- **`rl_bench_eval.py`** — RL proof-of-concept on MBPP: trains then evaluates.
- **`sft_eval.py`** — SFT evaluation on GSM8K with perplexity metric.
- **`ablations.py`** — 4 controlled component ablation experiments via `run_experiment()`.

### Key conventions

- **PiSSA ↔ PEFT naming**: PiSSA uses `A` (d_out × r) and `B` (r × d_in). PEFT stores `lora_A.weight = B` and `lora_B.weight = A`. This swap appears in `controller.py` init, `backend.py` serialization, and `prepare_pissa.py`.
- **vLLM module merges**: vLLM fuses `q_proj`/`k_proj`/`v_proj` → `qkv_proj` and `gate_proj`/`up_proj` → `gate_up_proj`. The backend's `hook_map` handles this translation for activation capture.
- **Adapter staging**: Adapters serialize to a configurable `staging_dir` (default `/dev/shm/ds_mezo/`, tmpfs) for zero-copy vLLM loading. Two adapter slots: `adapter_pos` (θ+ε) and `adapter_neg` (θ-ε) for SPSA. Staging is ephemeral; persistent artifacts go to `output_dir`.
- **Directory convention** (HuggingFace standard): `output_dir` is the single root for all persistent artifacts (checkpoints under `{output_dir}/checkpoints/`, eval results as `{output_dir}/*.json`). `staging_dir` is ephemeral (tmpfs) for vLLM adapter hot-swap. All paths are required CLI arguments — no hardcoded defaults.
- **Config as plain dict**: `DSMeZO_Controller` init takes a `dict[str, Any]`. Defaults are in `_CONFIG_DEFAULTS` module-level dict; unknown keys are silently ignored via `config.get(key, default)`.

### Config fields (config YAML surface)

Required: `model_path`, `adapter_path`, `output_dir`. Key optional fields with defaults:

| Field | Default | Purpose |
|:---|:---|:---|
| `total_steps` | 1000 | Training budget |
| `hybrid_switch_step` | 0 | Step to switch from SFT to RL (0 = RL only) |
| `eta_max` | 1e-4 | Peak learning rate (cosine schedule) |
| `momentum` | 0.9 | Momentum coefficient |
| `num_candidates` | 4 | RLOO candidate count per step |
| `T_max` | 1.0 | Max sampling temperature |
| `staging_dir` | `/dev/shm/ds_mezo` | Tmpfs path for adapter hot-swap |
| `seed` | 42 | RNG seed for reproducibility |
| `score_fn` | None | Custom reward callable (code sets this programmatically) |

Epsilon is derived from adapter rank. DD clipping uses adaptive EMA variance tracking. See `_CONFIG_DEFAULTS` comment in `controller.py`.

### Training flow

`scripts/train.py` reads `adapter_config.json` to extract rank and target modules, initializes vLLM with `max_lora_rank=max(64, rank)`, then calls `controller._calibrate_activation_bases_full()` (mandatory initialization) before `controller.train(prompts)`.

### Algorithm pipeline (RL mode, one step)

1. **Explore**: Generate N candidates under unperturbed weights, compute RLOO advantages
2. **Activation tracking**: Warm-started power iteration update of per-layer activation bases
3. **AGZO perturbation**: Project random vectors into activation subspace for B; into B's column space for A
4. **Dual perturbation**: Fused kernel computes θ+ = base + εz and θ- = base - εz
5. **Contrastive scoring**: Score winner/loser trajectories under θ+ and θ-
6. **ZClip**: Z-score clipping on directional derivative (3σ from tracked variance)
7. **ZO-Muon update**: Fused kernel applies SPSA gradient → momentum accumulation → Newton-Schulz orthogonalization → parameter update
8. **Schedule**: Cosine LR, cosine temperature annealing

## Environment

- Requires NVIDIA GPU (H100 target, any CUDA GPU for tests)
- Python 3.12+, PyTorch 2.10, Triton 3.6, vLLM 0.17.0
- Pinned deps: `safetensors==0.7.0`, `transformers==4.56.0`, `datasets==3.5.0`, `evaluate==0.4.3`, `PyYAML==6.0.1`
- `VLLM_ALLOW_INSECURE_SERIALIZATION=1` required (set by caller — see `scripts/launch.sh`)
- `launch.sh` locks GPU clocks (`nvidia-smi -lgc 1980,1980 && nvidia-smi -lmc 2619`), sets `OMP_NUM_THREADS=16`, `MKL_NUM_THREADS=16`, `TOKENIZERS_PARALLELISM=true`, disables TF imports
- Config files use YAML; `model_path`, `adapter_path`, and `output_dir` are required, all else defaults from `_CONFIG_DEFAULTS`

## Design docs

- **`objectives.md`** — Authoritative specification and single source of truth. Canonical mechanism-to-code mapping, execution paths, discrepancy log.
- **`DS_MeZO_Combined.md`** — Detailed design rationale. Contains known inaccuracies documented in `objectives.md` §6. Section references (§2, §3, etc.) appear throughout the code as comments linking implementation to spec.
- **`research_synthesis.md`** — Research context and related work survey.
