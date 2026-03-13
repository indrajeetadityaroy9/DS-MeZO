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

# Training (requires GPU, vLLM — train.py sets environment automatically)
python -m scripts.train --config <config.yaml> --prompts <prompts.jsonl>
# Optional: --lock-clocks to lock GPU to max clocks (requires sudo)

# Evaluation scripts (each requires GPU + PiSSA-prepared model)
python -m eval.rl_bench_eval --model-path <path> --adapter-path <path> --output-dir <path>
python -m eval.ablations --model-path <path> --adapter-path <path> --output-dir <path>

# GRPO baseline (all dependencies in core install)
python -m eval.grpo_baseline --model-path <path> --adapter-path <path> --output-dir <path> --dsmezo-results <path>
```

Entry points are also available via `pyproject.toml`: `ds-mezo-train` (→ `scripts.train:main`) and `ds-mezo-pissa` (→ `scripts.prepare_pissa:main`).

A minimal smoke-test config exists at `configs/smoke.yaml` (10 steps, requires PiSSA-prepared model at `/dev/shm/pissa_prep/`, outputs to `./output`). Sample prompts at `prompts/test.jsonl`.

**Prompts format**: JSONL with `{"prompt": "..."}` per line.

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

- **`controller.py`** — RL training loop implementing BSCO (Bayesian Subspace Contrastive Optimization). Single `step()` executes: shrinkage RLOO → dynamic sampling → AGZO perturbation → contrastive scoring → diagonal Kalman update → ZO-Muon spectral optimization. Config via `{**_CONFIG_DEFAULTS, **config}` dict merge. Layer state via `LayerState` dataclass. Algorithm-only — no vLLM imports. Uses `CosineAnnealingLR` and `optht`. Variance-weighted AGZO perturbation (DAP-optimal via Kalman posterior variance).
- **`backend.py`** — vLLM isolation layer. Handles adapter serialization (PEFT format), generation, prefill-only logprob scoring, and activation extraction via forward hooks. All vLLM-specific code lives here.
- **`kernels.py`** — Four H100-validated Triton kernels (161 launches/step, ~620μs total, <1% of step time): `zo_muon_update` (Frobenius normalize + 12-iteration minimax N-S orthogonalization + param update; Kalman gradient estimation runs in Python before kernel call; dispatches tall vs wide), `fused_power_iter` (fused H.T@H@V + modified Gram-Schmidt QR for activation subspace tracking), `fused_agzo_perturbation` (fused z_B projection + BV accumulation + in-register QR + z_A projection), `fused_perturb_dual` (θ+=base+z, θ-=base-z in one multi-CTA pass). All FP32 + allow_tf32=False (R=16 Gram products below HMMA efficiency threshold; N-S convergence requires full precision). Single-CTA grid=(1,) for kernels 1-3 (forced by serial data dependencies). Raw pointer arithmetic only (Triton 3.6.0 block-pointer stability constraint). Adapter rank >= 16 (Hopper tl.dot contraction minimum).
- **`model_config.py`** — Model-agnostic layer discovery via `torch.device("meta")` introspection (zero memory — no weights loaded). Returns `LayerSpec` dataclass with PEFT naming conventions.
- **`__init__.py`** — Public API exports: `DSMeZO_Controller`, `VLLMBackend`, `LayerSpec`, `discover_layers`.

### Eval modules (`eval/`)

- **`benchmarks.py`** — Standard benchmark implementations: `eval_mbpp()` (zero-shot), `eval_humaneval()`, `eval_sst2()`, `eval_rte()`, `eval_livecodebench()`.
- **`utils.py`** — Shared utilities: `extract_code()` (parses code blocks from LLM output), `make_exec_reward()` (closure factory returning `(reward_fn, set_problem_fn)`).
- **`rl_bench_eval.py`** — RL proof-of-concept on MBPP: trains then evaluates.
- **`ablations.py`** — 8 controlled component ablation experiments via `run_experiment()`. Includes BSCO-specific ablations: `no_kalman` (EMA momentum + N-S, pre-BSCO behavior), `no_shrinkage` (vanilla RLOO).
- **`grpo_baseline.py`** — GRPO backprop baseline via TRL GRPOTrainer. Same PiSSA init as DS-MeZO for controlled comparison. Requires `--dsmezo-results` path to DS-MeZO results JSON.

### Key conventions

- **PiSSA ↔ PEFT naming**: PiSSA uses `A` (d_out × r) and `B` (r × d_in). PEFT stores `lora_A.weight = B` and `lora_B.weight = A`. This swap appears in `controller.py` init, `backend.py` serialization, and `prepare_pissa.py`.
- **vLLM module merges**: vLLM fuses `q_proj`/`k_proj`/`v_proj` → `qkv_proj` and `gate_proj`/`up_proj` → `gate_up_proj`. The backend's `hook_map` handles this translation for activation capture.
- **Adapter staging**: Adapters serialize to a configurable `staging_dir` (default `/dev/shm/ds_mezo/`, tmpfs) for zero-copy vLLM loading. Two adapter slots: `adapter_pos` (θ+ε) and `adapter_neg` (θ-ε) for SPSA. Staging is ephemeral; persistent artifacts go to `output_dir`.
- **Directory convention** (HuggingFace standard): `output_dir` is the single root for all persistent artifacts (checkpoints under `{output_dir}/checkpoints/`, eval results as `{output_dir}/*.json`). `staging_dir` is ephemeral (tmpfs) for vLLM adapter hot-swap. All paths are required CLI arguments — no hardcoded defaults.
- **Config as plain dict**: `DSMeZO_Controller` init takes a `dict[str, Any]`. Caller config merged over `_CONFIG_DEFAULTS` via `{**defaults, **config}`.

### Config fields (config YAML surface)

Required in practice: `model_path`, `adapter_path`, `output_dir` (has default `"output"` but should always be set explicitly). Optional fields with defaults:

| Field | Default | Purpose |
|:---|:---|:---|
| `total_steps` | 1000 | Training budget |
| `staging_dir` | `/dev/shm/ds_mezo` | Tmpfs path for adapter hot-swap |
| `seed` | 42 | RNG seed for reproducibility |
| `score_fn` | None | Custom reward callable (code sets this programmatically) |
| `resume_from` | None | Path to checkpoint dir for resuming (e.g., `output/checkpoints/step_500`) |

Self-adaptive parameters (all derived from model/dtype at init — no manual tuning):
- **Epsilon**: `median(‖W‖_F) * eps_machine^(1/3)` (Numerical Recipes §5.7). Scale-invariant.
- **LR**: `eta_max = eps`, cosine decay to 0 via `torch.optim.lr_scheduler.CosineAnnealingLR`. N-S normalization makes update unit-spectral-norm; trust-region: step ≤ perturbation radius.
- **β (Kalman prediction decay)**: `1 - 1/min(step, √total_steps)` — same schedule as legacy momentum. Drives Kalman prediction step and reward EMA. Process noise `q = ε²·(1-β²)`, variance initialized to `ε²`.
- **r_calib**: Gavish-Donoho optimal singular value threshold via `optht` library (Marchenko-Pastur, determined once at calibration).
- **N-S iterations**: 12 iterations (from ℓ=√ε_f32 convergence criterion, Polar Express minimax composition). 27 global-memory passes per `zo_muon_update` call. <0.2% of vLLM-dominated step time. Convergence basin and iteration count fully derived from FP32 dtype — no manual constants.
- **Power iter steps**: `ceil(log(log(1/eps_dtype)/log(2))/log(3))` — warm-started convergence from FP32 precision.
- **SVD power iters**: Same warm-start formula — derived from FP32 precision (replaces Halko niter=2 default).
- **Norm floor**: `torch.finfo(torch.float32).tiny` — dtype-derived.
- **num_candidates**: 4 (RLOO practical default — Ahmadian et al. 2024, TRL default).

### Training flow

`scripts/train.py` reads `adapter_config.json` to extract rank and target modules, initializes vLLM with `max_lora_rank=max(64, rank)`, then calls `controller._calibrate_activation_bases_full()` (mandatory initialization) before `controller.train(prompts)`.

### Algorithm pipeline (one step — BSCO)

1. **Explore**: Generate N candidates under unperturbed weights, compute shrinkage RLOO advantages (James-Stein optimal baseline with reward EMA)
2. **Dynamic sampling**: If max(|advantage|) < ε, skip step (SPSA truncation-dominated). Step LR scheduler regardless.
3. **Activation tracking**: Warm-started power iteration update of per-layer activation bases
4. **AGZO perturbation**: Project random vectors into activation subspace for B (weighted by Kalman posterior variance — DAP-optimal); into B's column space for A
5. **Dual perturbation**: Fused kernel computes θ+ = base + εz and θ- = base - εz
6. **Contrastive scoring**: Score all N candidates under θ+ and θ- (advantage-weighted NLL)
7. **Kalman update**: Diagonal Kalman filter prediction + observation update on gradient posterior (Python). Posterior mean μ replaces momentum buffer.
8. **ZO-Muon update**: Fused kernel applies Frobenius normalize → minimax-optimal degree-3 Newton-Schulz orthogonalization (Polar Express) → parameter update
9. **Schedule**: Cosine LR (eta_max = eps → 0)

## Environment

- Requires NVIDIA GPU (H100 target, any CUDA GPU for tests)
- Python 3.12+, PyTorch 2.10, Triton 3.6, vLLM 0.17.1
- Pinned deps: `safetensors==0.7.0`, `transformers==4.56.0`, `datasets==3.5.0`, `evaluate==0.4.3`, `PyYAML==6.0.1`, `optht>=0.2.0`
- Environment variables (`VLLM_ALLOW_INSECURE_SERIALIZATION`, `OMP_NUM_THREADS`, `TOKENIZERS_PARALLELISM`, `PYTORCH_CUDA_ALLOC_CONF`) set automatically by `scripts/train.py` before imports
- vLLM `gpu_memory_utilization=0.95` hardcoded across all instantiation sites (no dynamic `mem_get_info()` calculation)
- Activation CPU→GPU transfers use pinned memory with non-blocking DMA
- Config files use YAML; `model_path`, `adapter_path`, and `output_dir` are required, all else defaults from `_CONFIG_DEFAULTS`

## Common pitfalls

- **PiSSA A/B swap**: PiSSA `A` maps to PEFT `lora_B.weight` and vice versa. Getting this wrong silently corrupts training. Always check which convention the current code path uses.
- **Triton rank >= 16**: The fused kernels require adapter rank >= 16 due to Hopper HMMA tile constraints. Smaller ranks will fail at kernel launch.
- **vLLM serialization**: `VLLM_ALLOW_INSECURE_SERIALIZATION=1` must be set or adapter loading fails silently. `train.py` sets this automatically.
- **Activation calibration**: `controller._calibrate_activation_bases_full()` must be called before `controller.train()`. Skipping it will cause runtime errors in AGZO perturbation.
- **Checkpoint resume**: Set `resume_from` in config YAML to a checkpoint directory path (e.g., `output/checkpoints/step_500`). Calibration is skipped on resume — activation bases are restored from checkpoint.
- **README vs CLAUDE.md commands**: The README uses direct script paths (`python scripts/prepare_pissa.py`); always use `python -m` module execution instead (`python -m scripts.prepare_pissa`).

## Design docs

- **`resources/`** — Reference papers: `sparse_mezo/`, `ds_mezo/`, `pissa/`, `dora/`.
