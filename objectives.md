# DS-MeZO: Objectives and Authoritative Specification

This file is the single source of truth for determining what belongs in this repository. Any code component that cannot be mapped to a requirement here is slated for deletion.

---

## 1. Core Research Problem

**Can zeroth-order (gradient-free) optimization fine-tune large language models on non-differentiable reward signals using a single GPU, at near-inference memory cost?**

Backpropagation-based RL post-training (PPO, GRPO, DeepSeek-R1) requires 2-4x model memory for gradients, optimizer states, and activations — demanding multi-GPU clusters. DS-MeZO replaces backpropagation with SPSA gradient estimation on PiSSA (LoRA-compatible) adapters, using forward passes only. The optimizer estimates gradients via symmetric two-point finite differences, operating entirely in the adapter's low-rank subspace.

**Claimed contributions:**

1. A complete ZO-RL training pipeline (AGZO + ZO-Muon + RLOO) that runs on a single H100 with ~0.8% memory overhead beyond inference
2. Activation-guided subspace perturbation (AGZO) reducing effective ZO dimensionality by ~8x
3. Four fused Triton kernels: ZO-Muon spectral update, power iteration, AGZO perturbation, dual perturbation
4. Empirical validation: Llama-3.1-8B MBPP pass@1 improvement via execution-based RL reward, no backpropagation

**Stated baselines (for memory comparison, not accuracy):**

| Method | Memory | Hardware |
|:-------|:-------|:---------|
| PPO | 3-4x model size (~48-64 GB for 8B) | Multi-GPU + DeepSpeed ZeRO-3 |
| GRPO | 2-3x model size (~32-48 GB for 8B) | Multi-GPU + FSDP |
| DS-MeZO | Model + ~630 MB overhead (~17 GB for 8B) | Single GPU |

---

## 2. Canonical Mechanism-to-Code Mapping

### 2.1 Core Algorithm (Novel — must be preserved)

| Mechanism | File:Function | Role |
|:----------|:--------------|:-----|
| **SPSA gradient estimation** | `controller.py:step()` | `dd = (loss_pos - loss_neg) / (2*eps)` — scalar directional derivative shared across all layers |
| **AGZO subspace perturbation** | `controller.py:_get_perturbation()` → `kernels.py:fused_agzo_perturbation()` | Projects perturbations into activation subspace (B) and B's column space (A). Reduces effective dims ~8x |
| **Activation subspace tracking** | `controller.py:_calibrate_activation_bases_full()`, `_update_activation_bases()` → `kernels.py:fused_power_iter()` | Full SVD at init; per-step K=3 warm-started power iteration with cubically convergent refinement |
| **ZO-Muon spectral update** | `kernels.py:zo_muon_update()` | Fused: grad → momentum → Frobenius normalize → 5-iter Newton-Schulz orthogonalization → param update. Dispatches tall/wide kernel |
| **Newton-Schulz orthogonalization** | `kernels.py:_zo_muon_tall_kernel`, `_zo_muon_wide_kernel` | Tall: `X = 0.5 * X @ (3I - X.T@X)`. Wide: `X = 0.5 * (3I - X@X.T) @ X`. Inner products are rank×rank (16×16) |
| **Fused dual perturbation** | `kernels.py:fused_perturb_dual()` | θ+ = base + z, θ- = base - z in one kernel pass (halves memory traffic) |
| **RLOO advantage computation** | `controller.py:_explore()` | `adv_i = r_i - mean(r_j, j≠i)`. Unbiased, minimum-variance, no tunable baseline. REINFORCE++ normalization by reward std. |
| **Contrastive scoring** | `controller.py:_score_contrastive()` | Advantage-weighted NLL under θ+ and θ-. Zero-advantage gives dd=0 (no update) |
| **ZClip** | `controller.py:_zclip()` | Z-score clipping on directional derivative: clips at dd_ema ± 3σ (Chebyshev bound). Adaptive EMA window. |
| **Cosine LR** | `controller.py:_update_lr()` | Cosine decay to zero (D2Z — CoLLAs 2025) |
| **Cosine temperature annealing** | `controller.py:_update_temperature()` | T_max → T_min = T_max/num_candidates, plain cosine schedule |
| **Fixed epsilon** | `controller.py.__init__` | `eps = 1e-3 / sqrt(rank)` (SPSA theory — Spall 1998; FlatZero — preserves flat-minima regularization) |

### 2.2 Infrastructure (Required — supports core algorithm)

| Component | File:Function | Role |
|:----------|:--------------|:-----|
| **PiSSA decomposition** | `scripts/prepare_pissa.py:decompose()` | Offline SVD: W0 → A@B + W_res. One-time preprocessing |
| **Layer discovery** | `model_config.py:discover_layers()` | Meta-device introspection for target module enumeration |
| **vLLM backend** | `backend.py:VLLMBackend` | Adapter serialization to /dev/shm, generation, prefill scoring, activation hook capture |
| **Adapter serialization** | `backend.py:_save_peft_adapter()` | FP32→BF16 with PiSSA↔PEFT naming swap (lora_A=B, lora_B=A) |
| **Activation extraction** | `backend.py:extract_activations()` | Forward hooks via collective_rpc; handles vLLM module merges (qkv_proj, gate_up_proj) |
| **Checkpointing** | `controller.py:_save_checkpoint()` | FP32 masters + momentum + activation bases + RNG state via safetensors + JSON |
| **Training entry** | `scripts/train.py:main()` | YAML config + JSONL prompts → vLLM engine → controller.train() |
| **Launch script** | `scripts/launch.sh` | GPU clock lock, env vars, PYTHONPATH |

### 2.3 SFT Mode (Secondary — vanilla MeZO baseline)

| Component | File:Function | Role |
|:----------|:--------------|:-----|
| **SFT step** | `controller.py:step()` via `hybrid_switch_step` | Same pipeline (AGZO + ZO-Muon) with NLL loss on completions instead of contrastive scoring |
| **SFT loss** | `controller.py:_compute_loss_sft()` | NLL on completion tokens under θ+ and θ- |

---

## 3. Primary Execution Paths

### 3.1 RL Post-Training (Primary — proves the central claim)

**Entry:** `eval/rl_bench_eval.py` or `scripts/train.py`

**Data:** MBPP sanitized train split (120 problems) via `datasets.load_dataset("google-research-datasets/mbpp", "sanitized", split="train")`

**Reward function:** `exec_reward()` — executes LLM-generated code against test assertions, returns fraction passing. Fully non-differentiable.

**Execution path:**
```
prepare_pissa.py (offline)
  → SVD decompose target weights → residual model + PEFT adapter

rl_bench_eval.py (online)
  → load vLLM engine (BF16, enable_lora, enforce_eager=True)
  → discover_layers() on meta device
  → VLLMBackend(engine, layer_specs, rank=16)
  → DSMeZO_Controller(backend, layer_specs, config)
  → _calibrate_activation_bases_full() — initial SVD on activation subspace
  → eval_mbpp() — pre-training baseline
  → for step in range(1000):
      controller.step([prompt])
        → _explore(): generate 4 candidates, score with exec_reward, RLOO advantages
        → extract_activations(): hook-based capture → power iteration update (fused_power_iter)
        → _get_perturbation(): AGZO subspace perturbation (fused_agzo_perturbation)
        → fused_perturb_dual(): θ+ = base + εz, θ- = base - εz
        → sync_adapters(): serialize to /dev/shm
        → _score_contrastive(): prefill scoring under θ+ and θ-, advantage-weighted NLL
        → _zclip(): z-score clipping on directional derivative
        → zo_muon_update(): fused Triton kernel (grad + momentum + N-S + param update)
        → _update_lr(), _update_temperature()
  → eval_mbpp() — post-training evaluation
```

**Prefills per step:** 1 generation + 1 activation extraction + 4 scoring = 6 total

### 3.2 SFT (Secondary)

**Entry:** `eval/sft_eval.py`

**Data:** GSM8K train split via `datasets.load_dataset("openai/gsm8k", "main", split="train")`

**Execution path:** Same pipeline (AGZO perturbation, ZO-Muon update) but with NLL loss on completions instead of contrastive RL scoring. No generation/exploration step.

### 3.3 Component Ablations

**Entry:** `eval/ablations.py`

**Experiments (4):** control, no_zomuon, no_agzo, static_bases. Each runs 200 steps with monkey-patched controller. Measures: pass@1, dd_ema, memory, time.

| Experiment | Ablation | Method |
|:-----------|:---------|:-------|
| `control` | None (full system) | Baseline |
| `no_zomuon` | Replace ZO-Muon with SGD+momentum | Monkey-patches `step()` to skip `zo_muon_update` kernel |
| `no_agzo` | Replace AGZO with random perturbation | Monkey-patches `_get_perturbation()` to skip `fused_agzo_perturbation` kernel |
| `static_bases` | Freeze activation bases at init | Monkey-patches `_update_activation_bases()` to no-op (skips `fused_power_iter` kernel) |

---

## 4. Evaluation Metrics

All metrics use standard libraries. No custom metric implementations.

| Metric | Library | Function | Used In |
|:-------|:--------|:---------|:--------|
| MBPP pass@1 | `evaluate.load("code_eval")` | `code_eval.compute(references, predictions)` | `eval/benchmarks.py:eval_mbpp()` |
| HumanEval pass@1 | `evaluate.load("code_eval")` | `code_eval.compute(references, predictions)` | `eval/benchmarks.py:eval_humaneval()` |
| GSM8K exact match | `evaluate.load("exact_match")` | `exact_match.compute(predictions, references)` | `eval/benchmarks.py:eval_gsm8k()` |
| Perplexity | `math.exp(avg_nll)` | Direct NLL via vLLM prefill logprobs | `eval/benchmarks.py:eval_perplexity()` |

**Mathematical definitions:**
- **pass@1:** Fraction of problems where at least one generated solution passes all test assertions. Computed by `evaluate.code_eval` with sandboxed execution.
- **Exact match:** Binary string equality between extracted numerical answer and ground truth, after stripping commas.
- **Perplexity:** `exp(-(1/T) Σ_t log p(y_t | y_{<t}))` where T = number of completion tokens. NLL extracted via vLLM `prompt_logprobs`.

---

## 5. Constraint Checklist

### 5.1 Deterministic Execution

- [x] Single execution path per mode (RL or SFT). No fallback logic.
- [x] GPU always available. No CPU fallbacks.
- [x] All dependencies pinned in `pyproject.toml`. No conditional imports.
- [x] First failure = immediate crash with traceback. No silent degradation.

### 5.2 Standard Libraries for Standard Tasks

- [x] Metrics: `evaluate` (HuggingFace). No custom pass@k, exact_match, or accuracy.
- [x] Data loading: `datasets` (HuggingFace). No custom data parsers for standard benchmarks.
- [x] Tokenization: `transformers.AutoTokenizer`. No custom tokenizers.
- [x] Model config: `transformers.AutoConfig` + `AutoModelForCausalLM`. No manual architecture parsing.
- [x] Tensor serialization: `safetensors`. No pickle.

### 5.3 Research-Code Assumptions

- [x] Target hardware: NVIDIA H100 80GB (SM 9.0). Any BF16-capable GPU (SM ≥ 8.0) for tests.
- [x] Adapter staging: `/dev/shm` (tmpfs). Always available on Linux.
- [x] PiSSA decomposition is offline preprocessing, not part of the training loop.
- [x] vLLM runs with `enforce_eager=True` (required for activation hooks — CUDA graphs disable Python hooks).

---

## 6. Spec-vs-Implementation Notes

### 6.1 DS_MeZO_Combined.md Divergences

`DS_MeZO_Combined.md` remains as detailed design rationale. The following areas diverge from the actual implementation:

| # | Area | DS_MeZO_Combined.md | Implementation |
|:--|:-----|:--------------------|:---------------|
| 1 | Memory budget | §9: Controller state ~0.22 GB | Measured: ~0.63 GB. "Near-inference" claim still holds (0.8% overhead). |
| 2 | Kernel count | §6.1: 5 fused Triton kernels | 4 implemented: zo_muon_update, fused_power_iter, fused_agzo_perturbation, fused_perturb_dual |
| 3 | CUDA graphs | §6.2.4: `enforce_eager=False` | `enforce_eager=True` — required for Python forward hooks |
| 4 | Prefill count | §7.2: "7 total prefills per step" | 1 generation + 1 activation extraction + 4 scoring = 6 prefills + 1 generation |

### 6.2 Paper Adaptations

DS-MeZO adapts insights from cited papers into a unified ZO framework:

| Paper | DS-MeZO Adaptation | Deviation |
|:------|:-------------------|:----------|
| **AGZO** (2601.17261) | Activation subspace perturbation | Uses symmetric two-point SPSA (lower variance than paper's one-point estimator). r_calib = rank/2 (paper uses r=1). |
| **ZO-Muon** (2602.17155) | Newton-Schulz orthogonalization on momentum buffers | Uses standard 2-term N-S `X = 0.5·X(3I - XᵀX)` (paper uses optimized 3-term variant). Substitutes AGZO subspace for ZO-Muon's random projection. |

### 6.3 Spec Features Not Implemented

| Feature | DS_MeZO_Combined.md | Status |
|:--------|:---------------------|:-------|
| Pinned memory for adapter transfer | §6.2.5 | Not implemented. Low priority — tmpfs eliminates I/O bottleneck. |
| NF4 quantization of W_res | §3.1 | Not implemented — residual model stored in BF16. |
| CUDA graph capture for scoring | §6.2.4 | Incompatible with activation hooks. |

---

## 7. Repository Structure

### Must Stay (maps to §2 mechanisms or §3 execution paths)

```
ds_mezo/
  controller.py     — Core algorithm: SPSA, AGZO, RLOO, ZO-Muon, ZClip, schedules
  backend.py         — vLLM isolation: adapter serialization, scoring, activation hooks
  kernels.py         — Four fused Triton kernels: zo_muon_update, fused_power_iter,
                       fused_agzo_perturbation, fused_perturb_dual
  model_config.py    — Layer discovery via meta-device introspection
  __init__.py

scripts/
  prepare_pissa.py   — Offline PiSSA decomposition (prerequisite)
  train.py           — Training entry point (YAML config + JSONL prompts)
  launch.sh          — Environment setup + GPU clock lock

eval/
  __init__.py        — Package marker
  utils.py           — Shared utilities (extract_code, pass_at_k, make_exec_reward)
  benchmarks.py      — Standard benchmark implementations (all use `evaluate` library)
  rl_bench_eval.py   — Primary RL proof-of-concept (MBPP, execution-based reward)
  sft_eval.py        — SFT evaluation (GSM8K)
  ablations.py       — Component ablation experiments (4 controlled experiments)

configs/
  smoke.yaml         — Minimal smoke test config

pyproject.toml       — Dependency manifest
```

### Retained Documentation

`DS_MeZO_Combined.md` remains as detailed design rationale. Its implementation-specific claims diverge from the actual implementation (see §6). `objectives.md` (this file) is the authoritative reference for what the code actually does.
