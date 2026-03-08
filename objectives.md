# DS-MeZO: Objectives and Authoritative Specification

This file is the single source of truth for determining what belongs in this repository. Any code component that cannot be mapped to a requirement here is slated for deletion.

---

## 1. Core Research Problem

**Can zeroth-order (gradient-free) optimization fine-tune large language models on non-differentiable reward signals using a single GPU, at near-inference memory cost?**

Backpropagation-based RL post-training (PPO, GRPO, DeepSeek-R1) requires 2-4x model memory for gradients, optimizer states, and activations — demanding multi-GPU clusters. DS-MeZO replaces backpropagation with SPSA gradient estimation on PiSSA (LoRA-compatible) adapters, using forward passes only. The optimizer estimates gradients via symmetric two-point finite differences, operating entirely in the adapter's low-rank subspace.

**Claimed contributions:**

1. A complete ZO-RL training pipeline (AGZO + ZO-Muon + RLOO) that runs on a single H100 with ~0.8% memory overhead beyond inference
2. Activation-guided subspace perturbation (AGZO) reducing effective ZO dimensionality by ~8x
3. Fused Triton kernels for ZO-Muon spectral update (Newton-Schulz orthogonalization + momentum-aligned masking)
4. Empirical validation: Llama-3.1-8B MBPP pass@1 improvement (+0.4%) via execution-based RL reward, no backpropagation

**Stated baselines (for memory comparison, not accuracy):**

| Method | Memory | Hardware |
|:-------|:-------|:---------|
| PPO | 3-4x model size (~48-64 GB for 8B) | Multi-GPU + DeepSpeed ZeRO-3 |
| GRPO | 2-3x model size (~32-48 GB for 8B) | Multi-GPU + FSDP |
| DS-MeZO | Model + ~630 MB overhead (~17 GB for 8B) | Single GPU |

---

## 2. Canonical Mechanism-to-Code Mapping

### 2.1 Core Algorithm (Novel — must be preserved)

| Mechanism | Spec Section | File:Function | Role |
|:----------|:-------------|:--------------|:-----|
| **SPSA gradient estimation** | §2 | `controller.py:step()` L399-400 | `dd = (loss_pos - loss_neg) / (2*eps)` — scalar directional derivative shared across all layers |
| **AGZO subspace perturbation** | §3.2 | `controller.py:_get_perturbation()` | Projects perturbations into activation subspace (B) and B's column space (A). Reduces effective dims ~8x |
| **Activation subspace tracking** | §3.5 | `controller.py:_calibrate_activation_bases_full()`, `_update_activation_bases_power_iter()` | Full SVD at init; per-step K=3 power iteration with drift detection (τ=0.95) triggering recalibration |
| **Momentum-aligned masking** | §3.3 | `kernels.py:_zo_muon_tall_kernel` L50-54, `_zo_muon_wide_kernel` L139-143 | `mask = sign(grad) == sign(momentum_buf)` — applied to A only, after warmup (10 steps) |
| **ZO-Muon spectral update** | §5.2 | `kernels.py:zo_muon_update()` | Fused: grad → masking → momentum → Frobenius norm → 5-iter Newton-Schulz → param update. Dispatches tall/wide kernel |
| **Newton-Schulz orthogonalization** | §5.2 | `kernels.py:_zo_muon_tall_kernel` L74-97 (tall), `_zo_muon_wide_kernel` L161-184 (wide) | Tall: `X = 0.5 * X @ (3I - X^T@X)`. Wide: `X = 0.5 * (3I - X@X^T) @ X`. Inner products are rank×rank (16×16) |
| **Fused dual perturbation** | §6.1 | `kernels.py:fused_perturb_dual()` | θ+ = base + z, θ- = base - z in one kernel pass (halves memory traffic) |
| **RLOO advantage computation** | §4.2 | `controller.py:_explore()` L288-293 | `adv_i = r_i - mean(r_j, j≠i)`. Unbiased, minimum-variance, no tunable baseline |
| **Contrastive scoring** | §4.3 | `controller.py:_score_contrastive()` | Advantage-weighted NLL under θ+ and θ-. Zero-advantage fallback to NLL on best candidate |
| **Asymmetric gradient regularization** | §4.3a | `controller.py:_score_contrastive()` L228-235 | `λ_GR * Σ(NLL_pos - NLL_neg)²` added to loss_pos only — does not cancel in finite differences |
| **KL constraint** | §4.3b | `controller.py:_apply_kl_constraint()` | Scales η down when `\|loss_pos - loss_neg\| > δ_KL` — post-hoc filter, not loss penalty |
| **Adaptive epsilon** | §2.1 | `controller.py:_update_eps()` | `eps = eps_base * max(loss_ema / initial_loss_ema, eps_floor)`. eps_base = 1e-3/√r |
| **DD clipping** | §2.1 | `controller.py:_clip_dd()` | Clips raw directional derivative at 3× running EMA |
| **Spike detection** | §4.4 | `controller.py:_check_health()` | Skip update if NLL > 5× EMA. Tracks on loss_neg (unbiased, no GR penalty) |
| **Cosine LR** | §5.3 | `controller.py:_update_lr()` | η_min = η_max/100, cosine annealing over total_steps |
| **Cosine temperature annealing** | §7.3 | `controller.py:_update_temperature()` | T_max=1.0 → T_min=0.3, with entropy collapse boost (1.5× if reward_range < 0.5 × initial) |

### 2.2 Infrastructure (Required — supports core algorithm)

| Component | File:Function | Role |
|:----------|:--------------|:-----|
| **PiSSA decomposition** | `scripts/prepare_pissa.py:decompose()` | Offline SVD: W0 → A@B + W_res. One-time preprocessing |
| **Layer discovery** | `model_config.py:discover_layers()` | Meta-device introspection for target module enumeration |
| **vLLM backend** | `backend.py:VLLMBackend` | Adapter serialization to /dev/shm, generation, prefill scoring, activation hook capture |
| **Adapter serialization** | `backend.py:_save_peft_adapter()` | FP32→BF16 with PiSSA↔PEFT naming swap (lora_A=B, lora_B=A) |
| **Activation extraction** | `backend.py:extract_activations()` | Forward hooks via collective_rpc; handles vLLM module merges (qkv_proj, gate_up_proj) |
| **Checkpointing** | `controller.py:_save_checkpoint()` | FP32 masters + momentum + optimizer state every 100 steps |
| **Training entry** | `scripts/train.py:main()` | YAML config + JSONL prompts → vLLM engine → controller.train() |
| **Launch script** | `scripts/launch.sh` | GPU clock lock, env vars, PYTHONPATH |

### 2.3 SFT Mode (Secondary — vanilla MeZO baseline)

| Component | File:Function | Role |
|:----------|:--------------|:-----|
| **SFT step** | `controller.py:_step_sft()` | Random full-rank perturbation + plain SGD (no AGZO, no ZO-Muon, no masking) |
| **SFT loss** | `controller.py:_compute_loss_sft()` | NLL on completion tokens under θ+ and θ-, with asymmetric GR |

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
        → extract_activations(): hook-based capture → power iteration update
        → _get_perturbation(): AGZO subspace perturbation for each layer
        → fused_perturb_dual(): θ+ = base + εz, θ- = base - εz
        → _sync_adapters(): serialize to /dev/shm
        → _score_contrastive(): prefill scoring under θ+ and θ-, advantage-weighted NLL + GR
        → _check_health(): spike detection
        → _clip_dd(): DD clipping
        → zo_muon_update(): fused Triton kernel (grad + mask + momentum + N-S + param update)
        → _update_lr(), _update_eps(), _update_temperature()
  → eval_mbpp() — post-training evaluation
```

**Prefills per step:** 1 generation + 1 activation extraction + 4 scoring = 6 total

### 3.2 SFT (Secondary)

**Entry:** `eval/sft_eval.py`

**Data:** GSM8K train split via `datasets.load_dataset("openai/gsm8k", "main", split="train")`

**Execution path:** Same as RL except: random perturbation (not AGZO), plain SGD (not ZO-Muon), NLL loss on completions (not contrastive scoring).

### 3.3 Component Ablations

**Entry:** `eval/ablations.py`

**Experiments (8):** control, no_zomuon, no_agzo, no_masking, no_gr, no_kl, static_bases, fixed_eps. Each runs 200 steps with monkey-patched controller. Measures: score, loss_ema, dd_ema, memory, time.

---

## 4. Evaluation Metrics

All metrics use standard libraries. No custom metric implementations.

| Metric | Library | Function | Used In |
|:-------|:--------|:---------|:--------|
| MBPP pass@1 | `evaluate.load("code_eval")` | `code_eval.compute(references, predictions)` | `eval/benchmarks.py:eval_mbpp()` |
| HumanEval pass@1 | `evaluate.load("code_eval")` | `code_eval.compute(references, predictions)` | `eval/benchmarks.py:eval_humaneval()` |
| GSM8K exact match | `evaluate.load("exact_match")` | `exact_match.compute(predictions, references)` | `eval/benchmarks.py:eval_gsm8k()` |
| MMLU accuracy | `evaluate.load("accuracy")` | `accuracy.compute(predictions, references)` | `eval/benchmarks.py:eval_mmlu()` |
| Perplexity | `math.exp(avg_nll)` | Direct NLL via vLLM prefill logprobs | `eval/benchmarks.py:eval_perplexity()` |

**Mathematical definitions:**
- **pass@1:** Fraction of problems where at least one generated solution passes all test assertions. Computed by `evaluate.code_eval` with sandboxed execution.
- **Exact match:** Binary string equality between extracted numerical answer and ground truth, after stripping commas.
- **Perplexity:** `exp(-(1/T) Σ_t log p(y_t | y_{<t}))` where T = number of completion tokens. NLL extracted via vLLM `prompt_logprobs`.
- **MMLU accuracy:** Fraction of questions where `argmax_c log p(c | prompt)` over c ∈ {A,B,C,D} matches ground truth.

---

## 5. Constraint Checklist

### 5.1 Deterministic Execution

- [x] Single execution path per mode (RL or SFT). No fallback logic, no "if missing then skip."
- [x] GPU always available. No CPU fallbacks, no `torch.cuda.is_available()` checks.
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

## 6. Spec-vs-Implementation Discrepancy Log

Discrepancies identified during forensic analysis (source: `eval/analysis_results.md` claim verification, `DS_MeZO_Combined.md` cross-reference, and code audit).

### 6.1 Discrepancies That Undermine Stated Claims

| # | Spec Claim | Implementation Reality | Impact |
|:--|:-----------|:----------------------|:-------|
| D1 | §9: Controller state ~0.22 GB | Measured: ~0.63 GB (2.6× over budget) | Memory budget table inaccurate. "Near-inference" claim still holds (0.8% overhead) but specific numbers mislead. |
| D2 | §6.1: 5 fused Triton kernels (subspace_perturb, zo_muon_update, spsa_gradient, score_reduce, health_monitor) | Only 2 implemented: `zo_muon_update`, `fused_perturb_dual`. Other ops are Python. | Spec overstates kernel coverage. Functionally equivalent but not "entirely in Triton" as §1 claims. |
| D3 | §2 SPSA formula: `ĝ = ((L⁺-L⁻)/2ε) · z` | Implementation uses eps-scaled z: `dd * z_scaled` where `z_scaled = eps * z_raw`. Epsilon cancels in effective gradient: `((L⁺-L⁻)/2) · z_raw`. | Adaptive epsilon controls perturbation magnitude only, not gradient magnitude. Formula in §2 is misleading about what epsilon actually scales. Internally consistent with §8 pseudocode. |
| D4 | §3.2 A perturbation formula includes sparse mask M_l: `z_A = Z_coeff' · Q^T ⊙ M_l` | Masking is not applied during perturbation — it's applied at update step inside the Triton kernel. | Spec formula is wrong about where masking occurs. Implementation is correct (masking at perturbation time is meaningless since z_A is random). Noted as "Bug 2 fix" in controller docstring. |
| D5 | §3.4 claims ~50% sparsity from momentum-aligned masking (vs 20% for magnitude masking) | Not empirically verified in codebase. | Analytical projection, not measured. |

### 6.2 Discrepancies That Are Neutral or Strengthen

| # | Spec Claim | Implementation Reality | Impact |
|:--|:-----------|:----------------------|:-------|
| D6 | §6.2.4: `enforce_eager=False` for CUDA graphs | Implementation: `enforce_eager=True` | Necessary — Python forward hooks for activation extraction don't fire under CUDA graphs. Correct engineering decision. |
| D7 | §8 pseudocode: `_score_contrastive` uses `max(len(logprobs), 1)` for NLL denominator | Implementation uses `len(logprobs)` directly | Simplification — logprobs list is never empty in practice (requires completion tokens to exist). |
| D8 | §8 pseudocode: separate `_zo_muon_update()` Python function with `torch.compile` | Implementation: fused Triton kernel `zo_muon_update()` | Upgrade — Triton kernel is faster and avoids `torch.compile` graph capture overhead. |
| D9 | §8 pseudocode: `_check_health` uses `avg_nll = (abs(loss_pos) + abs(loss_neg)) / 2` | Implementation uses `avg_nll = abs(loss_neg)` only | Improvement — loss_neg is unbiased (no GR penalty), better EMA signal. |
| D10 | §7.2: "7 total prefills per step" | Actual: 1 generation + 1 activation extraction + 4 scoring = 6 prefills + 1 generation | Terminology imprecise. Generation ≠ prefill. |

### 6.3 Spec Features Not Implemented (documented, not bugs)

| # | Feature | Spec Section | Status |
|:--|:--------|:-------------|:-------|
| N1 | Pinned memory for adapter transfer | §6.2.5 | Not implemented. Low priority — tmpfs already eliminates I/O bottleneck. |
| N2 | `torch.compile` for controller kernels | §6.2.7 | Superseded by Triton kernels. |
| N3 | NF4 quantization of W_res | §3.1 | Not implemented — residual model stored in original dtype (BF16). Would reduce VRAM by ~50% for the base model. |
| N4 | K-sample SPSA averaging | §13 | Future extension. |
| N5 | CUDA graph capture for scoring | §6.2.4 | Incompatible with activation hooks. |

### 6.4 Paper-vs-Implementation Comparison

Cross-reference of DS-MeZO's implementation against the four cited source papers. Each mechanism is an adaptation, not a direct reimplementation.

| Paper | Cited As | DS-MeZO Adaptation | Deviation | Impact |
|:------|:---------|:-------------------|:----------|:-------|
| **AGZO** (2601.17261) | §3.2 subspace perturbation | Paper uses one-point estimator `(f⁺ - f₀)/μ` with r=1 subspace. DS-MeZO uses symmetric two-point SPSA `(f⁺ - f⁻)/2ε` with r_calib=8. Subspace extraction (power iteration on activations → QR) matches. | **Strengthens** — two-point SPSA has lower variance than one-point. Higher r_calib captures more activation subspace structure. |
| **ZO-Muon** (2602.17155) | §5 spectral optimizer | Paper uses a separate random projection matrix P (resampled every 100 steps), N_q=4 perturbation queries per step, and a 3-term N-S recurrence with coefficients (3.4445, -2.8025, 0.8558). DS-MeZO applies standard 2-term N-S `X = 0.5·X(3I - XᵀX)` directly to AGZO-restricted momentum buffers — no separate projection matrix P, no multi-query perturbation. | **Significant divergence** — DS-MeZO uses N-S orthogonalization from ZO-Muon but substitutes AGZO's subspace for ZO-Muon's random projection. The N-S recurrence is the simpler standard form, not the paper's optimized 3-term variant. Empirically validated via ablation (no_zomuon: -48.8). |
| **Magma** (2602.15322) | §3.3 momentum-aligned masking | Paper operates at block granularity with sigmoid-scaled alignment `s = sigmoid(cos_sim(μ, g)/τ)`, EMA-damped scaling (β=0.9), and Bernoulli skip masks. DS-MeZO simplifies to element-wise binary masking `sign(grad) == sign(momentum_buf)` on A matrices only, no sigmoid, no temperature, no scaling EMA. | **Simplified adaptation** — captures the core insight (mask by gradient-momentum agreement) but discards block-level granularity and continuous scaling. Empirically validated via ablation (no_masking: -126.2). |
| **Gradient Reg.** (2602.18037) | §4.3a reward gradient penalty | Paper is a first-order method: penalizes `‖∇_ϕ J‖²` via finite-difference Hessian-vector products requiring 2 backward passes. DS-MeZO creates a ZO-native analog: `λ_GR · Σ(NLL_pos - NLL_neg)²` — the NLL divergence between perturbation directions, added asymmetrically to loss_pos only. No backward passes, no Hessian approximation. | **Different mechanism** — shares the motivation (penalize rapid reward changes to prevent hacking) but the formula is a ZO-specific construction, not the paper's first-order method. Asymmetric application ensures non-cancellation in SPSA finite differences. Empirically validated (no_gr: -205.8, second-largest ablation delta). |

**Summary:** DS-MeZO does not directly reimplement any of these papers. It adapts their core insights into a unified ZO framework: AGZO's activation subspace perturbation, ZO-Muon's spectral orthogonalization (simplified N-S form), Magma's momentum-alignment concept (simplified to binary masking), and gradient regularization's sharpness penalty (reformulated for ZO). All adaptations are empirically validated through ablation experiments showing positive contribution.

---

## 7. What Must Stay vs What Is Noise

### Must Stay (maps to §1-§5 mechanisms or §3.1-§3.3 execution paths)

```
ds_mezo/
  controller.py     — Core algorithm: SPSA, AGZO, RLOO, ZO-Muon, all schedules
  backend.py         — vLLM isolation: adapter serialization, scoring, activation hooks
  kernels.py         — Fused Triton kernels: zo_muon_update, fused_perturb_dual
  model_config.py    — Layer discovery via meta-device introspection
  __init__.py

scripts/
  prepare_pissa.py   — Offline PiSSA decomposition (prerequisite)
  train.py           — Training entry point (YAML config + JSONL prompts)
  launch.sh          — Environment setup + GPU clock lock

eval/
  __init__.py        — Package marker
  utils.py           — Shared utilities (extract_code)
  benchmarks.py      — Standard benchmark implementations (all use `evaluate` library)
  rl_bench_eval.py   — Primary RL proof-of-concept (MBPP, execution-based reward)
  sft_eval.py        — Secondary SFT evaluation (GSM8K)
  ablations.py       — Component ablation experiments (8 controlled experiments)

tests/
  test_evaluation.py — Mathematical property verification of kernel/algorithm correctness

configs/
  smoke.yaml         — Minimal smoke test config

pyproject.toml       — Dependency manifest
```

### Deleted (cleanup complete)

The following files were removed — none mapped to any objective:

- `eval/run.py` — Superseded by `rl_bench_eval.py`
- `eval/debug_hooks.py` — Development artifact
- `eval/results.json`, `eval/rl_bench_results.json`, `eval/sft_results.json` — Output artifacts (now gitignored)
- `eval/analysis_results.md` — Audit report, content distilled into §6
- `DS_MeZO_Parameter_Audit.md` — Superseded pre-v2 parameter audit

### Retained Documentation

`DS_MeZO_Combined.md` remains as detailed design rationale. Its implementation-specific claims (kernel count, memory budget, CUDA graph usage, pseudocode) are known to diverge from the actual implementation (see §6). `objectives.md` (this file) is the authoritative reference for what the code actually does.
