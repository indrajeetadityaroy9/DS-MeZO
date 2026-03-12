# DS-MeZO Research Audit

Comprehensive code-level audit of the DS-MeZO codebase. Covers bugs, deviations from referenced methods, design gaps, and verified correct implementations.

---

## Critical Bugs

### BUG-1: RLOO Advantage Normalization Division by Zero — `controller.py:278`

**Location**: `controller.py`, lines 274–280

```python
rewards = torch.tensor([r for _, r in scored])
N = len(scored)
baselines = (rewards.sum() - rewards) / (N - 1)
advantages = (rewards - baselines) / rewards.std()  # BUG: std()=0 when all equal
adv_w = float(advantages[0])
adv_l = float(advantages[-1])
```

**Problem**: When all N=4 candidates receive identical rewards, `rewards.std()` returns 0. Division by zero produces NaN, which propagates through momentum buffers and parameters — **permanently corrupting the model** with no recovery path.

**When it triggers**: Common early in training when all candidates fail (reward=0) or all pass (reward=1). Guaranteed to occur with binary code-execution rewards on easy or hard problems.

**Note on the normalization choice**: The baseline subtraction `(rewards.sum() - rewards) / (N - 1)` is standard RLOO (Ahmadian et al. 2024). However, dividing by `rewards.std()` is a separate technique (GRPO-style per-group normalization, not part of standard RLOO). Standard RLOO does not divide by std — the leave-one-out baseline alone provides an unbiased advantage. TRL's RLOO implementation guards against this with `std + 1e-4`. The REINFORCE++ paper (Hu 2025, arXiv 2501.03262) specifically argues against per-group std normalization, recommending global batch std instead, since per-group std introduces bias in the policy gradient.

**Severity**: Critical. Silent, irreversible model corruption.

**Recommended fix**: Guard with an epsilon floor (matching TRL convention) or early return:
```python
std = rewards.std()
if std < 1e-8:
    adv_w, adv_l = 0.0, 0.0  # no signal — skip update
else:
    advantages = (rewards - baselines) / std
    adv_w = float(advantages[0])
    adv_l = float(advantages[-1])
```

---

### BUG-2: N-S Iteration Count Always Returns Maximum — `controller.py:147-157`

**Location**: `controller.py`, lines 147–157

```python
@staticmethod
def _ns_iters_for_smin(s_min: float, eps_dtype: float) -> int:
    """Simulate scalar N-S map from s_min until convergence."""
    a, b, c = DEFAULT_A, DEFAULT_B, DEFAULT_C
    s = s_min
    for k in range(20):
        s2 = s * s
        s = s * (a + b * s2 + c * s2 * s2)
        if abs(s - 1.0) < eps_dtype:
            return k + 1
    return 20
```

**Problem**: The function simulates the scalar Newton-Schulz polynomial `s → s(a + bs² + cs⁴)` and checks whether singular values converge to 1.0. This is **fundamentally misconceived** — the Muon polynomial is intentionally designed to NOT converge to 1.0.

**Evidence**: PyTorch's own `torch/optim/_muon.py` documentation (lines 36–44) states:

> *"We opt to use a quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose of minimizing steps, it turns out to be empirically effective to keep increasing the slope at zero even beyond the point where the iteration no longer converges all the way to one everywhere on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model performance at all relative to UV^T."*

**Verification**: The coefficients give `p(1) = a + b + c = 3.4445 + (-4.775) + 2.0315 = 0.701`. Since `p(1) ≠ 1`, singular value 1.0 is not a fixed point. The scalar orbit oscillates indefinitely:

```
Scalar N-S from s_min=0.25 (rank=16):
  k=0: s=0.25  → 0.789
  k=1: s=0.789 → 0.994
  k=2: s=0.994 → 0.705    ← bounces away from 1.0
  k=3: s=0.705 → 1.109    ← overshoots
  k=4: s=1.109 → 0.715    ← oscillates
  ... never converges, returns 20 (max cap)
```

**Consequences**:
1. `_ns_iters_for_smin()` **always returns 20** regardless of rank or precision
2. DS-MeZO runs **4x more N-S iterations** than canonical Muon (`DEFAULT_NS_STEPS = 5`)
3. Wasted compute in every fused kernel call (N-S is the inner loop of `zo_muon_update`)
4. The oscillating polynomial applied 20 times may produce worse orthogonalization than 5 times — the output depends on which phase of the oscillation terminates
5. CLAUDE.md claim "3 iters for rank=16" is incorrect — it is always 20

**Severity**: High. 4x unnecessary compute overhead and potentially degraded optimization quality.

**Recommended fix**: Replace the simulation with the canonical constant:
```python
from torch.optim._muon import DEFAULT_NS_STEPS
self.ns_iterations = DEFAULT_NS_STEPS  # 5
```
Or simply hardcode `self.ns_iterations = 5`.

---

### BUG-3: Gavish-Donoho `optht` Beta Inverted — `controller.py:182`

**Location**: `controller.py`, lines 181–183

```python
m, n = H.shape
beta = max(m, n) / min(m, n)
ranks_per_activation.append(optht(beta, sv=sv, sigma=None))
```

**Problem**: The `optht` library requires `beta = min(m, n) / max(m, n)`, constrained to (0, 1]. The code computes the **inverse**: `max(m, n) / min(m, n)`, which gives beta >= 1. The library validates this and raises `ValueError('Parameter beta must be in (0,1].')` for any non-square activation matrix.

**Verification**: The `optht` source ([erichson/optht on GitHub](https://github.com/erichson/optht)) contains:
```python
if beta < 0 or beta > 1:
    raise ValueError('Parameter `beta` must be in (0,1].')
```
When an array is passed instead of a scalar, the library auto-computes `beta = min(shape) / max(shape)`.

**When it triggers**: Always, unless the activation matrix happens to be square (T = D). For a typical LLM prompt (T~100 tokens, D~4096 hidden dim), `beta = 4096/100 = 40.96` → immediate `ValueError` crash during calibration.

**Severity**: Critical. Calibration crashes before training begins for any non-square activation matrix (i.e., virtually always).

**Recommended fix**:
```python
beta = min(m, n) / max(m, n)
```

---

## Deviations from Referenced Methods

### DEV-1: AGZO — Adapted for PiSSA Decomposition (Not in Original Paper)

**Reference**: AGZO (arXiv 2601.17261) is a full-parameter zeroth-order method. It projects perturbations into the activation subspace for all linear layers uniformly. The paper does not discuss LoRA/PiSSA adapters or differentiated treatment of A vs B matrices.

**Implementation**: DS-MeZO adapts AGZO for the PiSSA two-factor decomposition with differentiated perturbation strategies for each factor:
- **B matrices** (input-facing, r × d_in): Perturbation projected through activation basis V → `z_B = (z_coeff_B @ V.T) * eps` (`kernels.py` Phase 1, lines 413–443)
- **A matrices** (output-facing, d_out × r): Perturbation projected through B's column space in the activation subspace → `Q = QR(B @ V)`, then `z_A = (z_coeff_A @ Q.T) * eps` (`kernels.py` Phases 2–3, lines 454–498)

Both factors use subspace projection — B uses the activation subspace directly, A uses the column space of B projected onto the activation subspace. There is no sparse masking anywhere in the codebase.

**Theoretical grounding**: The AGZO gradient confinement theorem (row(∇_W f) ⊆ col(H)) applies directly to B since H is the layer's input activation. For A, the column-space projection through Q = QR(B @ V) captures the subspace that B actually maps activations into, which is a reasonable proxy.

**Impact**: This PiSSA-specific adaptation is a novel contribution of DS-MeZO, not found in the AGZO paper or any published work. The two-factor treatment is well-motivated but should be clearly documented as a novel extension.

---

### DEV-2: ZO-Muon — Novel Synthesis, Not Direct Implementation

**Reference**: ZO-Muon (arXiv 2602.17155) proposes subspace zeroth-order optimization with Newton-Schulz orthogonalization in its own algorithmic framework.

**Implementation**: DS-MeZO integrates N-S orthogonalization into its own SPSA+RLOO pipeline. The optimizer step structure is:

```
SPSA gradient estimate → adaptive momentum → N-S orthogonalization → parameter update
```

This sequence is original to DS-MeZO, not a direct reproduction of the ZO-Muon paper's algorithm. The N-S orthogonalization step is borrowed from Muon/ZO-Muon, but everything around it (SPSA formulation, RLOO advantages, AGZO perturbation directions) is DS-MeZO's own design.

**Impact**: The contribution is the synthesis itself — combining AGZO subspace perturbation with Muon-style spectral normalization in a ZO setting. This should be clearly framed as a novel combination rather than an implementation of ZO-Muon.

---

### DEV-3: RLOO Uses Only Best/Worst of N Candidates

**Reference**: Standard RLOO (Ahmadian et al. 2024) computes advantages for all N candidates: `adv_i = r_i - mean(r_j, j≠i)` and uses all N gradient signals.

**Implementation** (`controller.py:271-280`): Advantages are computed for all N=4 candidates, but only `adv_w` (best, index 0 after descending sort) and `adv_l` (worst, index -1) are used. The middle 2 candidates are discarded.

```python
best_output, best_reward = scored[0]
worst_output, worst_reward = scored[-1]
# ...
adv_w = float(advantages[0])   # best
adv_l = float(advantages[-1])  # worst
```

**Impact**: Reduces to a max-min contrastive signal rather than full RLOO. Each RL step produces exactly one gradient estimate from the winner-loser pair instead of N-1 independent estimates. This sacrifices sample efficiency — the information from middle-ranked candidates is unused. However, this design pairs naturally with SPSA's two-point evaluation (θ+ scores winner, θ- scores loser).

---

### DEV-4: Single-Prompt Activation Calibration

**Location**: `scripts/train.py` calls `_calibrate_activation_bases_full([prompts[0]])` with only the **first prompt** from the dataset.

**Problem**: The Gavish-Donoho threshold (`optht`) is computed on activations from this single prompt. Activation statistics can vary significantly across prompts — especially between short/long prompts or different domains. The resulting `r_calib` (number of significant singular values to retain) may be unrepresentative of the training distribution.

**Impact**: The activation subspace basis and its dimensionality are determined by one data point. If the first prompt is atypical, the AGZO subspace will be poorly initialized. Warm-started power iteration during training can adapt, but the initial `r_calib` from Gavish-Donoho is fixed for the entire run.

**Recommended fix**: Calibrate on a small batch (e.g., 8–16 prompts) to get representative activation statistics.

---

### DEV-5: Momentum Zero at Step 1

**Location**: `controller.py:323-324`

```python
momentum = 1 - 1/min(step, sqrt(T))
```

At step=1: `momentum = 1 - 1/min(1, √T) = 1 - 1 = 0`. The first step has no momentum — the gradient estimate completely replaces the momentum buffer with no smoothing.

**Impact**: Combined with zeroth-order noise, the first parameter update is purely stochastic. Not necessarily a bug — it's equivalent to "the first gradient estimate initializes the momentum buffer" — but unusual compared to standard optimizers that start with non-zero momentum. The ramp from 0 to ~0.97 (at T=1000) is aggressive; most optimizers use constant momentum in the 0.9–0.99 range.

---

## Design Gaps

### GAP-1: Checkpoint Save Exists, But No Resume

`_save_checkpoint()` (`controller.py:372-408`) correctly serializes full optimizer state at regular intervals:
- Master weights (A, B per layer)
- Momentum buffers (per-layer)
- Activation bases (per-layer V matrices)
- RNG state (`self.rng.get_state()`)
- Training state (step count, eta, temperature) as JSON

However, there is no corresponding `_load_checkpoint()` or resume mechanism. If training is interrupted, the saved state exists on disk but cannot be loaded — the controller must be reconstructed from scratch. The LR scheduler state is also not saved (only `eta` is persisted, but the `CosineAnnealingLR` internal state is lost).

**Recommended fix**: Add a `resume_from` config field and a load function that restores all saved state including LR scheduler.

---

### GAP-2: Ablation Script Hardcodes RL Mode

`eval/ablations.py`'s `patch_sgd_momentum` monkey-patches the controller to always run in RL mode, ignoring `hybrid_switch_step`. This makes SFT-mode ablations impossible without code modification.

**Recommended fix**: The ablation patches should respect the controller's existing `hybrid_switch_step` setting.

---

## Verified Correct Implementations

The following components were audited and found to be correctly implemented:

| Component | Location | Verification |
|:----------|:---------|:-------------|
| PiSSA ↔ PEFT naming swap | `controller.py` init, `backend.py` serialization, `prepare_pissa.py` | A=lora_B.weight, B=lora_A.weight — consistent across all three sites |
| Epsilon derivation | `controller.py:98-102` | `median(‖W‖_F) * eps^(1/3)` correctly implements Numerical Recipes §5.7 cube-root scaling for centered finite differences |
| Kernel N-S coefficients | `kernels.py:134` | Hardcoded `3.4445, -4.775, 2.0315` exactly match `torch.optim._muon.DEFAULT_A/B/C` |
| vLLM activation hooks | `backend.py` | `inp[0].detach().float()` with pinned memory + non_blocking DMA — correct for AGZO |
| Power iteration formula | `controller.py:141-144` | `ceil(log(log(1/eps)/log(2))/log(3))` gives k=3 for FP32. Note: formula implicitly assumes cubic convergence (log base 3), but power iteration converges linearly. Gives a practically reasonable number for warm-started subspace tracking but for the wrong theoretical reasons. |
| Cosine LR schedule | `controller.py` | PyTorch `CosineAnnealingLR` with `eta_max = eps`, decays to 0 |
| Temperature decay | `controller.py` | Cosine from 1.0 to 0.0 — standard exploration→exploitation |
| `_extract_prompt_logprobs` | `backend.py:68-76` | Correctly looks up by token ID (fixed from v1 bug) |
| Fused kernel dispatch | `kernels.py` | Tall/wide selection based on M vs N minimizes inner product dimension to rank×rank |
| `fused_perturb_dual` | `kernels.py` | Correct elementwise θ+ = base + εz, θ- = base - εz |
| `extract_code` regex | `eval/utils.py:17-26` | Handles ` ```python ` blocks, generic ` ``` ` blocks, and raw text fallback |
| GRPO baseline parity | `eval/grpo_baseline.py` | Same PiSSA init, same N=4 candidates — controlled comparison with DS-MeZO |

---

## Summary of Recommended Actions

| Priority | Item | Type | Action |
|:---------|:-----|:-----|:-------|
| **P0** | BUG-3: `optht` beta inverted | Bug | Fix `controller.py:182`: `beta = min(m, n) / max(m, n)` — crashes calibration for non-square activations |
| **P0** | BUG-1: RLOO div-by-zero | Bug | Add `std` guard at `controller.py:278` to prevent NaN propagation |
| **P0** | BUG-2: N-S iteration count | Bug | Replace `_ns_iters_for_smin()` with `ns_iterations = 5` (canonical Muon default) |
| **P1** | GAP-1: No checkpoint resume | Gap | Add load/resume function (save already exists, missing load + LR scheduler state) |
| **P1** | CLAUDE.md N-S description | Docs | Already partially fixed; canonical 5 iterations should replace scalar simulation |
| **P2** | DEV-3: RLOO best/worst only | Deviation | Consider using all N candidate advantages for better sample efficiency |
| **P2** | DEV-4: Single-prompt calibration | Deviation | Calibrate on 8–16 prompts for representative activation statistics |
| **P3** | GAP-2: Ablation RL hardcoding | Gap | Honor `hybrid_switch_step` in ablation patches |
| **P3** | DEV-1: AGZO PiSSA adaptation | Deviation | Document the two-factor subspace projection as a novel extension of AGZO |
