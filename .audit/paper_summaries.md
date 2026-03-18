# DS-MeZO Paper Reference Audit

Consolidated summaries of the 7 arXiv papers referenced by the DS-MeZO codebase, with
focus on algorithmic contributions that DS-MeZO implements or builds upon.

---

## 1. AGZO: Activation-Guided Zeroth-Order Optimization for LLM Fine-Tuning

- **arXiv:** [2601.17261](https://arxiv.org/abs/2601.17261)
- **Authors:** Wei Lin, Yining Jiang, Qingyu Song, Qiao Xiang, Hong Xu
- **Year:** 2026
- **Venue:** Preprint (submitted Jan 2026, revised Feb 2026)

### Core Method
AGZO identifies that the gradient of a linear layer is confined to the subspace spanned by
its input activations. It extracts a compact, activation-informed subspace on-the-fly during
the forward pass and restricts ZO perturbations to this low-rank subspace. This provably
yields update directions with higher cosine similarity to the true gradient than isotropic
baselines, while maintaining the same peak memory footprint as standard ZO methods.

### Key Algorithmic Details Relevant to DS-MeZO
- **Activation subspace extraction:** During forward pass, record input activations for each
  linear layer; compute a low-rank basis (top singular vectors) of the activation matrix.
- **Subspace-restricted perturbation:** Instead of isotropic Gaussian perturbations over all
  parameters, sample perturbations only within the activation-informed subspace.
- **DS-MeZO implementation:** The `_get_perturbation()` method in `controller.py` implements
  AGZO for the B matrix. It projects perturbations through activation bases `V_l` extracted
  via `extract_activations()` and `_update_activation_bases()`. The Triton kernel
  `fused_agzo_perturbation()` fuses the subspace projection into a single GPU kernel. For
  the A matrix, perturbations are projected through B's column space via QR decomposition
  (a DS-MeZO extension beyond the original AGZO paper).

---

## 2. The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm

- **arXiv:** [2505.16932](https://arxiv.org/abs/2505.16932)
- **Authors:** Noah Amsel, David Persson, Christopher Musco, Robert M. Gower
- **Year:** 2025
- **Venue:** Preprint (submitted May 2025, revised Sep 2025)

### Core Method
Polar Express introduces an optimal method for computing the polar decomposition (matrix
sign function) using only matrix-matrix multiplications, making it GPU-friendly. Inspired by
Chen & Chow and Nakatsukasa & Freund, it adapts the Newton-Schulz update rule at each
iteration by solving a minimax optimization problem (Equioscillation Theorem / Chebyshev
theory), minimizing worst-case error. This converges as rapidly as possible both in early
iterations and asymptotically, and is practical in bfloat16.

### Key Algorithmic Details Relevant to DS-MeZO
- **Per-iteration minimax coefficients:** Each Newton-Schulz iteration uses different
  coefficients (c1, c3) derived from the Equioscillation Theorem, rather than fixed
  coefficients. The iteration: `X_{k+1} = X_k * (c1*I + c3*(X_k^T @ X_k))`.
- **Convergence-based iteration count:** Iterate until `1 - p(ell) < eps_f32`, no hardcoded
  iteration count needed.
- **DS-MeZO implementation:** The `_ns_coefficients()` function in `kernels.py` implements
  exactly this: it derives per-iteration (c1, c3) pairs from the Equioscillation closed-form
  using `ell = sqrt(eps_f32)` as the Gram matrix roundoff floor, iterating until convergence
  within FP32 precision. Coefficients are precomputed at import time as `_NS_COEFFS` tuples
  and hardcoded into Triton JIT kernels. The `zo_muon_update()` kernel applies these
  coefficients to orthogonalize momentum before weight updates.

---

## 3. BSZO: Adaptive Bayesian Subspace Zeroth-Order Optimizer

- **arXiv:** [2601.01452](https://arxiv.org/abs/2601.01452)
- **Authors:** Jian Feng, Zhihong Huang
- **Year:** 2026
- **Venue:** Preprint (submitted Jan 2026)

### Core Method
BSZO applies Kalman filtering to combine finite-difference information across multiple
perturbation directions within a subspace. Each finite-difference measurement is treated as
a noisy observation; BSZO builds a posterior distribution over the subspace-projected
gradient and updates it through Bayesian inference with residual-based adaptive noise
estimation. This improves convergence by a factor of k/gamma vs. standard ZO methods while
remaining robust under fp16/bf16 precision.

### Key Algorithmic Details Relevant to DS-MeZO
- **Kalman observation update:** After computing the finite-difference scalar `dd`, treat it
  as a noisy observation of the true directional derivative. Update the posterior mean
  (momentum) and posterior variance per-element using Kalman gain: `K = var * z / S`,
  `mu += innovation * K`, `var *= (1 - K * z)`.
- **Variance-weighted perturbation:** Sample perturbation magnitudes proportional to
  posterior uncertainty (higher variance = explore more).
- **DS-MeZO implementation:** The `_update_weights()` method in `controller.py` implements
  Kalman observation updates for both A and B matrices: `K = var * z_mat * inv_S`,
  `mu.add_(innovation * K)`, `var.mul_(1.0 - K * z_mat)`. The `_get_perturbation()` method
  uses `variance_B` to weight B-perturbation coefficients: `var_B_proj = layer.variance_B @
  (V_l ** 2)`, `z_coeff_B = randn(...) * sqrt(var_B_proj)`. State variables `variance_A` and
  `variance_B` are initialized to `eps^2` and checkpointed.

---

## 4. MeZO-BCD: Elucidating Subspace Perturbation in Zeroth-Order Optimization

- **arXiv:** [2501.19099](https://arxiv.org/abs/2501.19099)
- **Authors:** Sihwan Park, Jihun Yun, SungYub Kim, Souvik Kundu, Eunho Yang
- **Year:** 2025
- **Venue:** Preprint (submitted Jan 2025, revised May 2025)

### Core Method
MeZO-BCD develops a unified theoretical framework analyzing convergence and generalization
of ZO optimization under subspace perturbations. The key theoretical contribution is the
notion of "subspace alignment" — explaining how subspace perturbations reduce gradient noise
and accelerate convergence. The practical method uses block coordinate descent, perturbing
and updating only a subset of parameters per step. Achieves up to 2.77x wall-clock speedup
over MeZO on OPT-13B.

### Key Algorithmic Details Relevant to DS-MeZO
- **Subspace alignment theory:** Convergence depends on alignment between the perturbation
  subspace and the gradient's principal subspace. Better alignment = lower variance in the
  ZO gradient estimate = faster convergence. This provides theoretical justification for
  AGZO-style activation-guided perturbations.
- **All-at-once vs. BCD:** MeZO-BCD cycles through parameter blocks; DS-MeZO instead
  perturbs all layers simultaneously (all-at-once SPSA), reducing from hundreds of scoring
  calls per step to just 4 (positive/negative for RL, positive/negative for SFT scoring).
- **DS-MeZO usage:** The subspace alignment framework from this paper justifies DS-MeZO's
  design choice to use activation-informed subspaces (high alignment) rather than random
  subspaces (low alignment). DS-MeZO does NOT implement BCD — it explicitly rejected
  per-block cycling in favor of all-at-once perturbation for wall-clock efficiency.

---

## 5. ZO-Muon: Powering Up Zeroth-Order Training via Subspace Gradient Orthogonalization

- **arXiv:** [2602.17155](https://arxiv.org/abs/2602.17155)
- **Authors:** Yicheng Lang, Changsheng Wang, Yihua Zhang, Mingyi Hong, Zheng Zhang, Wotao Yin, Sijia Liu
- **Year:** 2026
- **Venue:** Preprint (submitted Feb 2026)

### Core Method
ZO-Muon unifies two complementary principles: (i) projection-based subspace view that
reduces gradient estimation variance via intrinsic low-rank structure, and (ii) Muon-style
spectral optimization that applies gradient orthogonalization (Newton-Schulz polar
decomposition) to extract informative spectral structure from noisy ZO gradients. This
yields a low-rank Muon optimizer in the ZO setting. Requires only 24.7% of MeZO's queries
to reach the same SST-2 performance.

### Key Algorithmic Details Relevant to DS-MeZO
- **ZO gradient + Newton-Schulz orthogonalization:** After estimating the ZO gradient
  (finite-difference), apply Newton-Schulz iterations to compute the polar factor (matrix
  sign), then use that orthogonalized direction as the update. This extracts spectral
  structure that isotropic ZO gradients miss.
- **Low-rank projection:** Restrict perturbations to a low-rank subspace, then
  orthogonalize in that subspace — combining variance reduction with spectral optimization.
- **DS-MeZO implementation:** The `zo_muon_update()` kernel in `kernels.py` directly
  implements this: after the Kalman update produces momentum `mu`, the kernel applies
  Newton-Schulz orthogonalization (using Polar Express coefficients) to `mu` before using it
  as the weight update direction. The update: `param -= eta * NS_orthogonalize(mu)`. This
  is applied per-layer for both A and B matrices in the Kalman update loop of
  `_update_weights()`.

---

## 6. MeZO: Fine-Tuning Language Models with Just Forward Passes

- **arXiv:** [2305.17333](https://arxiv.org/abs/2305.17333)
- **Authors:** Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D. Lee, Danqi Chen, Sanjeev Arora
- **Year:** 2023
- **Venue:** NeurIPS 2023 (oral)

### Core Method
MeZO adapts classical ZO-SGD (SPSA) to operate in-place, fine-tuning LLMs with the same
memory footprint as inference. Using only two forward passes with shared random seeds for
perturbation reconstruction, it avoids storing activations or gradients. Demonstrates that
adequate pre-training enables ZO optimization of billion-scale models despite classical
theory suggesting otherwise. Trains 30B parameter models on a single A100 80GB GPU.

### Key Algorithmic Details Relevant to DS-MeZO
- **SPSA (Simultaneous Perturbation Stochastic Approximation):** Perturb all parameters
  with shared random seed z, evaluate f(theta+eps*z) and f(theta-eps*z), estimate gradient
  direction as `(f+ - f-) / (2*eps) * z`. The scalar `dd = (loss+ - loss-) / (2*eps)`.
- **In-place perturbation with random seed:** Reconstruct perturbation from seed rather
  than storing it, achieving inference-level memory.
- **DS-MeZO foundation:** DS-MeZO is built on MeZO's core SPSA mechanism but extends it
  with: (a) AGZO subspace perturbations instead of isotropic, (b) Bayesian (BSZO) variance
  tracking instead of plain SGD, (c) ZO-Muon orthogonalization of the update direction,
  (d) PiSSA adapters instead of full-parameter or LoRA, (e) RLOO RL advantages instead of
  supervised loss. The contrastive scoring loop in `_score_contrastive()` and the
  `dd = (loss_pos - loss_neg) / (2*eps)` pattern directly descend from MeZO's SPSA.

---

## 7. PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models

- **arXiv:** [2404.02948](https://arxiv.org/abs/2404.02948)
- **Authors:** Fanxu Meng, Zhaohui Wang, Muhan Zhang
- **Year:** 2024
- **Venue:** NeurIPS 2024 (spotlight)

### Core Method
PiSSA shares LoRA's architecture (W = W_res + A @ B) but initializes A and B with the
principal singular values/vectors of the original weight matrix W (via SVD), freezing the
residual W_res. Unlike LoRA (which initializes A with noise and B with zeros and updates
the "noise" adapter), PiSSA updates the principal components — the most important directions
for the weight matrix. This leads to faster convergence and consistently outperforms LoRA
across models from 184M to 70B parameters. Compatible with quantization (QPiSSA).

### Key Algorithmic Details Relevant to DS-MeZO
- **SVD initialization:** Compute `W = U @ diag(S) @ V^T`, take top-r singular triplets:
  `A = U[:,:r] @ diag(S[:r])`, `B = V[:,:r]^T`. Freeze `W_res = W - A @ B`.
- **Adapter form:** Same as LoRA (additive low-rank), so compatible with vLLM's LoRA
  serving infrastructure. The frozen residual is absorbed into base model weights.
- **Fast SVD:** Uses randomized SVD for negligible initialization cost.
- **DS-MeZO implementation:** The `prepare_pissa.py` script performs the SVD decomposition
  to create PiSSA adapters. During training, only A and B matrices are perturbed and
  updated (W_res stays frozen in the base model). The adapter rank `r` is calibrated via
  Gavish-Donoho optimal threshold (`optht` library) rather than manual selection. The
  `_calibrate_activation_bases_full()` method uses power iteration (with iteration count
  from `svd_power_iters()`) to extract activation bases, building on PiSSA's low-rank
  structure.

---

## Cross-Paper Integration in DS-MeZO

DS-MeZO synthesizes these papers into a unified pipeline:

```
                  MeZO (SPSA core)
                       |
              +--------+--------+
              |                 |
         PiSSA adapters    RLOO RL advantages
         (A, B matrices)   (reward-based, not loss-based)
              |
     AGZO subspace perturbation
     (activation-guided, not isotropic)
              |
     BSZO Bayesian tracking
     (Kalman posterior on ZO gradient)
              |
     ZO-Muon orthogonalization
     (Newton-Schulz on momentum)
              |
     Polar Express coefficients
     (minimax-optimal N-S iterations)
              |
     MeZO-BCD theory
     (subspace alignment justification)
```

**Per-step flow:**
1. Generate N rollouts, compute RLOO advantages (rewards - leave-one-out baselines)
2. Extract activation bases via forward hooks (AGZO)
3. Sample variance-weighted perturbations in activation subspace (AGZO + BSZO)
4. Create positive/negative adapter copies (SPSA from MeZO)
5. Score trajectories with both copies (4 vLLM calls)
6. Compute finite-difference scalar `dd` (MeZO's SPSA)
7. Kalman observation update on momentum and variance (BSZO)
8. Newton-Schulz orthogonalize momentum, apply as weight update (ZO-Muon + Polar Express)
9. Step cosine LR scheduler

**Key design choices vs. papers:**
- All-at-once perturbation (not BCD cycling from MeZO-BCD)
- PiSSA adapters only (not full-parameter as in original MeZO)
- SGD+momentum with N-S orthogonalization (not Adam as in MeZO)
- Rank calibrated via Gavish-Donoho (not manual as in PiSSA)
- N-S coefficients from Equioscillation Theorem (Polar Express, not fixed Muon defaults)
