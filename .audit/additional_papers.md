# DS-MeZO Additional Paper Audit

Audit of foundational papers that DS-MeZO builds on, with analysis of deviations.

---

## 1. MeZO: Fine-Tuning Language Models with Just Forward Passes

- **arXiv:** 2305.17333
- **Authors:** Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D. Lee, Danqi Chen, Sanjeev Arora
- **Year:** 2023 (NeurIPS 2023, oral)
- **Key algorithm:** Memory-efficient ZO-SGD (MeZO). Uses SPSA (Simultaneous Perturbation Stochastic Approximation) to estimate gradients with only two forward passes. The key insight is in-place perturbation: by saving the random seed and resampling identical noise `z`, MeZO avoids storing the perturbation vector, achieving inference-level memory. The gradient estimate is `g = (L(theta + eps*z) - L(theta - eps*z)) / (2*eps) * z`, applied element-wise via the resampled seed.

### How DS-MeZO builds on MeZO

DS-MeZO retains the core SPSA two-evaluation structure but makes several significant departures:

1. **Perturbation subspace:** MeZO perturbs all parameters with isotropic Gaussian noise in the full parameter space. DS-MeZO restricts perturbations to an activation-guided low-rank subspace (AGZO-style), perturbing only in the column space of activation bases for B and through QR(B@V) for A. This dramatically reduces the effective dimensionality of the perturbation from d (full parameter count) to ~128 effective dimensions per layer.

2. **Optimizer:** MeZO uses vanilla ZO-SGD (the estimated gradient is directly used for the update with a learning rate). DS-MeZO replaces this with a diagonal Kalman filter for gradient estimation (maintaining posterior mean and variance per parameter) followed by Newton-Schulz orthogonalization (ZO-Muon spectral update). This is a fundamental departure from the original simplicity.

3. **RL integration:** MeZO optimizes differentiable losses (cross-entropy). DS-MeZO uses SPSA within an RL loop: the loss difference is computed on advantage-weighted NLL across multiple candidate trajectories, enabling optimization of non-differentiable reward signals.

4. **Parameter scope:** MeZO operates on all model parameters (or LoRA adapters). DS-MeZO operates exclusively on PiSSA adapters (principal singular components), which changes what the perturbation "reaches."

### Assessment

The subspace restriction (point 1) is **well-justified** by AGZO theory showing gradients of linear layers lie in the activation subspace. The Kalman filter + Muon replacement of SGD (point 2) is a **significant departure** that adds complexity; the justification rests on the weak adaptivity hypothesis (MeZO-A3dam) and the observation that N-S normalization decouples direction from magnitude. The RL integration (point 3) is **novel and justified** since MeZO already demonstrated compatibility with non-differentiable objectives. Overall, DS-MeZO is a principled but aggressive extension of MeZO rather than a conservative adaptation.

---

## 2. PiSSA: Principal Singular Values and Singular Vectors Adaptation

- **arXiv:** 2404.02948
- **Authors:** Fanxu Meng, Zhaohui Wang, Muhan Zhang
- **Year:** 2024 (NeurIPS 2024, spotlight)
- **Key algorithm:** PiSSA shares LoRA's architecture (W = W_res + A @ B) but initializes A and B with the principal singular components of W via SVD, freezing the residual W_res. Specifically: W = U_r @ S_r @ V_r^T (principal, rank r) + W_res (residual). A = U_r @ sqrt(S_r), B = sqrt(S_r) @ V_r^T. This means the trainable parameters start at the most important directions, enabling faster convergence than LoRA's noise/zero initialization.

### How DS-MeZO builds on PiSSA

1. **Direct adoption:** DS-MeZO uses PiSSA initialization via HuggingFace PEFT's `init_lora_weights="pissa_niter_N"` parameter. The `prepare_pissa.py` script decomposes the base model into adapter (principal components) and residual, saved separately.

2. **No DoRA augmentation:** The v2 design explicitly removed DoRA (weight decomposition adaptation) that was considered in v1. DS-MeZO uses plain PiSSA adapters only, avoiding the complexity of magnitude vectors and three-phase BCD.

3. **Rank selection:** PiSSA uses a fixed user-specified rank. DS-MeZO adds automatic rank calibration via the Gavish-Donoho optimal threshold (`optht` library) applied to activation singular values, not weight singular values. This determines `r_calib` (the activation subspace rank), which is distinct from the adapter rank `r`.

4. **Convention swap:** In the codebase, PiSSA's `lora_A` maps to DS-MeZO's `B` and `lora_B` maps to `A` (line 78-79 of controller.py: `B = adapter_tensors[...lora_A.weight]`, `A = adapter_tensors[...lora_B.weight]`). This follows PiSSA's convention where lora_A is the right factor (input projection) and lora_B is the left factor (output projection).

### Assessment

The adoption of PiSSA is **well-justified** and straightforward. PiSSA's theoretical advantage (training principal components converges faster) is especially important in a ZO setting where convergence is already slower than first-order methods. The rank calibration via Gavish-Donoho on activations rather than weights is a **reasonable extension** but represents a departure: PiSSA's theory is about weight matrix principal components, while DS-MeZO calibrates rank based on activation structure. This could be **potentially problematic** if the activation rank and weight rank differ significantly, though in practice transformer activations and weights tend to share similar effective rank profiles.

---

## 3. RLOO: Back to Basics (REINFORCE Leave-One-Out for RLHF)

- **arXiv:** 2402.14740
- **Authors:** Arash Ahmadian, Chris Cremer, Matthias Galle, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Ustun, Sara Hooker
- **Year:** 2024
- **Key algorithm:** Revisits REINFORCE-style optimization for RLHF, showing that simple REINFORCE with a leave-one-out (LOO) baseline outperforms PPO and DPO. The LOO baseline for sample i from K samples is: `b_i = (sum of all rewards - r_i) / (K - 1)`. This provides a low-variance, unbiased baseline without requiring a separate value network. The paper shows that many PPO components (value function, GAE, clipping) are unnecessary for LLM alignment.

### How DS-MeZO builds on RLOO

1. **Core adoption:** DS-MeZO implements the standard RLOO advantage formula exactly:
   ```
   baselines_rloo = (rewards.sum() - rewards) / (N - 1)
   ```
   with N=4 candidates (NUM_CANDIDATES = 4).

2. **James-Stein shrinkage extension:** DS-MeZO adds a shrinkage-optimal baseline that blends the RLOO baseline with a reward EMA:
   ```
   lam = reward_var / (reward_var + (r_bar - reward_ema)^2 + floor)
   baselines = (1 - lam) * baselines_rloo + lam * reward_ema
   ```
   This is a James-Stein estimator: when reward variance is high relative to the bias of the EMA, lambda is large and the baseline shrinks toward the EMA (more stable). When variance is low, lambda is small and the pure RLOO baseline dominates.

3. **No backpropagation:** The original RLOO paper uses advantages in a standard policy gradient backward pass. DS-MeZO instead uses advantages as weights in contrastive SPSA scoring: advantage-weighted NLL is computed under perturbed parameters (theta+z and theta-z), and the difference drives the ZO gradient estimate.

4. **Small sample size:** The paper evaluates RLOO with typical batch sizes for RLHF (hundreds of samples). DS-MeZO uses only K=4 candidates per prompt, making the LOO baseline much noisier.

### Assessment

The RLOO adoption is **well-justified** as the simplest effective RL baseline for LLM alignment. The James-Stein shrinkage (point 2) is a **novel and theoretically grounded** extension not present in any existing RLHF framework; it should help stabilize training when K is small. The small sample size (K=4) is a **potential concern**: with only 4 candidates, the LOO baseline has high variance (each baseline is the mean of only 3 rewards). The James-Stein shrinkage partially mitigates this, but the interaction between noisy advantages and noisy ZO gradient estimates could compound errors. This is the most speculative component of the system.

---

## 4. Sparse MeZO: Less Parameters for Better Performance in ZO LLM Fine-Tuning

- **arXiv:** 2402.15751
- **Authors:** Yong Liu, Zirui Zhu, Chaoyu Gong, Minhao Cheng, Cho-Jui Hsieh, Yang You (NUS-HPC-AI-Lab)
- **Year:** 2024 (NeurIPS 2025)
- **Key algorithm:** Applies ZO perturbation only to a carefully chosen subset of parameters. Key insight: ZO estimation error hurts more when applied to large-magnitude weights than small-magnitude weights. Therefore, Sparse MeZO creates a binary mask `m` that selects parameters with smaller magnitudes for perturbation, leaving large weights unperturbed. The sparse gradient estimate is `g = (L(theta + eps*m*z) - L(theta - eps*m*z)) / (2*eps) * m*z`. Achieves 9% accuracy improvement and 3.5x speedup over MeZO on RTE.

### How DS-MeZO builds on / deviates from Sparse MeZO

1. **Not adopted (magnitude masking):** DS-MeZO does NOT use Sparse MeZO's magnitude-based sparse masking approach. The MEMORY.md explicitly notes: "Column-space projection for A: z_A projected through QR(B@V) -- B's column space in the activation subspace. (NOT sparse masking -- that was a v2 design doc idea never implemented.)"

2. **Alternative sparsity via subspace:** Instead of binary masks on individual parameters, DS-MeZO achieves dimensionality reduction through AGZO-style subspace restriction. Rather than selecting WHICH parameters to perturb, DS-MeZO restricts HOW parameters are perturbed (only along activation-informed directions). This is a fundamentally different approach to the same problem (reducing ZO estimation error).

3. **Philosophical difference:** Sparse MeZO is a parameter-space method (which weights to touch), while DS-MeZO uses a function-space method (which directions in weight space matter for the loss, as determined by activations). The AGZO paper (2601.17261) provides theoretical justification: gradients of linear layers are confined to the subspace spanned by input activations, so perturbations outside this subspace contribute only noise.

### Assessment

The decision to use subspace restriction instead of sparse masking is **well-justified** theoretically: AGZO's activation-subspace insight provides a principled basis for which directions matter, while magnitude-based masking is a heuristic (small weights may still be important if they lie in high-gradient directions). However, there is a **potential missed opportunity**: the two approaches are not mutually exclusive. Sparse MeZO's masking could be applied within the activation subspace (i.e., after projecting perturbations to the subspace, additionally mask out high-magnitude components). This composition could further reduce estimation error but would add complexity. The current design choice of subspace-only is defensible as simpler and more principled.

---

## 5. Muon: An Optimizer for Hidden Layers of Neural Networks

- **Reference:** Keller Jordan, GitHub (github.com/KellerJordan/Muon), originally described in an X thread and writeup (October 2024). Scaled to LLM training in arXiv:2502.16982 (Liu, Su, et al., February 2025).
- **Authors:** Keller Jordan (original); Jingyuan Liu, Zhilin Yang et al. (scaling paper)
- **Year:** 2024-2025
- **Key algorithm:** Muon computes the steepest descent direction under the spectral norm (operator norm) for weight matrices. The update rule is:
  1. Accumulate momentum: `M_t = beta * M_{t-1} + G_t`
  2. Compute polar factor: `U = polar(M_t)` (the orthogonal matrix closest to M in Frobenius norm)
  3. Update: `W_{t+1} = W_t - lr * U`

  The polar factor is approximated via Newton-Schulz (N-S) iteration: `X_{k+1} = X_k @ (c1*I + c3*X_k^T@X_k)`, starting from `X_0 = M/||M||_F`. The canonical implementation uses 5 iterations with fixed coefficients `(a, b, c) = (3.4445, -4.7750, 2.0315)` derived from the Muon repo. The scaling paper adds weight decay and per-parameter update scale adjustment for large models.

### How DS-MeZO builds on Muon

1. **ZO-Muon spectral update:** DS-MeZO applies the N-S orthogonalization step to the Kalman posterior mean (ZO gradient estimate) rather than to a true stochastic gradient. The fused Triton kernel (`_zo_muon_tall_kernel` / `_zo_muon_wide_kernel`) implements: Frobenius normalize the Kalman mean, run N-S iterations, then apply `param -= eta * X_final`.

2. **Per-iteration optimal coefficients (Polar Express):** The canonical Muon uses fixed coefficients for all N-S iterations. DS-MeZO uses per-iteration minimax-optimal coefficients derived from the Equioscillation Theorem (Amsel et al. 2025, arXiv:2505.16932, "The Polar Express"). The `_ns_coefficients()` function derives coefficients from the dtype precision:
   - Starting bound: `l = sqrt(eps_f32) ~ 3.45e-4`
   - Each iteration computes unique `(c1, c3)` from the current interval `[l, u]`
   - Iterates until convergence: `1 - l < eps_f32`
   - This produces 12 iterations (vs. canonical Muon's 5)

3. **No momentum in the Muon sense:** Canonical Muon maintains momentum `M_t = beta * M_{t-1} + G_t` and orthogonalizes the momentum. DS-MeZO's "momentum" is the Kalman posterior mean, which serves an analogous but distinct role: it is a variance-weighted running estimate of the ZO gradient, updated via Kalman prediction-observation rather than exponential averaging.

4. **No weight decay:** The scaling paper (2502.16982) identifies weight decay as crucial for scaling Muon to large models. DS-MeZO does not apply weight decay, relying instead on the PiSSA residual freezing (only adapter parameters are updated) as an implicit regularizer.

5. **12 vs. 5 N-S iterations:** The canonical Muon uses 5 iterations with fixed coefficients, which gives sufficient accuracy for the pre-normalized gradient regime. DS-MeZO's 12 iterations (from starting at l=sqrt(eps_f32)) are significantly more, motivated by convergence to machine precision. This is more iterations than necessary: the previous v3 audit found that hardcoding 5 was sufficient and the original scalar simulation bug returned 20 iterations.

### Assessment

The core idea of applying spectral normalization (N-S orthogonalization) to ZO gradient estimates is **novel and well-motivated** by the weak adaptivity hypothesis: N-S normalization decouples the update direction from the gradient magnitude, which is especially important when the gradient is a noisy ZO estimate. The per-iteration optimal coefficients from Polar Express are **theoretically superior** to the canonical fixed coefficients but come at the cost of 12 iterations vs. 5, a 2.4x increase in N-S compute. Given that N-S is <0.2% of step time (dominated by vLLM inference), this is **practically irrelevant**.

The absence of weight decay (point 4) is **potentially problematic** for long training runs, though the PiSSA structure (training only low-rank adapters of rank r) provides implicit regularization via the rank constraint. The Kalman filter replacing Muon's momentum (point 3) is a **significant but justified deviation**: in the ZO setting, the Kalman filter naturally handles the heteroscedastic noise in SPSA estimates, while simple exponential momentum does not.

**Potential issue:** The 12-iteration count contradicts the v3 audit finding that iterations should be hardcoded to 5. The current implementation derives this dynamically from dtype, producing 12 iterations starting from l=sqrt(eps_f32). While more accurate, the question is whether the ZO gradient estimate noise (which is much larger than the N-S approximation error from 5 iterations) makes the extra 7 iterations pointless. The additional precision in orthogonalization is wasted if the input (Kalman mean) is itself a noisy estimate. **Recommendation:** Consider reverting to 5-7 iterations as a minor efficiency improvement, though the practical impact is negligible given vLLM dominance.

---

## Cross-Cutting Analysis

### Coherence of the combined system

DS-MeZO weaves together five distinct methods into a single pipeline:

```
MeZO (SPSA framework)
  + PiSSA (adapter initialization)
  + AGZO-style subspace restriction (replacing Sparse MeZO's masking)
  + RLOO with James-Stein shrinkage (RL advantage estimation)
  + Muon-style N-S orthogonalization (update normalization)
  + Diagonal Kalman filter (gradient state estimation)
```

The **strongest theoretical link** is between the subspace restriction (AGZO) and the N-S orthogonalization (Muon): AGZO constrains perturbations to the activation subspace where gradients live, and N-S normalizes the estimated gradient to a spectral-norm steepest descent direction within that subspace.

The **weakest theoretical link** is between the Kalman filter and N-S orthogonalization: the Kalman filter produces a posterior mean with heterogeneous uncertainty across parameters, but N-S orthogonalization treats the entire matrix uniformly. A more principled approach might apply N-S to a variance-whitened gradient, but this would be prohibitively expensive.

### Risk ranking of deviations

| Deviation | Risk | Justification |
|:---|:---|:---|
| SPSA on RL advantages (vs. MeZO's direct loss) | Medium | Novel composition, untested at scale |
| K=4 candidates for RLOO | Medium | High-variance baseline, mitigated by James-Stein |
| Kalman filter replacing Muon momentum | Low | Theoretically superior for heteroscedastic ZO noise |
| Subspace restriction replacing sparse masking | Low | Stronger theoretical basis (AGZO) |
| PiSSA adoption | Very Low | Direct, well-supported |
| 12 N-S iterations vs. 5 | Very Low | Negligible cost, slightly wasteful |
| No weight decay (vs. scaled Muon) | Low | Implicit regularization via rank constraint |
