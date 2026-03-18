# DS-MeZO Resource Paper Analysis

Comprehensive extraction of algorithmic details from reference papers for implementation validation.

---

## 1. DS-MeZO (Zeroth-Order Optimization Finds Flat Minima)

**Source:** `resources/ds_mezo/paper.md`

### Key Algorithm
The paper proves that zeroth-order optimization with the standard two-point SPSA estimator has an implicit regularization that favors solutions with small trace of Hessian (flat minima). This provides theoretical justification for using ZO methods in LLM fine-tuning.

### Mathematical Formulas

**Two-point SPSA gradient estimator (MeZO formulation):**
```
g_z(theta) = [L(theta + eps*z) - L(theta - eps*z)] / (2*eps) * z
```
where `z ~ N(0, I_d)` and `eps` is the perturbation scale.

**Memory-efficient trick:** MeZO saves the random seed and resamples the same noise `z` to avoid storing the perturbation vector. The parameter is perturbed to `theta + eps*z`, loss computed, then resampled back to `theta - eps*z` for the second loss.

**Convergence result:** ZO methods converge to approximate flat minima (minimizers with smallest trace of Hessian among all optimal solutions) for convex and sufficiently smooth functions.

### Implementation Guidance
- The ZO gradient estimator requires exactly 2 forward passes per gradient estimate (no backward pass).
- The perturbation scale `eps` controls the bias-variance tradeoff: smaller `eps` reduces bias but increases variance.
- The implicit flat-minima regularization is a property of the SPSA estimator itself, not an explicit regularizer.

### DS-MeZO Usage
DS-MeZO uses the SPSA two-point estimator as its core gradient estimation mechanism. It extends this by:
- Operating on PiSSA adapter parameters (not full model weights)
- Using RLOO advantages to weight the contrastive scores instead of raw loss differences
- Applying Newton-Schulz orthogonalization to the ZO gradient estimate (ZO-Muon)
- Using activation-guided subspace perturbation (AGZO) rather than full-dimensional random perturbation

---

## 2. PiSSA (Principal Singular Values and Singular Vectors Adaptation)

**Source:** `resources/pissa/paper.md`

### Key Algorithm
PiSSA decomposes a pretrained weight matrix W via SVD and initializes low-rank adapters with the principal (largest) singular components, freezing the residual.

### Mathematical Formulas

**SVD decomposition:**
```
W = U * Sigma * V^T
```

**Partition into principal and residual:**
```
W^{pri} = U[:, :r] * Sigma[:r, :r] * V[:, :r]^T = A * B
W^{res} = U[:, r:] * Sigma[r:, r:] * V[:, r:]^T
```
where:
- `A = U[:, :r] * Sigma[:r, :r]^{1/2}` (shape m x r)
- `B = Sigma[:r, :r]^{1/2} * V[:, :r]^T` (shape r x n)
- `r << min(m, n)` is the adapter rank

**Effective weight during inference:**
```
W = W^{res} + A * B
```
where `A, B` are trainable and `W^{res}` is frozen.

**Quantized variant (QPiSSA):**
```
W = nf4(W^{res}) + A * B
```
QPiSSA quantizes the *residual* (not the original weight like QLoRA), preserving principal components in full precision.

**Fast SVD:** PiSSA supports approximate initialization via randomized SVD (Halko et al.) for speed, with `niter` controlling accuracy.

### Implementation Guidance
- Initialization requires a one-time SVD of each target weight matrix.
- Fast SVD significantly reduces init time (tens of times faster) with minimal training loss impact when niter >= 4.
- The adapter matrices A, B have identical architecture to LoRA — PiSSA is a drop-in replacement differing only in initialization.
- PiSSA initializes adapters with the most impactful directions of the original weight, enabling faster convergence than LoRA's noise/zero initialization.

### DS-MeZO Usage
DS-MeZO uses PiSSA as its adapter initialization strategy (v2 decision: plain PiSSA, no DoRA). Key deviations:
- DoRA magnitude/direction decomposition was removed in v2 to simplify.
- The rank `r` is one of the two primary hyperparameters.
- Rank selection uses Gavish-Donoho optimal threshold via `optht` library (v3 upgrade from ad-hoc argmax-of-diffs).
- PiSSA adapters are the substrate on which SPSA perturbation and ZO-Muon updates operate.

---

## 3. Sparse MeZO (S-MeZO)

**Source:** `resources/sparse_mezo/paper.md`

### Key Algorithm
Sparse MeZO applies a binary mask `m` based on parameter magnitudes to restrict ZO perturbation to a subset of parameters, updating only those with smaller magnitudes.

### Mathematical Formulas

**Sparse perturbation:**
```
z_hat = m * z,    where z ~ N(0, I_d)
```

**Sparse gradient estimator:**
```
g_m(theta) = [L(theta + eps * m * z) - L(theta - eps * m * z)] / (2 * eps) * (m * z)
```

**Mask construction (magnitude-based):**
For each layer i with threshold h_i:
```
m_i = 1  if |theta_i| < h_i    (update small-magnitude parameters)
m_i = 0  otherwise              (freeze large-magnitude parameters)
```
The threshold h_i is set to achieve a target sparsity ratio (fraction of parameters masked out).

**Optimal sparsity:** Experiments show sparsity 0.5-0.8 yields best results, with 0.8 optimal for most tasks.

### Implementation Guidance

**Memory-efficient mask computation:** Two approaches:
1. **1-bit quantization:** Store mask as binary, but still adds memory.
2. **Forward-pass mask computation (recommended):** Compute mask and perturb parameters during forward pass, eliminating need to store perturbed parameters. This avoids the issue that the mask (based on magnitudes) changes when parameters are perturbed.

**Algorithm pseudocode:**
```
1. Set threshold h_i per layer before training
2. For each step:
   a. GetMask: compare |theta_i| against h_i to create mask m
   b. PerturbParameters: sample z ~ N(0,1), apply z_hat = m * z
   c. Compute L+ = L(theta + eps*z_hat) and L- = L(theta - eps*z_hat)
   d. proj_grad = (L+ - L-) / (2*eps)
   e. gradient = proj_grad * z_hat
   f. Update: theta = theta - lr * gradient
```

### DS-MeZO Usage
DS-MeZO does NOT use Sparse MeZO's magnitude-based masking (this was a v2 design idea that was never implemented). Instead, DS-MeZO uses:
- **AGZO subspace perturbation:** Activation-guided perturbation in low-rank subspace (128 effective dims/layer for B matrix).
- **Column-space projection for A:** z_A projected through QR(B@V) — B's column space in the activation subspace.
- These are conceptually related (restricting perturbation to a subspace) but use activation statistics rather than parameter magnitudes.

---

## 4. Polar Express (Optimal Matrix Sign Methods)

**Source:** `resources/polar_express/polar_express.md` and `2505.16932v3.pdf` (Amsel et al., 2025)

### Key Algorithm
Polar Express computes the polar decomposition polar(M) = UV^T by iteratively applying minimax-optimal odd polynomials. Unlike fixed Newton-Schulz, it adapts the polynomial at each iteration to be optimal over the current singular value interval.

### Mathematical Formulas

**Polar decomposition:**
```
M = U * Sigma * V^T  =>  polar(M) := U * V^T
```

**Standard Newton-Schulz (degree-3):**
```
X_0 = M / ||M||_F
X_{t+1} = (3/2)*X_t - (1/2)*X_t*(X_t^T * X_t)
```
Equivalently: p(x) = (3/2)x - (1/2)x^3

**Muon's Newton-Schulz (Jordan's degree-5, heuristic):**
```
X_{t+1} = a*X_t + b*(X_t*X_t^T)*X_t + c*(X_t*X_t^T)^2*X_t
```
where (a, b, c) = (3.4445, -4.7750, 2.0315). This does NOT converge — plateaus at error ~0.3.

**Polar Express (degree-5, optimal):**
Each iteration applies a different minimax-optimal polynomial p_t(x) = a_t*x + b_t*x^3 + c_t*x^5:
```
A = X @ X.mT                      # Gram matrix X*X^T
B = b * A + c * A @ A             # b*X^3 + c*X^5 part
X = a * X + B @ X                 # full update
```

**Theorem 4.1 (Greedy optimality):** Define ell_1 = ell, u_1 = u. For t = 1,...,T:
```
p_t = argmin_{p in P^odd_d} max_{x in [ell_t, u_t]} |1 - p(x)|
ell_{t+1} = min_{x in [ell_t, u_t]} p_t(x)
u_{t+1} = max_{x in [ell_t, u_t]} p_t(x)
```
The resulting composition p* = p_T o ... o p_1 is globally optimal.

**Equioscillation Theorem (degree-5 case):**
For p(x) = ax + bx^3 + cx^5 on [ell, u], optimality requires 4 equioscillating points {ell, q, r, u}:
```
p(ell) = 1 - E,  p(q) = 1 + E,  p(r) = 1 - E,  p(u) = 1 + E
```
Algorithm 3 alternates between:
1. Solving 4x4 linear system for a, b, c, E given q, r
2. Finding new q, r from roots of p'(x) (quadratic formula on even degree-4 polynomial)
until q, r converge.

**Degree-3 closed-form:**
```
p(x) = beta * p_NS(alpha * x)
where p_NS(x) = (3/2)x - (1/2)x^3
alpha = sqrt(3 / (u^2 + ell*u + ell^2))
beta = 4 / (2 + ell*u*(ell + u)*alpha^3)
```

**Convergence (Theorem 4.3):**
```
||polar(M) - X_T||_2 <= |1 - ell^2|^{(q+1)^T}
```
For d=3: quadratic convergence. For d=5: cubic convergence.

### Choosing the Starting Bound ell

**Upper bound u:** Trivially u = 1 after normalizing X_0 = M / (||M||_F + 10^{-2}).

**Lower bound ell:** Must be guessed. Consequences of a bad guess are mild:
- Any ell in (0, u] works; wrong guess only delays convergence by a few iterations.
- For bfloat16: machine epsilon = 2^{-7} = 0.0078125, so ell ~ eps_mach is a good guess.
- Paper uses ell = 10^{-3} for experiments.

### Finite Precision Stabilization (Section 4.4)

Three modifications for bfloat16:
1. **Safety factor:** Replace p_t(x) with p_t(x/1.01) to prevent singular values from growing beyond u_t due to roundoff. Coefficients adjusted offline: `(a/1.01, b/1.01^3, c/1.01^5)`.
2. **Cushioning:** When ell_t < u_t/10, set ell_t = u_t/10 to ensure p_t(x)/x >= 0.236, preventing sign flips.
3. **Normalization offset:** Use ||M||_F + 10^{-2} instead of ||M||_F.

### Algorithm 1 Precomputed Coefficients (ell=10^{-3}, d=5)
```python
coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.41880731195256773),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.37500016454474248),
    (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
]
# Safety factor applied: (a/1.01, b/1.01^3, c/1.01^5)
```

### DS-MeZO Usage
DS-MeZO implements Newton-Schulz orthogonalization as part of its ZO-Muon update. Key design decisions from the project memory:
- **v3:** Upgraded from 2-term Pade to 3-term canonical Muon coefficients (from `torch.optim.Muon`).
- **v4:** Replaced heuristic coefficients entirely with Equioscillation Theorem closed-form derivation from Polar Express.
  - ell = sqrt(eps_f32) (Gram matrix roundoff floor, dtype-derived)
  - Iterate until max error < eps_f32 (convergence criterion)
  - Per-iteration coefficients from closed-form (no hardcoded table)
- **N-S iteration count:** v3 identified a BUG where scalar simulation always returned 20 (Muon polynomial p(1)=0.701, intentionally doesn't converge to 1.0). Fixed to hardcoded 5 (canonical Muon DEFAULT_NS_STEPS).
- Triton kernel constants MUST be hardcoded (JIT compilation to PTX).

---

## 5. UNSO (Unified Newton-Schulz Orthogonalization)

**Source:** `resources/unso/unso.md` and `2602.02500v1.pdf` (Hu et al., 2026)

### Key Algorithm
UNSO replaces the iterative N-S structure with a single unified polynomial operation, avoiding repeated matrix multiplications along the long dimension.

### Mathematical Formulas

**Conventional NS iteration:**
```
X_{k+1} = (1/2) * X_k * (3I - X_k^T * X_k),   X_0 in R^{w x h}
```
Converges when sigma_max(X_0) < sqrt(3).

**General polynomial NS family:**
```
X_{k+1} = a*X_k + b*(X_k*X_k^T)*X_k + c*(X_k*X_k^T)^2*X_k
```
Jordan's coefficients: (a, b, c) = (3.4445, -4.7750, 2.0315) — achieves quadratic convergence but introduces noise.

**UNSO unified polynomial (non-iterative):**
Instead of iterating, UNSO expands N iterations into a single polynomial:
```
X_N = a*X_0 + b*X_0*X_0^T*X_0 + ... + c*(X_0*X_0^T)^n*X_0
```

Specifically, with exponential step-size parameterization n_k = 2^{k-1}:
```
Y = X + sum_{k=1}^{N-1} a_k * (I - X*X^T)^{2^{k-1}} * X + c * (I - X*X^T)^{2^{N-1}} * X
```

**Normalization:**
```
A = M    if h <= w;    M^T    if h > w
X = A / ||A*A^T||_F^{1/2}
```
This ensures singular values lie in [0, 1].

**Coefficient optimization:**
- Coefficients a_k are learnable, optimized via Adam.
- The last coefficient c is derived analytically:
```
c = e^{1/2} * (2^{N/2} - 1) - sum_{k=1}^{N-1} |a_k|
```

**Term significance:** Exponential growth n_k = 2^{k-1} produces sharper, more localized peaks for higher k. The N-th term (highest power) converges to 1 first. Default N=14 provides best accuracy/stability tradeoff.

**Error metric:**
```
Error = sqrt(sum_{i,j} E_{i,j}^2)    where E = Y^T*Y - I
```

### Implementation Guidance
- UNSO computes the Gram matrix X*X^T once (h x h), then applies polynomial powers.
- The key efficiency gain: multiplications along the long dimension (w) reduced from N to 1.
- Matrix multiplications along the short dimension (h x h) are added but much cheaper.
- Recommended polynomial order: N=14.
- Achieves Error ~0.040 vs. Origin NS ~0.487, Muon NS ~3.84 (128x128 matrices).

### DS-MeZO Usage
DS-MeZO does not directly use UNSO. The UNSO paper is referenced as context for understanding the N-S iteration landscape. DS-MeZO's N-S implementation follows the Polar Express approach (minimax-optimal per-iteration polynomials) rather than UNSO's unified single-polynomial expansion. Key differences:
- Polar Express: adaptive per-iteration polynomials with theoretical optimality guarantees.
- UNSO: single-shot learned polynomial, more efficient but less theoretically grounded.
- DS-MeZO v4 derives coefficients from the Equioscillation Theorem (Polar Express), not learned coefficients (UNSO).

---

## 6. DoRA (Weight-Decomposed Low-Rank Adaptation)

**Source:** `resources/dora/paper.md`

### Key Algorithm
DoRA decomposes pretrained weights into magnitude and direction components, then fine-tunes both — magnitude as a learned vector, direction via LoRA.

### Mathematical Formulas

**Weight decomposition:**
```
W = m * V / ||V||_c = ||W||_c * W / ||W||_c
```
where:
- `m in R^{1 x k}` is the magnitude vector (column-wise norms)
- `V in R^{d x k}` is the directional matrix
- `||.||_c` is the vector-wise norm across each column

**DoRA initialization:**
```
m = ||W_0||_c        (magnitude from pretrained weights)
V = W_0              (directional matrix = pretrained weights)
```

**DoRA fine-tuning update:**
```
W' = m * (V + Delta_V) / ||V + Delta_V||_c
```
where:
- `m` is trainable (magnitude vector)
- `V` is frozen (original directional matrix)
- `Delta_V = B * A` (LoRA decomposition of directional update)
- `B in R^{d x r}`, `A in R^{r x k}` are trainable low-rank matrices

**Key insight from weight decomposition analysis:**
- LoRA tends to make proportional magnitude and directional changes.
- Full fine-tuning (FT) shows a distinctive negative correlation: magnitude decreases while directional change increases.
- DoRA decouples these, allowing LoRA to focus purely on directional adaptation.

**Gradient analysis:** DoRA's gradient for the directional component is:
```
grad_V' = m / ||V + Delta_V||_c * (I - V'*V'^T / ||V'||^2) * grad_L
```
The projection `(I - V'*V'^T / ||V'||^2)` removes the component along the current direction, stabilizing optimization.

### Implementation Guidance
- DoRA adds one trainable vector `m` per adapted weight matrix (negligible parameter overhead).
- The column-wise normalization `||V + Delta_V||_c` must be recomputed at each forward pass.
- DoRA is compatible with quantization (QDoRA) and other LoRA variants.
- For NF4 quantization, fusing DoRA back into the base model requires care to avoid corruption.

### DS-MeZO Usage
**DoRA was REMOVED in DS-MeZO v2.** Key reasons from project memory:
- Eliminated three-phase BCD (block coordinate descent), fusion SVD, and magnitude vector.
- `fuse_dora_to_lora` had an NF4 corruption bug — DoRA fusion was eliminated entirely.
- DS-MeZO uses plain PiSSA adapters only (no magnitude/direction decomposition).
- This simplification reduced the parameter surface and removed a source of bugs.

---

## 7. Polar Express PDF — Detailed Newton-Schulz Formulas

**Source:** `2505.16932v3.pdf` (Amsel, Persson, Musco, Gower — September 2025)

### Exact Newton-Schulz Iteration Formulas

**Degree-3 (classical Newton-Schulz):**
```
X_{t+1} = (3/2)*X_t - (1/2)*X_t*(X_t^T*X_t)
p(x) = (3/2)x - (1/2)x^3
```
Convergence: quadratic, order (q+1)^T = 2^T for q=1.

**Degree-5 (higher-order Newton-Schulz / Pade):**
```
p(x) = (15x - 10x^3 + 3x^5) / 8
```
Convergence: cubic, order 3^T.

**Degree-5 Polar Express form (per-iteration, used in Algorithm 1):**
```
A = X @ X.mT                     # Gram matrix (h x h)
B = b * A + c * A @ A            # polynomial in Gram
X = a * X + B @ X                # update: aX + bX^3 + cX^5
```
where (a, b, c) change per iteration from the precomputed `coeffs_list`.

### Equioscillation Theorem Details

**For degree d = 2q+1 odd polynomials on [ell, u]:**
An odd polynomial p is the minimax-optimal approximation to the constant function 1 on [ell, u] if and only if there exist q+2 equioscillating points where |1 - p(x)| = E (the minimax error) and the sign alternates.

**Degree-3 (q=1, 3 equioscillating points):**
The equioscillating set is {ell, sqrt(-a/(3b)), u}, yielding a closed-form solution via rescaling Newton-Schulz:
```
p(x) = beta * p_NS(alpha * x)
alpha = sqrt(3 / (u^2 + ell*u + ell^2))
beta = 4 / (2 + ell*u*(ell + u)*alpha^3)
```

**Degree-5 (q=2, 4 equioscillating points):**
The equioscillating set is {ell, q, r, u} where p achieves alternating errors:
```
p(ell) = 1 - E,  p(q) = 1 + E,  p(r) = 1 - E,  p(u) = 1 + E
```
Solved by Algorithm 3 (alternating iteration):
1. Given q, r: solve 4x4 linear system for a, b, c, E
2. Given a, b, c: find q, r as roots of p'(x) via quadratic formula (p' is degree-4 even polynomial)
3. Repeat until convergence (< 5 iterations observed in practice)

### Convergence Criterion

**Theorem 4.3:** For M normalized so sigma(M) in [ell, 1]:
```
||polar(M) - X_T||_2 <= |1 - ell^2|^{(q+1)^T}
```
- Degree 3 (q=1): error <= |1 - ell^2|^{2^T}  (quadratic)
- Degree 5 (q=2): error <= |1 - ell^2|^{3^T}  (cubic)

**Practical convergence:** For ell = 10^{-3}, d=5:
- After 7 iterations: coefficients converge to (1.875, -1.25, 0.375) which is the Pade approximant.
- Once ell/u >= 1 - eps_double^{1/3}, the optimal polynomial equals the Pade approximant up to machine precision.

### Starting Bound ell Selection

**Upper bound:** u = 1 after normalization X_0 = M / (||M||_F + 10^{-2}).

**Lower bound ell:**
- Difficult to efficiently find a good lower bound on sigma_min.
- **Must guess.** Consequences of bad guess: only delays convergence by a few iterations.
- For bfloat16: eps_mach = 2^{-7} = 0.0078125, so ell ~ eps_mach is reasonable.
- Paper's experiments: ell = 10^{-3} and u = 1.
- Since bounds are fixed for all inputs, polynomials are precomputed once (offline stage).
- **DS-MeZO v4 deviation:** Uses ell = sqrt(eps_f32) ~ 1.09e-4 as the Gram matrix roundoff floor, justified as sigma^2 < epsilon means below representable precision.

---

## 8. UNSO PDF — Detailed Formulas

**Source:** `2602.02500v1.pdf` (Hu et al., January 2026)

### Conventional NS Iteration
```
X_{k+1} = (1/2) * X_k * (3I - X_k^T * X_k)
```
Requires sigma_max(X_0) < sqrt(3) for convergence.

### UNSO Unified Framework

**Core insight:** N iterations of NS can be unrolled into a single polynomial in the projection operator (I - X*X^T):

**Unified polynomial (Eq. 13):**
```
Y = X + sum_{k=1}^{N-1} a_k * (I - X*X^T)^{2^{k-1}} * X
    + [e^{1/2} * (2^{N/2} - 1) - sum_{k=1}^{N-1} |a_k|] * (I - X*X^T)^{2^{N-1}} * X
```

**Simplified (Eq. 15):**
```
Y = X + sum_{k=1}^{N-1} a_k * (I - X*X^T)^{n_k} * X + b * (I - X*X^T)^{n_N} * X
```
where n_k = 2^{k-1} (exponential step-size).

**Per-term scalar function (Eq. 18):**
```
f_k(x) = x * (1 - x^2)^{2^{k-1}}    for k >= 1
f_0(x) = x                            for k = 0
```

**Extreme point of f_k (Eq. 20):**
```
x* = 1 / sqrt(2^k + 1)
y* ~ (1/sqrt(2^k + 1)) * e^{-1/2}
```
Higher k produces sharper, more localized peaks closer to 0.

### Coefficient Derivation

**Last coefficient b (Eq. 23-24):**
```
b = (sqrt(2^N + 1) - 1 - sum_{k=1}^{N-1} a_k * (2^N / (2^N + 1))^{2^{k-1}}) * ((2^N + 1) / (2^N))^{2^{N-1}}
```
Approximately:
```
b ~ e^{1/2} * (2^{N/2} - 1) - sum_{k=1}^{N-1} a_k
```

**Learnable coefficients a_k:** Optimized via Adam optimizer with lr=0.1, decayed by 0.5 every 10000 iterations over 20000 epochs. Loss: uniform sampling of 1000 points from (0, 1) to approximate sign(x).

### UNSO Algorithm (Algorithm 1)

```
Input: X in R^{n x n}, polynomial order N, learned coefficients {a_k}
Pre-compute: c = e^{1/2} * (2^{N/2} - 1) - sum_{k=1}^{N-1} |a_k|
Initialize: I = identity, X_1 = I - X*X^T
Polynomial expansion: for k = 2 to N: X_k = X_{k-1} * X_{k-1}
Polynomial aggregation: Y = I; for k=1 to N-1: Y = Y + a_k*X_k; Y = Y + c*X_N
Final: Y = Y * X
```

### Key Results
- N=14 gives best accuracy/stability tradeoff (Error ~0.040).
- Outperforms Origin NS (0.487), Muon NS (3.846), Cesista NS (0.330), CANS (1.311).
- Single matrix multiplication along long dimension (w) vs. N multiplications for iterative methods.

---

## Cross-Reference Summary: What DS-MeZO Uses vs. What It Doesn't

| Component | Used? | Source Paper | Notes |
|-----------|-------|-------------|-------|
| SPSA two-point estimator | YES | DS-MeZO | Core ZO gradient estimation |
| PiSSA adapter init | YES | PiSSA | SVD-based principal component adapters |
| Sparse magnitude masking | NO | Sparse MeZO | Replaced by AGZO subspace perturbation |
| N-S orthogonalization | YES | Polar Express | ZO-Muon update direction |
| Equioscillation coefficients | YES (v4) | Polar Express | Replaced hardcoded Jordan/Muon coefficients |
| UNSO unified polynomial | NO | UNSO | Context only; DS-MeZO uses iterative Polar Express |
| DoRA magnitude/direction | NO (removed v2) | DoRA | Caused NF4 corruption bug; simplified to plain PiSSA |
| RLOO advantages | YES | Standard RL | With James-Stein shrinkage extension |
| Cosine LR schedule | YES (v2+) | PyTorch | Replaced GreedyLR; eta_min = eta_max/100 |
| SGD + momentum | YES (v2+) | Standard | Replaced Noise-Calibrated Tensor Adam |
| Gavish-Donoho rank | YES (v3+) | optht library | Replaced ad-hoc rank selection |

---

## Key Implementation Validation Points

1. **SPSA scalar:** The contrastive score difference `(L+ - L-) / (2*eps)` must be computed with the SAME perturbation vector z for both forward passes.

2. **PiSSA init:** A and B must be initialized from the TOP-r singular values/vectors of W, not random or bottom-r.

3. **N-S polynomial form:** Must be `X = aX + b(XX^T)X + c(XX^T)^2 X`, NOT `X = aX + bX(X^TX) + cX(X^TX)^2`. The Gram matrix form `XX^T` (h x h) is more efficient when h < w.

4. **N-S normalization:** X_0 = M / (||M||_F + 10^{-2}), NOT X_0 = M / ||M||_F. The offset prevents division instability.

5. **Safety factor:** Coefficients must be adjusted as (a/1.01, b/1.01^3, c/1.01^5) for numerical stability in bfloat16.

6. **Convergence criterion:** DS-MeZO v4 uses ell = sqrt(eps_f32) and iterates until error < eps_f32, which is dtype-derived (no magic numbers).

7. **Jordan coefficients (3.4445, -4.7750, 2.0315) do NOT converge to polar(M).** They plateau at error ~0.3. This was the v3 bug — using these with a convergence check that expects 1.0 would loop forever.

8. **RLOO formula:** `baselines = (rewards.sum() - rewards) / (N - 1)` — the leave-one-out mean excludes the current sample.

9. **Sparse MeZO mask:** Only small-magnitude parameters are updated (m=1 where |theta| < threshold). DS-MeZO does not use this but uses activation-guided subspace projection instead.

10. **DoRA fusion:** Converting DoRA back to LoRA format for inference requires `W' = m * (V + BA) / ||V + BA||_c`, which is numerically sensitive with NF4 quantization. DS-MeZO avoids this entirely.
