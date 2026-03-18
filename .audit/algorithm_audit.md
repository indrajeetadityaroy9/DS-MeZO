# DS-MeZO Algorithm Audit

**Date:** 2026-03-17
**Scope:** All algorithmic components across 9 source files
**Methodology:** Line-by-line code review against claimed mathematical specifications

---

## 1. SPSA Gradient Estimation

**Status: CORRECT**

### Two-point SPSA formula: dd = (L+ - L-) / (2*eps)

**Evidence:** `controller.py` lines 321-322:
```python
loss_pos, loss_neg = self._score_contrastive(trajectories, advantages, prompt_len)
dd = float(loss_pos - loss_neg) / (2.0 * self.eps)
```
This is the standard two-point simultaneous perturbation stochastic approximation (SPSA) finite difference. Correct.

### Perturbation symmetry: theta+ = theta + z, theta- = theta - z

**Evidence:** `controller.py` lines 262-267 calls `fused_perturb_dual()` which invokes `_perturb_dual_kernel` (`kernels.py` lines 472-484):
```python
tl.store(pos_ptr + offs, base + z, mask=mask)
tl.store(neg_ptr + offs, base - z, mask=mask)
```
Symmetric perturbation confirmed for both A and B matrices.

### Contrastive scoring computes advantage-weighted NLL

**Evidence:** `controller.py` lines 213-220:
```python
lp_pos = [lp[prompt_len:] for lp in self.backend.score(trajectories, self.backend.lora_pos)]
lp_neg = [lp[prompt_len:] for lp in self.backend.score(trajectories, self.backend.lora_neg)]
loss_pos = sum(adv * _mean_nll(lp) for adv, lp in zip(advantages, lp_pos))
loss_neg = sum(adv * _mean_nll(lp) for adv, lp in zip(advantages, lp_neg))
```
The scoring evaluates all NUM_CANDIDATES=4 trajectories under both theta+ and theta- adapters. The advantage-weighted NLL aggregation is correct: each trajectory's NLL is weighted by its RLOO advantage, producing a policy-gradient-aligned loss signal for the SPSA estimate. `_mean_nll` (line 24-25) computes `-sum(logprobs) / len(logprobs)` which is correct (negating log-probs gives NLL).

**Detail:** The log-probs are sliced from `prompt_len:` onward (line 214-215), so only the generated tokens contribute to the loss, not the prompt tokens. This is the standard RLHF/RL-from-feedback convention.

---

## 2. AGZO (Activation-Guided Zeroth-Order) Perturbation

**Status: CONCERN**

### Subspace projection for z_B

**Evidence:** `controller.py` lines 197-211:
```python
V_l = self.activation_bases[layer.key]  # (d_in, r_calib)
var_B_proj = layer.variance_B @ (V_l ** 2)  # (R, d_in) @ (d_in, r_calib) = (R, r_calib)
z_coeff_B = torch.randn(...) * torch.sqrt(var_B_proj)  # (R, r_calib)
```
Then in the kernel (`kernels.py` lines 366-375):
```python
z_B_tile = zcb @ V_tile.T  # (R, r_calib) @ (r_calib, BLOCK_D) -> (R, BLOCK_D)
z_B_tile *= eps
```
So z_B = eps * z_coeff_B @ V^T. Since z_coeff_B is sampled in the r_calib-dimensional subspace and projected back to d_in via V^T, this confines perturbations to the activation subspace. Correct.

### Variance-weighted sampling for z_B

**Evidence:** `controller.py` lines 202-205:
```python
var_B_proj = layer.variance_B @ (V_l ** 2)
z_coeff_B = torch.randn(...) * torch.sqrt(var_B_proj)
```
This projects the diagonal Kalman variance through V^2 to get per-coefficient variance in the subspace, then scales the random perturbation by sqrt(variance). This means dimensions with higher posterior uncertainty are perturbed more aggressively. Mathematically sound.

### Column-space projection for z_A

**Evidence:** The kernel computes `BV = B @ V` (lines 384-390), then `Q = QR(BV)` (lines 394-405), then `z_A = eps * z_coeff_A @ Q^T` (lines 421-430).

This projects z_A through the column space of B@V, which is the image of B restricted to the activation subspace. This ensures A perturbations align with the column space that B occupies in the activation-relevant directions. Correct.

### CONCERN: z_coeff_A is not variance-weighted

**Evidence:** `controller.py` lines 207-209:
```python
z_coeff_A = torch.randn(
    A.shape[0], V_l.shape[1], device="cuda", generator=self.rng,
)
```
Unlike z_coeff_B, z_coeff_A uses raw standard normal without variance weighting. This is an asymmetry. The Kalman filter maintains variance_A, but it is not used for A perturbation generation.

**Assessment:** This may be intentional -- the column-space projection through QR(B@V) already constrains the A perturbation direction, and variance weighting on top might over-constrain. However, this asymmetry is not documented and could be a missed optimization. Marking as CONCERN rather than BUG since the algorithm still functions correctly; it just may not be optimally efficient.

### CONCERN: z_coeff_A dimension mismatch with column space

**Evidence:** `controller.py` line 208: `z_coeff_A` has shape `(A.shape[0], V_l.shape[1])` = `(d_out, r_calib)`. But Q from QR(BV) has shape `(R, r_calib)` where R is the adapter rank. So z_A = z_coeff_A @ Q^T has shape `(d_out, R)`, which matches A's shape `(d_out, R)`.

Wait -- the kernel Phase 3 (line 421): `z_A_tile = zca_tile @ Q.T = (BLOCK_D, RC) @ (RC, R) = (BLOCK_D, R)`. Here RC = r_calib and R = adapter rank. The final z_A is (d_out, R). This is dimensionally correct since A is (d_out, R).

However, z_coeff_A has r_calib columns, and Q has r_calib columns. If r_calib < R (adapter rank), then Q^T is (r_calib, R) and the projection spans at most r_calib directions in R-dimensional space. This means z_A is rank-deficient (at most rank r_calib). This could be limiting if r_calib < R. Typically r_calib should be >= R for this to be well-conditioned. The code doesn't enforce this constraint.

**Assessment:** If r_calib < R (adapter rank), z_A perturbations would be confined to a subspace of A's column space, potentially missing important directions. The Gavish-Donoho calibration typically gives r_calib much larger than the adapter rank R (which is often 16), so in practice this is likely fine, but there's no explicit guard.

---

## 3. Newton-Schulz Orthogonalization (ZO-Muon)

**Status: CORRECT (with one minor note)**

### N-S iteration formula

**Evidence - Tall kernel** (`kernels.py` lines 88-111):
```python
XtX = ... tl.dot(tl.trans(X_tile), X_tile) ...
S = _NS_C1[ns_iter] * I_N + _NS_C3[ns_iter] * XtX
X_new = tl.dot(X_tile, S)
```
For tall matrices (M >= N), the iteration is X_{k+1} = X_k @ (c1*I + c3*X_k^T@X_k). This is the correct form: the inner product G = X^T@X is (N,N), and the update multiplies X from the right. Correct.

**Evidence - Wide kernel** (`kernels.py` lines 163-186):
```python
XXt = ... tl.dot(X_tile, tl.trans(X_tile)) ...
S = _NS_C1[ns_iter] * I_M + _NS_C3[ns_iter] * XXt
X_new = tl.dot(S, X_tile)
```
For wide matrices (M < N), the iteration is X_{k+1} = (c1*I + c3*X_k@X_k^T) @ X_k. The inner product G = X@X^T is (M,M), and the update multiplies X from the left. Correct.

### Minimax-optimal coefficient derivation

**Evidence:** `kernels.py` lines 18-32 (`_ns_coefficients()`):
```python
s = u * u + l * u + l * l
alpha = (3.0 / s) ** 0.5
alpha3 = alpha * alpha * alpha
beta = 4.0 / (2.0 + l * u * (l + u) * alpha3)
c1 = 1.5 * beta * alpha
c3 = -0.5 * beta * alpha3
l = c1 * l + c3 * l * l * l
u = 2.0 - l
```

This implements the Equioscillation Theorem closed-form from Amsel et al. 2025 (arXiv:2505.16932). The polynomial p(x) = c1*x + c3*x^3 is the unique minimax-optimal degree-3 odd polynomial on [l, u]. The recurrence updates the interval bounds: the new lower bound is p(l_old) (since p is monotone increasing on the relevant interval), and the new upper bound is 2 - l_new (since the target function is 1/x which maps [l, u] to [1/u, 1/l], and we're tracking normalized singular values mapped toward 1).

Actually, the `u = 2.0 - l` update deserves scrutiny. After applying p(x), the new singular value range is [p(l), p(u)]. For the minimax polynomial, p(u) should be close to 1 (the target for orthogonalization). The claim u_new = 2 - l_new suggests symmetry around 1. Let me verify: by the equioscillation property of the minimax polynomial, |p(x) - sign(x)| achieves its maximum at exactly 3 points on [l, u]. The maximum deviation epsilon satisfies p(l) = 1 - epsilon and p(u) = 1 + epsilon (or close to it). Then u_new = p(u) and l_new = p(l), and u_new + l_new = 2 approximately. The code uses the exact relation u = 2 - l. This is correct for the degree-3 equioscillation polynomial where the error symmetry gives p(l) + p(u) = 2 exactly.

### Convergence criterion

**Evidence:** `kernels.py` line 22:
```python
while 1.0 - l >= eps:
```
This iterates until the lower bound l is within machine epsilon of 1.0, meaning all singular values have converged to 1.0 (orthogonalization complete). The comment on line 38 says this produces 12 iterations from l=sqrt(eps_f32). Correct.

### Frobenius normalization

**Evidence:** `kernels.py` lines 64-85 (tall) and 140-161 (wide):
```python
norm_sq += tl.sum(buf_tile * buf_tile)
inv_norm = 1.0 / tl.sqrt(tl.maximum(norm_sq, NORM_FLOOR))
tl.store(scratch_ptr + ptrs, buf_tile * inv_norm, mask=mask)
```
The buffer (Kalman posterior mean) is normalized to unit Frobenius norm before N-S iterations. This maps all singular values into [0, 1], which is the domain of the minimax polynomial. Correct.

### Tall/wide dispatch

**Evidence:** `kernels.py` lines 200-218:
```python
if M >= N:
    _zo_muon_tall_kernel[...]  # G = X^T@X is (N,N) — cheaper when N < M
else:
    _zo_muon_wide_kernel[...]  # G = X@X^T is (M,M) — cheaper when M < N
```
Correct: always forms the smaller Gram matrix.

### Minor note: Single program launch

Both kernels are launched with grid `(1,)` — a single Triton program. This means the entire N-S computation runs on one SM. For small adapter matrices (e.g., 16x4096), this is fine and avoids inter-SM synchronization. For larger matrices, this could be a throughput bottleneck, but the comment on lines 38-39 says the total N-S time is <0.2% of step time, so this is not a concern.

### Note on BLOCK sizes as `tl.constexpr`

N is `tl.constexpr` in the tall kernel and M is `tl.constexpr` in the wide kernel. This means Triton will JIT-compile a specialized kernel for each unique matrix dimension. For the N-S Gram matrix (which lives entirely in registers), the constexpr dimension must fit in registers. For rank-16 adapters, N=16 or M=16, which is fine. For larger ranks, this could hit register pressure limits, but this is an infrastructure constraint, not a correctness issue.

---

## 4. Diagonal Kalman Filter

**Status: BUG (minor) + CONCERN**

### Prediction step

**Evidence:** `controller.py` lines 283-286:
```python
mu.mul_(beta)
var.mul_(beta * beta).add_(q)
```
This implements mu_pred = beta * mu, sigma^2_pred = beta^2 * sigma^2 + q. This is the standard Kalman prediction for a random walk with drift coefficient beta and process noise q. Correct.

### Process noise q

**Evidence:** `controller.py` line 276:
```python
q = self.eps ** 2 * (1.0 - beta * beta)
```
This sets q so that the steady-state variance is eps^2: at equilibrium, beta^2 * sigma^2 + q = sigma^2 => q = sigma^2 * (1 - beta^2) => sigma^2 = q / (1 - beta^2) = eps^2. Correct.

### Observation step

**Evidence:** `controller.py` lines 279-303:

**Pass 1 (global reductions):**
```python
y_hat += (z_mat * mu).sum()       # y_hat = sum_all(z * mu)
s_sum += (z_mat * z_mat * var).sum()  # S = sum_all(z^2 * sigma^2)
```

**Between passes:**
```python
S = s_sum.item() + self.norm_floor
innovation = dd - y_hat.item()
inv_S = 1.0 / S
```

**Pass 2 (local updates):**
```python
K = var * z_mat * inv_S       # K_i = sigma_i^2 * z_i / S
mu.add_(innovation * K)       # mu_i += (dd - y_hat) * K_i
var.mul_(1.0 - K * z_mat)    # sigma_i^2 *= (1 - K_i * z_i)
```

Let me verify the math:
- y_hat = sum(z_i * mu_i) across all parameters: this is the predicted observation (inner product of perturbation with mean). Correct.
- S = sum(z_i^2 * sigma_i^2) + floor: this is the observation noise variance. In standard Kalman, S = H @ P @ H^T + R. Here H = z (the perturbation vector), P = diag(sigma^2), so H@P@H^T = sum(z_i^2 * sigma_i^2). The floor prevents division by zero. Correct.
- innovation = dd - y_hat: the difference between actual and predicted observation. Correct.
- K_i = sigma_i^2 * z_i / S: the Kalman gain for diagonal P. Standard formula K = P @ H^T @ S^{-1} = sigma_i^2 * z_i / S. Correct.
- mu_i += innovation * K_i: posterior mean update. Correct.
- sigma_i^2 *= (1 - K_i * z_i): This simplifies to sigma_i^2 * (1 - sigma_i^2 * z_i^2 / S). Standard Joseph form for diagonal case: P_post = (I - K@H) @ P = P - K@H@P. For element i: sigma_i^2_post = sigma_i^2 - K_i * z_i * sigma_i^2 = sigma_i^2 * (1 - K_i * z_i). Correct.

### BUG: Variance can go negative

**Evidence:** `controller.py` line 303:
```python
var.mul_(1.0 - K * z_mat)
```
Since K = var * z_mat / S, we have K * z_mat = var * z_mat^2 / S. The term (1 - K * z_mat) = 1 - var * z_mat^2 / S.

In exact arithmetic, S = sum(z_j^2 * var_j) >= z_i^2 * var_i for any single element i, so 0 <= var_i * z_i^2 / S <= 1, and the factor is in [0, 1]. However, in floating-point arithmetic with millions of parameters, the cumulative sum S is computed via sequential `.sum()` calls across multiple layers (lines 287-288). Floating-point roundoff could cause the factor to go slightly negative for some elements, leading to negative variance.

**Severity:** Low. The variance is initialized to eps^2 (very small), and the process noise q adds positive values each step, so any negative variance would be corrected next step. But a negative variance could cause sqrt(var) in `_get_perturbation` to produce NaN on the very next step if the negative element survives.

**Recommended fix:** Clamp: `var.clamp_(min=self.norm_floor)` after the update. Or use the numerically stable Joseph form: `var.mul_((1.0 - K * z_mat).clamp(min=0.0))`.

### CONCERN: Two-pass global reduction ordering

The code iterates over layers twice (pass 1 for reductions, pass 2 for updates). This is correct -- the innovation and S must be computed globally before any local update. If updates were interleaved with reductions, the result would be incorrect. The two-pass structure handles this correctly.

### CONCERN: Momentum schedule

**Evidence:** `controller.py` lines 273-274:
```python
max_window = int(math.sqrt(self.total_steps))
self._momentum = 1.0 - 1.0 / min(self.step_count, max_window)
```
At step 1: momentum = 1 - 1/1 = 0 (no momentum).
At step 2: momentum = 1 - 1/2 = 0.5.
At step sqrt(T): momentum = 1 - 1/sqrt(T), which for T=1000 gives ~0.968.

This saturates at 1 - 1/sqrt(T). The MEMORY.md confirms this is a heuristic. Note that at step 0, `min(0, max_window) = 0` which would cause division by zero. However, `step_count` is incremented to 1 at the beginning of `step()` (line 313), so `_update_weights` is always called with step_count >= 1. Safe.

---

## 5. RLOO with James-Stein Shrinkage

**Status: CORRECT**

### RLOO baselines

**Evidence:** `controller.py` lines 235-238:
```python
rewards = torch.tensor([r for _, r in scored])
N = len(scored)
r_bar = float(rewards.mean())
baselines_rloo = (rewards.sum() - rewards) / (N - 1)
```
RLOO baseline for candidate i: b_i = (sum_j r_j - r_i) / (N-1) = (total - r_i) / (N-1). This is the leave-one-out mean. Correct.

### James-Stein shrinkage

**Evidence:** `controller.py` lines 239-241:
```python
reward_var = float(rewards.var())
lam = reward_var / (reward_var + (r_bar - self.reward_ema) ** 2 + self.norm_floor)
baselines = (1.0 - lam) * baselines_rloo + lam * self.reward_ema
```

The James-Stein shrinkage factor lambda = var(r) / (var(r) + (r_bar - EMA)^2 + floor).

When var(r) is large relative to (r_bar - EMA)^2: lambda -> 1, so baseline -> EMA (shrink toward prior).
When var(r) is small relative to (r_bar - EMA)^2: lambda -> 0, so baseline -> RLOO (trust the data).

This is the standard James-Stein positive-part shrinkage estimator (without the positive-part truncation, but the floor in the denominator prevents division by zero and ensures lambda in [0, 1)). Correct.

The `norm_floor` prevents division by zero when both var and (r_bar - EMA)^2 are zero.

### Baseline blending

**Evidence:** Line 241:
```python
baselines = (1.0 - lam) * baselines_rloo + lam * self.reward_ema
```
Convex combination of RLOO and EMA. Correct.

### Reward EMA update

**Evidence:** `controller.py` lines 244-245:
```python
beta = self._momentum
self.reward_ema = beta * self.reward_ema + (1.0 - beta) * r_bar
```
Standard exponential moving average with the same momentum used by the Kalman filter. Correct.

### Note: Advantage normalization

The advantages (line 242) are `rewards - baselines` but are NOT divided by std(rewards). The MEMORY.md mentions "REINFORCE++ normalization" with `torch.std()` but the current code does not apply it. This is not a bug per se -- unnormalized advantages scale the SPSA loss, and the Kalman filter + N-S normalization should be robust to scale. But it differs from what the MEMORY.md describes.

**Assessment:** The MEMORY.md entry about "Vectorized with torch.tensor ops + torch.std()" may refer to an earlier version. The current implementation is internally consistent and correct.

---

## 6. Cosine LR Schedule

**Status: CONCERN**

### PyTorch CosineAnnealingLR usage

**Evidence:** `controller.py` lines 100-104:
```python
self._lr_param = torch.nn.Parameter(torch.empty(0, device="cuda"))
self._lr_opt = torch.optim.SGD([self._lr_param], lr=self.eta_max)
self.lr_scheduler = CosineAnnealingLR(
    self._lr_opt, T_max=self.total_steps, eta_min=0.0,
)
```
This creates a dummy optimizer to use PyTorch's CosineAnnealingLR scheduler. The LR is read back via:
```python
self.eta = self._lr_opt.param_groups[0]["lr"]
```
This is a standard pattern for using PyTorch LR schedulers without an actual optimizer. Correct.

### CONCERN: eta_min = 0.0

**Evidence:** Line 103: `eta_min=0.0`

The MEMORY.md states: "eta_min = eta_max/100". But the code uses `eta_min=0.0`. This means the learning rate decays all the way to zero at the end of training, rather than maintaining a small floor.

This is a **discrepancy with documentation**. With eta_min=0.0, the final steps of training have vanishingly small learning rates, which means the model effectively stops learning. Whether this is desirable depends on the use case, but it contradicts the documented design.

### eta_max = eps relationship

**Evidence:** `controller.py` lines 91-98:
```python
norms = torch.stack([torch.linalg.norm(l.A) for l in self.layers] +
                     [torch.linalg.norm(l.B) for l in self.layers])
self.eps = float(torch.median(norms)) * float(eps_machine) ** (1.0 / 3.0)
self.eta_max = self.eps
```
The trust-region argument: step size (eta) should not exceed perturbation radius (eps). Setting eta_max = eps satisfies this. The eps derivation follows Numerical Recipes 5.7 for finite-difference step sizing: eps ~ ||W|| * eps_machine^{1/3}. Correct.

---

## 7. PiSSA Adapter Preparation

**Status: CORRECT**

### PiSSA decomposition via PEFT

**Evidence:** `prepare_pissa.py` lines 39-47:
```python
pissa_config = LoraConfig(
    r=rank,
    lora_alpha=rank,
    target_modules=target_modules,
    init_lora_weights=f"pissa_niter_{_SVD_NITER}",
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, pissa_config)
```
The `init_lora_weights="pissa_niter_K"` tells PEFT to initialize LoRA weights using PiSSA (Principal Singular Values and Singular vectors Adaptation) with K power iterations in the SVD. Correct.

### A/B swap convention

**Evidence:** `controller.py` lines 77-79:
```python
# PEFT -> PiSSA: lora_A = B, lora_B = A
B = adapter_tensors[f"{ls.peft_prefix}.lora_A.weight"].float()
A = adapter_tensors[f"{ls.peft_prefix}.lora_B.weight"].float()
```
And `backend.py` lines 22-34:
```python
# lora_A.weight = B, lora_B.weight = A (PiSSA/PEFT swap)
tensors[key_a] = bf16_cache[key_a]  # stores B_l as lora_A.weight
tensors[key_b] = bf16_cache[key_b]  # stores A_l as lora_B.weight
```

In PiSSA, the decomposition is W = W_res + A @ B where A captures top singular vectors. In PEFT's LoRA convention, the adapter computes delta_W = lora_B @ lora_A. So to match: lora_B = A and lora_A = B. The code loads lora_A.weight as B and lora_B.weight as A, which is correct.

When saving back, B_l is written as lora_A.weight and A_l as lora_B.weight. Consistent.

**Verification:** The shapes should be:
- B (stored as lora_A): shape (r, d_in) -- r is the low rank
- A (stored as lora_B): shape (d_out, r)
- delta_W = A @ B = (d_out, r) @ (r, d_in) = (d_out, d_in). Correct.

### SVD power iteration count

**Evidence:** `model_config.py` lines 47-49:
```python
def svd_power_iters(dtype=torch.float32):
    eps = float(torch.finfo(dtype).eps)
    return math.ceil(math.log(math.log(1.0 / eps) / math.log(2.0)) / math.log(3.0))
```
This computes ceil(log_3(log_2(1/eps))). For float32, eps ~ 1.19e-7:
- 1/eps ~ 8.39e6
- log_2(8.39e6) ~ 23.0
- log_3(23.0) ~ 2.85
- ceil(2.85) = 3

So K=3 power iterations. The MEMORY.md confirms "Power iteration: Linear convergence... Formula gives k=3 empirically." Used in both `prepare_pissa.py` (PiSSA decomposition) and `controller.py` (activation basis update). Correct.

---

## 8. Activation Basis Calibration & Update

**Status: CORRECT**

### Gavish-Donoho optimal rank via optht

**Evidence:** `controller.py` lines 168-172:
```python
sv = torch.linalg.svdvals(H).cpu().numpy()
m, n = H.shape
beta = min(m, n) / max(m, n)
ranks_per_activation.append(optht(beta, sv=sv, sigma=None))
```
The `optht` function implements the Gavish-Donoho optimal hard threshold for singular values based on the Marchenko-Pastur distribution. The `beta` parameter is the aspect ratio min(m,n)/max(m,n). Passing `sigma=None` with `sv=sv` tells optht to estimate the noise level from the singular values themselves. Correct.

The median rank across all activations is used (line 173):
```python
self.r_calib = int(np.median(ranks_per_activation))
```
Taking the median is robust to outlier layers. Correct.

### Initial SVD via torch.svd_lowrank

**Evidence:** `controller.py` lines 178-181:
```python
_, _, V = torch.svd_lowrank(H, q=self.r_calib, niter=self.power_iter_steps)
```
`torch.svd_lowrank` computes a randomized low-rank SVD. Only V (right singular vectors) is retained, which spans the top r_calib directions in d_in space. These are the activation basis vectors. Correct.

### Incremental update via fused power iteration

**Evidence:** `controller.py` lines 184-195 and `kernels.py` lines 226-310.

The `_update_activation_bases` method calls `fused_power_iter(H, V, num_iters, norm_floor)` which runs the Triton kernel `_power_iter_kernel`.

The kernel (lines 256-285):
1. Computes HtHV = H^T @ (H @ V) via tiled matmul (lines 258-270)
2. Applies Modified Gram-Schmidt QR to HtHV (lines 272-285)
3. Repeats for num_iters iterations

This is subspace iteration (also called simultaneous power iteration): V_{k+1} = QR(H^T @ H @ V_k). This converges the columns of V to the top r_calib right singular vectors of H. Correct.

### Activation deduplication for merged modules

**Evidence:** `controller.py` lines 155-160 and 163-172:
```python
for layer in self.layers:
    act = activations[layer.key]
    if id(act) not in gpu_acts:
        gpu_acts[id(act)] = act.cuda(non_blocking=True)
```
And in `backend.py` lines 68-72:
```python
def hook_fn(mod, inp, out, ks=keys):
    act = inp[0].detach().float()
    for k in ks:
        worker._ds_mezo_activations[k] = act
```
For merged modules (e.g., qkv_proj), the same activation tensor is assigned to all sub-keys (q_proj, k_proj, v_proj). The `id(act)` deduplication ensures the SVD is computed only once per unique activation tensor, then shared. Correct.

**Evidence in `_update_activation_bases`** (`controller.py` lines 184-195): Same deduplication pattern with `processed[act_id]`. Correct.

### Note on hook_map construction

**Evidence:** `backend.py` lines 109-113:
```python
for ls in layer_specs:
    vllm_name = _VLLM_MERGES.get(ls.module_name, ls.module_name)
    hook_map.setdefault(vllm_name, set()).add(ls.module_name)
```
The `_VLLM_MERGES` dict maps q_proj/k_proj/v_proj to qkv_proj, and gate_proj/up_proj to gate_up_proj. So a single hook on qkv_proj captures activations for all three sub-modules. Correct.

---

## 9. Triton Kernel Correctness

### Kernel 1: ZO-Muon (Tall)

**Status: CORRECT**

**Index arithmetic:** Lines 69-71: `offs_m = m_start + tl.arange(0, BLOCK_M)`, `offs_n = tl.arange(0, N)`. Pointer computation: `ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n`. Standard 2D tiling over M with full N in registers (since N is constexpr). Correct.

**Masking:** `mask = offs_m[:, None] < M` ensures only valid rows are loaded. The N dimension is fully covered (constexpr). Correct.

**XtX accumulation:** Lines 91-98: Accumulates X^T@X across chunks of M. The transpose `tl.trans(X_tile)` is (N, BLOCK_M), dotted with X_tile (BLOCK_M, N) to give (N, N). Correct.

**Identity construction:** Lines 100-101: `I_N = (eye_n[:, None] == eye_n[None, :]).to(tl.float32)`. This creates an N x N identity matrix in registers. Correct.

**allow_tf32=False:** All matmuls use full FP32 precision. Important for N-S convergence accuracy. Correct.

### Kernel 1: ZO-Muon (Wide)

**Status: CORRECT**

Mirror of tall kernel with M/N swapped. Tiles over N columns, M is constexpr. XXt = X@X^T is (M, M). S multiplied from the left. Verified symmetric to the tall kernel. Correct.

### Kernel 2: Fused Power Iteration

**Status: CORRECT (with note)**

**D and R padding:** Lines 302: `D=triton.next_power_of_2(D), R=triton.next_power_of_2(R)`. Actual dimensions masked with `d_mask` and `r_mask`. Correct.

**HtHV computation:** Lines 258-270: Two-step matmul: first H_tile @ V = (BLOCK_T, D) @ (D, R) = (BLOCK_T, R), then H_tile^T @ result = (D, BLOCK_T) @ (BLOCK_T, R) = (D, R). Accumulated across T chunks. This is mathematically equivalent to (H^T @ H) @ V but avoids forming the (D, D) matrix. Correct and memory-efficient.

**Modified Gram-Schmidt QR:** Lines 274-285: Mask-based column extraction pattern (since Triton disallows dynamic indexing). For each column: extract, orthogonalize against all previous columns, normalize, write back. This is standard MGS. The `NORM_FLOOR` prevents division by zero for zero columns.

**Note:** The R dimension is padded to next power of 2. If R_actual is, say, 8 and R is 8, there's no waste. But if R_actual is 9, R becomes 16 and the extra 7 columns go through MGS unnecessarily (though they're zero-initialized and masked on store). The computational waste is O(R * (R_pad - R_actual)) per QR, which is negligible for small ranks.

### Kernel 3: Fused AGZO Perturbation

**Status: CORRECT**

**Phase 1 - z_B and BV computation:** Lines 355-390. Shared tiling over d_in. For each chunk:
- z_B_tile = z_coeff_B @ V_tile^T via rank-1 accumulation over RC (lines 369-374). Since RC is small (typically 8-32), this avoids tl.dot's minimum dimension requirements.
- BV += B_tile @ V_tile via tl.dot (line 390).

Correct fused computation sharing the V_tile load.

**Phase 2 - QR on BV:** Lines 394-405. Same Modified Gram-Schmidt pattern as in the power iteration kernel. BV is (R, RC) which is small. Correct.

**Phase 3 - z_A computation:** Lines 410-436. z_A_tile = z_coeff_A_tile @ Q^T via rank-1 accumulation over RC. Output z_A is (d_out, R). Correct.

**Contiguity assertion:** `kernels.py` lines 440-441: Asserts z_coeff tensors are C-contiguous because the kernel uses hardcoded stride=RC for them. Correct safety check.

### Kernel 4: Fused Dual Perturbation

**Status: CORRECT**

**Evidence:** Lines 472-494. Standard elementwise kernel: pos = base + z, neg = base - z. Grid is ceil(N / BLOCK). Masking: `offs < N`. Trivially correct.

### Numerical safety across all kernels

- **Division by zero:** Protected by NORM_FLOOR in all norm computations (Frobenius norm, QR column norms). NORM_FLOOR is `torch.finfo(torch.float32).tiny` ~ 1.18e-38. Adequate.
- **Overflow:** No overflow risk -- all values are FP32 adapter weights (bounded) or normalized values.
- **tl.constexpr usage:** N in tall kernel, M in wide kernel, R/RC/D in AGZO kernel. These are all adapter dimensions that vary across runs but are fixed per JIT compilation. Correct use of constexpr for register allocation.

---

## 10. Checkpoint Save/Load

**Status: CONCERN**

### State completeness

**Save** (`controller.py` lines 344-376):
- Master weights: A, B for all layers
- Optimizer state: momentum_A, momentum_B, variance_A, variance_B for all layers
- Activation bases: activation_bases for all layers
- RNG state: `self.rng.get_state()`
- Training state JSON: step, eta, reward_ema, lr_scheduler state dict

**Load** (`controller.py` lines 129-150):
- Restores all of the above

### CONCERN: Missing state in checkpoint

The following state is NOT saved/restored:
1. **`self._momentum`**: The momentum coefficient. It is recomputed from `self.step_count` in `_update_weights`, so this is fine (it's derived state).
2. **`self._bf16_ckpt`**: The BF16 staging buffer cache. Not saved, but this is just a pre-allocation cache and will be lazily recreated. Fine.
3. **`self.eps`**: The perturbation scale. This is recomputed from adapter norms at initialization. But after training, the adapter norms will have changed, so `eps` will differ between the original run and a resumed run. This could cause a discontinuity.

**Assessment of eps concern:** When loading a checkpoint, `__init__` computes `self.eps` from the freshly-loaded adapter tensors. But then `_load_checkpoint` overwrites A and B with the checkpoint values. The eps was computed from the INITIAL adapter values (before checkpoint restore), not the checkpoint values.

**Evidence:** `__init__` line 70-95 loads the original adapter and computes eps. Then line 126-127: `if cfg["resume_from"]: self._load_checkpoint(...)` overwrites A and B. So eps is based on the original (pre-training) adapter values, not the checkpoint's current values. This is actually **consistent** behavior -- eps is always computed from the initial adapter, regardless of training progress. Whether this is intentional is debatable; the design doc says eps should be based on adapter norms, and using the initial norms provides a stable reference. Marking as CONCERN rather than BUG.

Actually, wait. Let me re-read. `__init__` loads `adapter_tensors` from `cfg["adapter_path"]` (line 70-73). This is the PiSSA adapter path, not a checkpoint path. So eps is always based on the initial PiSSA decomposition norms. During resume, A/B are then overwritten with checkpoint values. This means eps is deterministic regardless of resume point. This is likely intentional -- the perturbation scale should be consistent throughout training. Correct.

### CONCERN: eta not restored before lr_scheduler

**Evidence:** `_load_checkpoint` line 147-149:
```python
self.step_count = state["step"]
self.eta = state["eta"]
...
self.lr_scheduler.load_state_dict(state["lr_scheduler"])
```
The `eta` is set from the JSON state, and the lr_scheduler is restored from its state dict. On the next `_step_lr()` call, the scheduler will step and set eta from the optimizer's param_groups. The scheduler state dict should correctly restore the last_epoch counter, so subsequent steps should produce correct LR values. Correct.

### RNG state preservation

**Evidence:** Save: `tensors["rng_state"] = self.rng.get_state()` (line 366).
Load: `self.rng.set_state(tensors["rng_state"])` (line 142).

The CUDA generator state is saved as a tensor in the safetensors file. This ensures bitwise-identical perturbation sequences upon resume. Correct.

### Adapter format compatibility with PEFT

**Evidence:** `backend.py` lines 10-18 (`write_adapter_config`) and 21-36 (`save_peft_adapter`).

The adapter is saved as:
1. `adapter_model.safetensors` with PEFT-convention key names (`base_model.model.*.lora_A.weight`, etc.)
2. `adapter_config.json` via `LoraConfig.save_pretrained()`

This is the standard PEFT LoRA adapter format. Any PEFT-compatible tool can load these adapters. Correct.

---

## 11. vLLM Backend

**Status: CORRECT (with one note)**

### Activation hook registration

**Evidence:** `backend.py` lines 55-75:
```python
for name, module in model.named_modules():
    suffix = name.rsplit(".", 1)[-1]
    if suffix not in hook_map:
        continue
    ...
    def hook_fn(mod, inp, out, ks=keys):
        act = inp[0].detach().float()
        for k in ks:
            worker._ds_mezo_activations[k] = act
    worker._ds_mezo_hooks.append(module.register_forward_hook(hook_fn))
```

Hooks capture the input activation (not output) in float32. The `ks=keys` default argument captures the closure correctly (avoids the Python closure-in-loop bug). Hooks are registered via `collective_rpc` which executes on the vLLM worker. Correct.

### Activation collection and cleanup

**Evidence:** `backend.py` lines 78-93:
```python
seen = {}
for k, v in worker._ds_mezo_activations.items():
    tid = id(v)
    if tid not in seen:
        cpu_t = torch.empty(v.shape, dtype=v.dtype, pin_memory=True)
        cpu_t.copy_(v, non_blocking=True)
        seen[tid] = cpu_t
    activations[k] = seen[tid]
torch.cuda.current_stream().synchronize()
for h in worker._ds_mezo_hooks:
    h.remove()
```

Uses pinned memory + non_blocking copy for GPU->CPU transfer. Deduplicates by tensor id (same pattern as activation basis calibration). Synchronizes before returning to ensure copies are complete. Hooks are properly removed to avoid interfering with subsequent vLLM calls. Correct.

### Log-prob extraction

**Evidence:** `backend.py` lines 96-101:
```python
def _extract_prompt_logprobs(output, prompt_token_ids):
    logprobs = []
    for i, token_lp in enumerate(output.prompt_logprobs[1:], 1):
        tok_id = prompt_token_ids[i]
        logprobs.append(token_lp[tok_id].logprob)
    return logprobs
```

`prompt_logprobs[0]` is None (no logprob for the first token), so the slice `[1:]` starts from the second entry. `token_lp` is a dict mapping token_id -> LogProb. The code looks up by the actual token_id at position i. This fixes the v1 bug mentioned in MEMORY.md ("now looks up by token ID, not first dict entry"). Correct.

### Adapter sync path

**Evidence:** `backend.py` lines 136-156:
```python
def sync_adapters(self, pos_overrides, neg_overrides, layers):
    ...
    save_peft_adapter(A_pos, B_pos, self.adapter_dir_pos, layers, self._bf16_pos)
    save_peft_adapter(A_neg, B_neg, self.adapter_dir_neg, layers, self._bf16_neg)
    self.engine.llm_engine.add_lora(self.lora_pos)
    self.engine.llm_engine.add_lora(self.lora_neg)
```

The BF16 staging buffers (`self._bf16_pos`, `self._bf16_neg`) are pre-allocated and reused across calls (lazy init in `save_peft_adapter`). This avoids allocation overhead on the hot path. The adapters are serialized to safetensors on `/dev/shm` (tmpfs) for fast I/O, then loaded by vLLM via `add_lora` with `load_inplace=True`. Correct.

### Note: LoRARequest with load_inplace

**Evidence:** `backend.py` lines 125-126:
```python
self.lora_pos = LoRARequest("adapter_pos", 1, str(self.adapter_dir_pos), load_inplace=True)
self.lora_neg = LoRARequest("adapter_neg", 2, str(self.adapter_dir_neg), load_inplace=True)
```
`load_inplace=True` tells vLLM to reload the adapter weights from disk on each `add_lora` call, even if the adapter ID is already loaded. This is essential since the safetensors files are being updated each step. Without this flag, vLLM would cache the first version. Correct.

### Note: score() method

**Evidence:** `backend.py` lines 165-174:
```python
def score(self, token_sequences, lora_request):
    prompts = [{"prompt_token_ids": seq} for seq in token_sequences]
    outputs = self.engine.generate(
        prompts, sampling_params=self.score_params, lora_request=lora_request,
    )
    return [
        _extract_prompt_logprobs(out, seq)
        for out, seq in zip(outputs, token_sequences)
    ]
```

Scoring is done by passing full token sequences (prompt + response) as "prompts" with `max_tokens=1` and `prompt_logprobs=1`. This makes vLLM compute logprobs for each token in the sequence conditioned on preceding tokens, without actually generating new tokens. The `max_tokens=1` generates one dummy token that is discarded. This is the standard vLLM prompt-scoring pattern. Correct.

---

## Summary Table

| Component | Status | Key Finding |
|-----------|--------|-------------|
| 1. SPSA Gradient Estimation | CORRECT | Standard two-point SPSA with advantage-weighted NLL |
| 2. AGZO Perturbation | CONCERN | z_A not variance-weighted (asymmetry with z_B); potential rank deficiency if r_calib < adapter rank |
| 3. Newton-Schulz (ZO-Muon) | CORRECT | Faithful implementation of Polar Express minimax-optimal N-S |
| 4. Diagonal Kalman Filter | BUG (minor) | Variance can go slightly negative due to FP roundoff; recommend clamping |
| 5. RLOO + James-Stein | CORRECT | Standard shrinkage estimator; advantages not std-normalized (differs from MEMORY.md) |
| 6. Cosine LR | CONCERN | eta_min=0.0 contradicts MEMORY.md claim of eta_min=eta_max/100 |
| 7. PiSSA Preparation | CORRECT | Proper PEFT integration with correct A/B swap |
| 8. Activation Basis | CORRECT | Gavish-Donoho rank + power iteration update with deduplication |
| 9. Triton Kernels | CORRECT | All index arithmetic, masking, and tiling verified; MGS QR correct |
| 10. Checkpoint | CONCERN | eps computed from initial adapter (intentional but undocumented) |
| 11. vLLM Backend | CORRECT | Proper hook lifecycle, log-prob extraction by token ID, BF16 staging |

## Critical Findings

### BUG: Kalman variance negativity (Severity: LOW)
- **File:** `controller.py` line 303
- **Issue:** `var.mul_(1.0 - K * z_mat)` can produce negative variance due to FP32 roundoff in the global sum S
- **Fix:** Add `var.clamp_(min=self.norm_floor)` after line 303

### CONCERN: eta_min documentation mismatch (Severity: MEDIUM)
- **File:** `controller.py` line 103
- **Issue:** Code uses `eta_min=0.0`, MEMORY.md says `eta_min=eta_max/100`
- **Impact:** Learning rate decays to zero at end of training instead of maintaining a floor

### CONCERN: z_A perturbation not variance-weighted (Severity: LOW)
- **File:** `controller.py` lines 207-209
- **Issue:** z_coeff_A uses raw standard normal while z_coeff_B uses variance-weighted sampling
- **Impact:** A perturbation efficiency may be suboptimal, though column-space projection provides alternative constraint

### CONCERN: No guard on r_calib vs adapter rank (Severity: LOW)
- **File:** `controller.py` line 173, `kernels.py` AGZO kernel
- **Issue:** If Gavish-Donoho gives r_calib < adapter rank R, z_A perturbations are rank-deficient
- **Impact:** Unlikely in practice (r_calib >> R typically) but no explicit assertion
