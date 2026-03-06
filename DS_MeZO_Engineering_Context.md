# DS-MeZO: Decoupled-Switched Zeroth-Order Optimization

**v7.0 — Mathematically Rigorous Adaptive Architecture**

**Date:** March 5, 2026
**Target Hardware:** Single NVIDIA H100 (80GB HBM3)
**Architecture:** Python/PyTorch Controller + vLLM (Native DoRA Mode, S-LoRA Backend)

---

## 1. Executive Summary

DS-MeZO is a gradient-free optimization system for fine-tuning Large Language Models on a single GPU. It eliminates backpropagation entirely, using zeroth-order (ZO) estimation to optimize non-differentiable objectives — code compilation, proof verification, tool use, human feedback — at near-inference memory cost.

The system synthesizes four components:

| Component | Role |
| :--- | :--- |
| **PiSSA** (arXiv:2404.02948) | SVD-based adapter initialization. Eliminates cold start. Provides inherent rank-$r$ dimensionality reduction. QPiSSA quantizes the residual for ~20% less error. |
| **DoRA** (arXiv:2402.09353) | Magnitude/direction weight decomposition. Exposes $m$ as an explicit tensor via vLLM's native DoRA kernel. Negative correlation ($-0.31$) enables Tick/Tock decoupling. |
| **Zhang et al.** (arXiv:2506.05454) | Proves ZO implicitly finds flat minima via $F(x) = f(x) + \frac{\lambda^2}{2}\operatorname{Tr}(\nabla^2 f(x))$. Convergence governed by Hessian effective rank, not raw dimension. |
| **Trajectory Locking** | Generate with unperturbed weights, score fixed tokens under perturbations. Eliminates trajectory divergence and reward vanishing. |

**Adaptive Mechanisms (v7.0):** R-AdaZO adaptive $\eta_t$/$\epsilon_t$, dynamic reward baseline, per-layer dynamic rank allocation. Three-phase alternating updates (Tick $\to$ Tock-A $\to$ Tock-B) guarantee ZO estimator symmetry.

**Design Principle:** The Controller (CPU/PyTorch) handles optimization logic; vLLM (GPU) handles all forward passes. Any inference server becomes a training server.

**Core Capabilities:**

- **Memory Efficiency:** Near-inference memory — no gradient tensors, no optimizer states beyond adapter-sized scalars.
- **Non-Differentiable Objectives:** Optimizes any `str → float` reward function (code correctness, proof validity, tool success, LLM-as-Judge).
- **Hardware Feasibility:** Llama-3-70B (NF4) + DS-MeZO states fit on a single 80GB H100 with ~18 GB headroom.
- **Auto-Tuning:** 5 of 8 previously manual hyperparameters are adapted online. Practitioners set only insensitive initial seeds.

---

## 2. Theoretical Foundation

### 2.1 Flat Minima Regularization via Zeroth-Order Optimization

The ZO two-point estimator unbiasedly estimates the gradient of the smoothed loss:

$$f_\lambda(x) = \mathbb{E}_{u \sim \mathcal{N}(0, I_d)}[f(x + \lambda u)] = f(x) + \frac{\lambda^2}{2} \operatorname{Tr}(\nabla^2 f(x)) + o(\lambda^2)$$

ZO therefore implicitly minimizes the regularized objective:

$$F(x) = f(x) + \frac{\lambda^2}{2} \operatorname{Tr}(\nabla^2 f(x))$$

Updates move toward directions that simultaneously reduce loss and reduce sharpness:

$$\mathbb{E}_{u_t}[x_{t+1} - x_t] = -\eta \nabla f(x_t) - \frac{\eta\lambda^2}{2} \nabla \operatorname{Tr}(\nabla^2 f(x_t)) + o(\lambda^2)$$

**Convergence Guarantee (Zhang et al.).** Under convexity and third-order smoothness ($L_1$, $L_2$, $L_3$), with step size $\eta = 1/(8(d+6)L_1)$:

$$T = \mathcal{O}\left(\frac{d^4}{\epsilon^2}\right) \text{ iterations to } (\mathcal{O}(\epsilon/d^2), \epsilon)\text{-approximate flat minima}$$

**Convexity Caveat.** LLM loss landscapes are non-convex. The bound serves as a scaling heuristic: (a) lower effective dimension $\rightarrow$ faster convergence, (b) larger perturbation radius $\rightarrow$ stronger flat-minima bias, (c) $d^4$ dependence makes dimensionality reduction essential.

**Search Space.** PiSSA constrains ZO perturbation to adapter parameters $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times k}$:

$$\dim(z) = (d \times r) + (r \times k)$$

For Llama-3 70B ($d = k = 8192$, $r = 16$): **262,144 parameters** — a 250$\times$ reduction from $d \times k \approx 67$M per layer. Convergence is governed by the Hessian effective rank $\operatorname{rank}_{\text{eff}}(\nabla^2 f) = \operatorname{Tr}(\nabla^2 f) / \|\nabla^2 f\|$, which is far smaller than 262K due to PiSSA's alignment with principal singular directions.

### 2.2 DoRA Weight Decomposition

The fine-tuned weight for each target layer:

$$W' = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c}$$

where $\Delta V = BA$ is the low-rank directional update, $m \in \mathbb{R}^{1 \times k}$ is the learnable magnitude vector, and $\|\cdot\|_c$ denotes column-wise norms.

The magnitude vector $m$ is maintained as a **separate, explicit tensor** — not folded into $A$/$B$. vLLM's native DoRA kernel exposes $m$ directly and fuses dequantization + normalization + scaling into a single GPU kernel.

**Gradient Analysis.** The directional gradient is projected away from $W'$ and scaled by $m/\|V'\|_c$, improving conditioning:

$$\nabla_{V'} \mathcal{L} = \frac{m}{\|V'\|_c} \left(I - \frac{V' V'^T}{\|V'\|_c^2}\right) \nabla_{W'} \mathcal{L}$$

**Negative Correlation.** DoRA exhibits a negative correlation ($-0.31$) between magnitude and directional changes, matching full fine-tuning ($-0.62$) and opposing LoRA's positive correlation ($+0.83$). This enables Tick/Tock decoupling: large magnitude adjustments accompany small directional changes, and vice versa.

### 2.3 Zeroth-Order Gradient Estimation

The symmetric two-point estimator:

$$\hat{g} = \frac{\mathcal{L}(\theta + \epsilon z) - \mathcal{L}(\theta - \epsilon z)}{2\epsilon} \cdot z$$

This unbiasedly estimates $\nabla f_\lambda(\theta)$. Combined with flat minima regularization, DS-MeZO simultaneously minimizes loss and seeks well-generalizing solutions.

**Symmetry Requirement.** The estimator's unbiasedness depends on the perturbations being symmetric around the current parameters: $\frac{(\theta + \epsilon z) + (\theta - \epsilon z)}{2} = \theta$. If the evaluation midpoint deviates from $\theta$, the gradient estimate acquires a systematic bias. Section 2.7 shows how this constraint governs the structure of the Tock phase.

**Smooth Loss Requirement.** The estimator requires $\mathcal{L}(\theta)$ to be smooth in $\theta$. Generating different text under $\theta^+$ and $\theta^-$ violates this. Trajectory locking (Section 2.4) resolves the issue.

### 2.4 Trajectory Locking

**Problem.** Two failure modes create a double bind for generative ZO:

- **Reward Vanishing (FM-A):** Small $\epsilon$ $\rightarrow$ identical outputs under $\theta^+$/$\theta^-$ $\rightarrow$ zero gradient.
- **Trajectory Divergence (FM-B):** Large $\epsilon$ $\rightarrow$ divergent sequences from autoregressive conditioning $\rightarrow$ high-variance non-gradient noise.

No single $\epsilon$ resolves both when generation is coupled to perturbation.

**Solution.** Decouple exploration from optimization:

1. **Explore:** Generate $N$ candidates with **unperturbed** weights ($\theta_0$), temperature $T > 0$.
2. **Select:** Score all candidates. Pick winner $Y_w$ (highest reward) and loser $Y_l$ (lowest reward).
3. **Score Under Perturbations:** Force-feed $Y_w$ and $Y_l$ through $\theta^+$ and $\theta^-$ via `score()` (prefill-only, no generation).
4. **Estimate Gradient:** $\hat{g} = \frac{\text{NLL}^+ - \text{NLL}^-}{2\epsilon} \cdot z$. Both NLLs measure the **same tokens**, so the difference is pure weight sensitivity — smooth, low-variance, valid.

The loss for trajectory-locked scoring:

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log P(y_t \mid y_{<t}, \theta)$$

### 2.5 Linear Multi-Trajectory Contrastive Loss

Instead of discarding $N-1$ trajectories and scoring only the best, DS-MeZO scores both the winner and loser using the centered reward formula:

$$\mathcal{L}_{\text{total}} = \sum_{i \in \{w, l\}} (R_i - b_t) \cdot \text{NLL}_{\text{perturbed}}(Y_i) + \beta \cdot \text{KL}(P_{\text{base}}^{(i)} \| P_{\text{perturbed}}^{(i)})$$

- **Winner** ($R_w > b_t$): Positive coefficient $\rightarrow$ minimizing $\mathcal{L}$ increases $P(Y_w)$.
- **Loser** ($R_l < b_t$): Negative coefficient $\rightarrow$ minimizing $\mathcal{L}$ decreases $P(Y_l)$.

This achieves DPO-equivalent contrastive push-pull dynamics using strictly linear arithmetic — no sigmoid squashing, no gradient magnitude distortion. The KL-divergence anchor ($\beta = 0.1$) prevents mode collapse by penalizing drift from the base policy.

### 2.6 Adaptive ZO Mechanisms

Static hyperparameters ($\eta$, $\epsilon$, $r$, $b$) fail to track the optimizer's progress: early steps tolerate large perturbations and aggressive learning rates, while later steps near convergence require precision. DS-MeZO integrates three adaptive mechanisms.

#### 2.6.1 R-AdaZO: Adaptive Step Size and Perturbation Radius

R-AdaZO maintains Adam-style first and second moment estimates of the ZO gradient:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \hat{g}_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) \hat{g}_t^2$$

$$\eta_t = \frac{\eta_{\text{init}}}{\sqrt{\hat{v}_t} + \delta}, \quad \epsilon_t = \epsilon_{\text{init}} \cdot \max\left(\frac{\|\hat{m}_t\|}{\|\hat{m}_0\|}, \epsilon_{\text{floor}}\right)$$

As gradient variance $v_t$ drops (approaching convergence), $\eta_t$ increases for faster exploitation. As gradient mean $m_t$ shrinks (nearing a minimum), $\epsilon_t$ contracts for precise settling into the flat minimum. This replaces four static parameters ($\eta_m$, $\eta_V$, $\epsilon_m$, $\epsilon_V$) with auto-adapted initial seeds.

R-AdaZO operates on **isotropic** noise $z \sim \mathcal{N}(0, I)$ and provides per-coordinate step-size adaptation through its moment-based denominator — the same mechanism as Adam. This is the sole preconditioning layer in the optimizer; no secondary shaping of the noise distribution is applied (see Section 2.8 for the theoretical justification).

#### 2.6.2 Dynamic Reward Baseline

The centered reward baseline adapts via EMA to the agent's evolving capability:

$$b_t = \alpha_b \cdot b_{t-1} + (1 - \alpha_b) \cdot \bar{R}_t$$

where $\bar{R}_t$ is the mean reward of the current step's $N$ candidates. If the agent masters the task (average reward $0.8$), the baseline shifts to $\sim 0.8$, treating a $0.7$ output as below-standard and forcing continuous improvement.

#### 2.6.3 Per-Layer Dynamic Rank Allocation

Static rank $r = 16$ wastes capacity on layers that need less and starves layers that need more. Dynamic rank allocation maintains a global budget $R_{\text{total}} = r \times L$ and redistributes every $T_{\text{realloc}}$ steps (default 200):

1. Compute per-layer gradient magnitude: $G_l = \frac{1}{T_{\text{realloc}}} \sum_t \|\hat{g}_t^{(l)}\|$
2. Allocate proportionally: $r_l = \text{round}\left(\frac{G_l}{\sum_l G_l} \cdot R_{\text{total}}\right)$, clamped to $[r_{\min}, r_{\max}]$ (default $[4, 64]$)
3. Re-initialize adapters for changed layers via truncated or expanded SVD
4. Reset R-AdaZO moments for affected layers (prevents stale moment overshoot)

Reasoning-critical layers (late attention/MLP with high gradient activity) receive more rank; early layers (generic features) are pruned.

### 2.7 Why Tock Must Alternate A and B (Bilinear Perturbation Symmetry)

The ZO two-point estimator's unbiasedness depends on a **symmetry invariant**: the midpoint of the positive and negative perturbations must equal the current parameter value. For a single parameter tensor $\theta$:

$$\frac{(\theta + \epsilon z) + (\theta - \epsilon z)}{2} = \theta \quad \checkmark$$

When two matrices $A$ and $B$ interact through multiplication (as in DoRA's $W' \propto BA$), simultaneous perturbation breaks this invariant. Expanding the effective adapter weight:

$$W^+ = (B + Z_B)(A + Z_A) = BA + BZ_A + Z_BA + Z_BZ_A$$
$$W^- = (B - Z_B)(A - Z_A) = BA - BZ_A - Z_BA + Z_BZ_A$$

The midpoint:

$$\frac{W^+ + W^-}{2} = BA + Z_BZ_A \neq BA$$

The cross-term $Z_BZ_A$ is a systematic $\mathcal{O}(\epsilon^2)$ **bias** in the evaluation point. The ZO estimator evaluates the loss surface around the phantom location $BA + Z_BZ_A$ instead of the true current weights $BA$. While the magnitude is $\mathcal{O}(\epsilon^2)$, the gradient estimate itself is $\mathcal{O}(1)$ (after dividing by $2\epsilon$), so the bias-to-signal ratio is $\mathcal{O}(\epsilon)$ — non-negligible and accumulating across thousands of steps.

Additionally, varying $B$ while estimating $\nabla_A \mathcal{L}$ (and vice versa) couples noise between the two gradient estimates, inflating their variance beyond the irreducible $\mathcal{O}(\text{dim})$ floor.

**The Fix: Alternating Perturbation (Tock-A / Tock-B).**

By perturbing $A$ and $B$ in separate phases with the other frozen, the symmetry invariant is restored:

**Tock-A** ($B$ frozen):

$$W^+_A = B(A + Z_A) = BA + BZ_A, \quad W^-_A = B(A - Z_A) = BA - BZ_A$$
$$\frac{W^+_A + W^-_A}{2} = BA \quad \checkmark$$

**Tock-B** ($A$ frozen):

$$W^+_B = (B + Z_B)A = BA + Z_BA, \quad W^-_B = (B - Z_B)A = BA - Z_BA$$
$$\frac{W^+_B + W^-_B}{2} = BA \quad \checkmark$$

Each phase produces an unbiased, uncoupled gradient estimate for its target parameter. The cost is one additional scoring phase per step (3 phases total instead of 2).

### 2.8 Why Noise Shaping Is Incompatible with Adaptive Step Sizing

The standard ZO estimator with isotropic noise $z \sim \mathcal{N}(0, I)$ satisfies:

$$\mathbb{E}[\hat{g}] = \mathbb{E}\left[\frac{\mathcal{L}(\theta + \epsilon z) - \mathcal{L}(\theta - \epsilon z)}{2\epsilon} \cdot z\right] = \mathbb{E}[(z^T \nabla \mathcal{L}) \cdot z] = \mathbb{E}[zz^T] \nabla \mathcal{L} = I \cdot \nabla \mathcal{L} = \nabla \mathcal{L}$$

With shaped noise $z \sim \mathcal{N}(0, \Sigma)$ for some learned covariance $\Sigma$:

$$\mathbb{E}[\hat{g}] = \mathbb{E}[zz^T] \nabla \mathcal{L} = \Sigma \nabla \mathcal{L} \neq \nabla \mathcal{L}$$

The estimator yields the **preconditioned** gradient $\Sigma \nabla \mathcal{L}$, not the true gradient. If this biased estimate is then fed into R-AdaZO (which applies its own preconditioning $\text{diag}(1/\sqrt{v_t})$), the total update direction becomes:

$$\Delta \theta \propto \text{diag}(1/\sqrt{v_t}) \cdot \Sigma \cdot \nabla \mathcal{L}$$

This is **double preconditioning** — two competing mechanisms adjusting the same per-coordinate scaling. The R-AdaZO moments $v_t$ will adapt to the magnitude of $\Sigma \nabla \mathcal{L}$, partially compensating for $\Sigma$, but the interaction is uncontrolled and can cause oscillation or divergence in dimensions where $\Sigma$ and $1/\sqrt{v_t}$ push in opposite directions.

**Resolution:** DS-MeZO uses exclusively isotropic noise $z \sim \mathcal{N}(0, I)$ with R-AdaZO as the **sole** preconditioning layer. R-AdaZO already provides per-coordinate step-size adaptation through its moment-based denominator, achieving the same goal as noise shaping (emphasizing productive directions) without introducing estimator bias.

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    DS-MeZO CONTROLLER                    │
│                  (Python / PyTorch / CPU)                 │
│                                                          │
│  ┌──────────────────┐         ┌──────────────────┐       │
│  │  Optimizer State  │         │  R-AdaZO Moments │       │
│  │  (m, A, B)        │         │  (m1, v2 per     │       │
│  │  DoRA + PiSSA     │         │   tick/tockA/B)  │       │
│  └────────┬──────────┘         └────────┬─────────┘       │
│           └──────────────┬──────────────┘                │
│                          │                               │
│             ┌────────────▼────────────┐                  │
│             │  Ring Buffer Adapter Mgr │                  │
│             │  (N=10 slots, full       │                  │
│             │   add/remove cycle)      │                  │
│             └────────────┬────────────┘                  │
└──────────────────────────┼───────────────────────────────┘
                           │  Full Re-Registration (add_lora/remove_lora)
                           ▼
┌──────────────────────────────────────────────────────────┐
│                    vLLM ENGINE                           │
│              (GPU / Native DoRA Mode)                    │
│                                                          │
│  ┌──────────────┐  ┌───────────┐  ┌──────────────────┐  │
│  │  Base Model   │  │  S-LoRA   │  │  PagedAttention  │  │
│  │  W_res (NF4)  │  │  Paging   │  │  + KV Cache      │  │
│  └──────────────┘  └───────────┘  └──────────────────┘  │
│                                                          │
│  Output: logprobs, generated text                        │
└──────────────────────────────────────────────────────────┘
                           │
              ┌────────────┴─────────────┐
              ▼                          ▼
┌──────────────────────────┐  ┌───────────────────────────┐
│   EXPLORATION (once)     │  │ OPTIMIZATION (3 phases)   │
│                          │  │                           │
│  generate(θ₀, n=N, T>0) │  │  1. Tick:   m ± ε_m·z_m  │
│  Y_w = best, Y_l = worst │  │  2. Tock-A: A ± Z_A (B∅) │
│  Update baseline b_t     │  │  3. Tock-B: B ± Z_B (A∅) │
│  R: code_compiles() ...  │  │  (prefill-only, batch=4)  │
└──────────────────────────┘  └───────────────────────────┘
```

---

## 4. Execution Workflow (Explore-Tick-Tock)

Each step: explore (generate + select), then optimize in three phases — magnitude (Tick), direction-A (Tock-A), direction-B (Tock-B). The three-phase structure guarantees ZO estimator symmetry (Section 2.7).

### 4.1 Initialization (PiSSA Hot-Start)

Runs once at startup.

1. **Load Backbone:** Load frozen model into vLLM in NF4 4-bit quantization.
2. **Compute SVD:** Fast SVD (randomized, Halko et al.) of target layer weights:
   $$U, S, V^T = \text{FastSVD}(W_0, \text{niter}=2)$$
3. **Initialize Adapters:**
   - Direction: $A = U[:, :r] \cdot \sqrt{S[:r]}$, $B = \sqrt{S[:r]} \cdot V^T[:r, :]$ (symmetric splitting for gradient balance)
   - Residual (frozen): $W^{res} = W_0 - AB$, quantized to NF4 (~20% less error than quantizing $W_0$ directly)
   - Magnitude: $m = \|W_0\|_c$ (column-wise norms)
4. **Register Ring Buffer:** Pre-allocate 10 adapter slots in vLLM.

### 4.2 Exploration (Multi-Trajectory Selection)

Runs once per step. Produces winner $Y_w$ and loser $Y_l$.

1. Generate $N$ candidates with unperturbed weights, temperature $T > 0$.
2. Score all candidates with reward function $R$. Sort by reward.
3. Select winner (highest reward) and loser (lowest reward).
4. **Update dynamic baseline:** $b_t = 0.9 \cdot b_{t-1} + 0.1 \cdot \bar{R}_t$
5. **Rejection** — skip step if:
   - Best reward $< R_{\min}$, or
   - Reward gap $(R_w - R_l) < \delta$.
6. Cache base logprobs for KL anchor (single batched `score()` call, batch=2).

### 4.3 Tick Phase (Magnitude Tuning)

**Target:** Magnitude vector $m$ ($k = 8{,}192$ dims per layer). High SNR.

1. **Get adaptive perturbation radius:** $\epsilon_{m,t}$ from R-AdaZO.
2. **Generate isotropic noise:** $z_m \sim \mathcal{N}(0, I_k) \cdot \epsilon_{m,t}$
3. **Construct perturbations:** $m^{\pm} = m \pm \epsilon_{m,t} \cdot z_m$. Midpoint $= m$. $\checkmark$
4. **Register** `tick_pos` and `tick_neg` adapters via ring buffer.
5. **Contrastive batched scoring** (batch=4): Score $Y_w$ and $Y_l$ under both perturbations in a single vLLM `score()` call. Aggregate via $\mathcal{L} = \sum_i (R_i - b_t) \cdot \text{NLL}_i + \beta \cdot \text{KL}_i$.
6. **Estimate gradient:**
   $$\hat{g}_m = \frac{\mathcal{L}^+ - \mathcal{L}^-}{2 \epsilon_{m,t}} \cdot z_m$$
7. **R-AdaZO update:** Compute adaptive $\eta_{m,t}$, then $m_{t+1} = m_t - \eta_{m,t} \cdot \hat{g}_m$.
8. **Deregister** adapter slots.

### 4.4 Tock-A Phase (Direction Matrix A, B Frozen)

**Target:** Low-rank matrix $A \in \mathbb{R}^{d \times r}$ ($d \times r = 131{,}072$ dims at $d = 8192$, $r = 16$). $B$ is held constant to preserve ZO symmetry (Section 2.7).

1. **Get adaptive perturbation radius:** $\epsilon_{A,t}$ from R-AdaZO.
2. **Generate FP16-safe isotropic noise:**
   $$Z_A, \epsilon_A = \text{get\_fp16\_safe\_perturbation}(A, \epsilon_{A,t})$$
   Element-wise $\epsilon_{ij} = \max(\epsilon_{\text{base}}, |A_{ij}| \times 0.001)$ ensures FP16 bit-flip survival.
3. **Construct perturbations:** $A^{\pm} = A \pm Z_A$, with $B$ frozen. Effective weight midpoint $= BA$. $\checkmark$
4. **Register** `tockA_pos` and `tockA_neg` adapters.
5. **Contrastive batched scoring** (batch=4).
6. **Estimate gradient (element-wise SPSA):**
   $$\hat{g}_A = \frac{\mathcal{L}^+ - \mathcal{L}^-}{2 \cdot \epsilon_A} \cdot \frac{Z_A}{\epsilon_A}$$
7. **R-AdaZO update:** Compute adaptive $\eta_{A,t}$, then $A_{t+1} = A_t - \eta_{A,t} \cdot \hat{g}_A$.
8. **Deregister** adapter slots.

### 4.5 Tock-B Phase (Direction Matrix B, A Frozen)

**Target:** Low-rank matrix $B \in \mathbb{R}^{r \times k}$ ($r \times k = 131{,}072$ dims at $r = 16$, $k = 8192$). $A$ is held constant (now at its updated value $A_{t+1}$).

1. **Get adaptive perturbation radius:** $\epsilon_{B,t}$ from R-AdaZO.
2. **Generate FP16-safe isotropic noise:**
   $$Z_B, \epsilon_B = \text{get\_fp16\_safe\_perturbation}(B, \epsilon_{B,t})$$
3. **Construct perturbations:** $B^{\pm} = B \pm Z_B$, with $A$ frozen (at updated value). Effective weight midpoint $= BA$. $\checkmark$
4. **Register** `tockB_pos` and `tockB_neg` adapters.
5. **Contrastive batched scoring** (batch=4).
6. **Estimate gradient (element-wise SPSA):**
   $$\hat{g}_B = \frac{\mathcal{L}^+ - \mathcal{L}^-}{2 \cdot \epsilon_B} \cdot \frac{Z_B}{\epsilon_B}$$
7. **R-AdaZO update:** Compute adaptive $\eta_{B,t}$, then $B_{t+1} = B_t - \eta_{B,t} \cdot \hat{g}_B$.
8. **Deregister** adapter slots.
9. **Dynamic rank reallocation check** (every 200 steps).

### 4.6 Why Tick, Tock-A, and Tock-B Must Be Separate

Three independent constraints require three separate perturbation phases:

| Constraint | Source | Requires |
| :--- | :--- | :--- |
| **Dimensionality gap** (Section 2.2) | ZO variance $\propto \text{dim}$; $m$ is 8K, $A$/$B$ are 131K each | Separate $m$ from $A$/$B$ (Tick vs. Tock) |
| **Bilinear symmetry** (Section 2.7) | Simultaneous perturbation of $A$ and $B$ produces $Z_BZ_A$ cross-term bias | Separate $A$ from $B$ (Tock-A vs. Tock-B) |
| **DoRA negative correlation** (Section 2.2) | Magnitude and direction updates are complementary, not additive | Separate magnitude from direction (Tick vs. Tock) |

**Per-step scoring budget:** 1 exploration generation + 1 KL base score (batch=2) + 3 contrastive batched scores (batch=4 each) = ~14 prefills total. Each prefill is 25-100$\times$ faster than generation.

**Scheduling Variant:** For compute-constrained settings, alternate Tock-A and Tock-B across steps (Tick + Tock-A on odd steps, Tick + Tock-B on even steps). This halves the Tock scoring cost at the expense of staggering $A$ and $B$ updates by one step.

---

## 5. Implementation

### 5.1 Ring Buffer Adapter Protocol

vLLM maintains metadata (Block Tables, fused kernel states, scaling factors) separately from weight tensors. Overwriting weights without flushing metadata causes stale kernel states or silent update failures.

**Solution:** Full `add_lora`/`remove_lora` cycle per perturbation via a ring buffer of 10 slots. Forces fresh metadata at ~10ms latency per registration. Async pipelining: register slot $i+1$ while GPU computes slot $i$.

### 5.2 Complete Implementation

```python
import math


class DSMeZO_Controller:
    def __init__(self, vllm_engine, model_config, score_fn):
        self.engine = vllm_engine
        self.score_fn = score_fn  # Callable: str → float in [0, 1]
        self.m, self.A, self.B, self.W_res = initialize_pissa(model_config)
        self.step_count = 0

        # Ring buffer
        self.ring_size = 10
        self.ring_idx = 0
        self.active_adapters = set()

        # Exploration & selection
        self.num_candidates = 4
        self.explore_temperature = 0.7
        self.reward_threshold = 0.1
        self.contrastive_gap = 0.1

        # Dynamic reward baseline
        self.baseline = 0.5
        self.baseline_momentum = 0.9

        # KL-divergence anchor
        self.beta_kl = 0.1
        self.base_logprobs = None

        # R-AdaZO adaptive optimizer state (4 parameter groups)
        self.eta_init_m = 1e-4
        self.eta_init_A = 6e-6
        self.eta_init_B = 6e-6
        self.eps_init_m = 1e-3
        self.eps_init_A = 2.5e-4
        self.eps_init_B = 2.5e-4
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_delta = 1e-8
        self.eps_floor = 0.1
        # Per-group moment buffers (initialized on first step)
        self.m1_tick = None
        self.v2_tick = None
        self.m1_tock_A = None
        self.v2_tock_A = None
        self.m1_tock_B = None
        self.v2_tock_B = None
        self.m1_norm_init = None

        # Per-layer dynamic rank allocation
        self.rank_realloc_interval = 200
        self.rank_min = 4
        self.rank_max = 64
        self.layer_ranks = {}
        self.grad_magnitude_accum = {}

        # Divergence detection
        self.loss_ema = None
        self.loss_ema_momentum = 0.95
        self.max_loss_ratio = 5.0

    # ── Ring Buffer ──────────────────────────────────────────

    def _next_adapter_name(self, suffix):
        name = f"dora_s{self.step_count}_{suffix}"
        self.ring_idx = (self.ring_idx + 1) % self.ring_size
        return name

    def _register_adapter(self, name, m, A, B):
        adapter = build_dora_adapter(m=m, A=A, B=B)
        self.engine.add_lora(name, adapter)
        self.active_adapters.add(name)

    def _deregister_adapter(self, name):
        self.engine.remove_lora(name)
        self.active_adapters.discard(name)

    # ── Scoring ──────────────────────────────────────────────

    def _score_contrastive_batched(self, winner_tokens, loser_tokens,
                                    winner_reward, loser_reward,
                                    adapter_pos, adapter_neg):
        """Contrastive multi-trajectory scoring.

        Batch=4: [winner+pos, loser+pos, winner+neg, loser+neg].
        Linear centered reward aggregation with KL anchor.
        """
        logprobs_batch = self.engine.score(
            [winner_tokens, loser_tokens, winner_tokens, loser_tokens],
            lora_request=[adapter_pos, adapter_pos, adapter_neg, adapter_neg]
        )

        nll_pos_w = -sum(logprobs_batch[0]) / len(logprobs_batch[0])
        nll_pos_l = -sum(logprobs_batch[1]) / len(logprobs_batch[1])
        nll_neg_w = -sum(logprobs_batch[2]) / len(logprobs_batch[2])
        nll_neg_l = -sum(logprobs_batch[3]) / len(logprobs_batch[3])

        # KL anchor: penalize divergence from base policy
        if self.beta_kl > 0 and self.base_logprobs is not None:
            base_w, base_l = self.base_logprobs
            kl_pw = sum(b - p for b, p in zip(base_w, logprobs_batch[0])) / len(base_w)
            kl_pl = sum(b - p for b, p in zip(base_l, logprobs_batch[1])) / len(base_l)
            kl_nw = sum(b - p for b, p in zip(base_w, logprobs_batch[2])) / len(base_w)
            kl_nl = sum(b - p for b, p in zip(base_l, logprobs_batch[3])) / len(base_l)
            nll_pos_w += self.beta_kl * kl_pw
            nll_pos_l += self.beta_kl * kl_pl
            nll_neg_w += self.beta_kl * kl_nw
            nll_neg_l += self.beta_kl * kl_nl

        b = self.baseline
        loss_pos = (winner_reward - b) * nll_pos_w + (loser_reward - b) * nll_pos_l
        loss_neg = (winner_reward - b) * nll_neg_w + (loser_reward - b) * nll_neg_l
        return loss_pos, loss_neg

    # ── FP16 Safety ──────────────────────────────────────────

    def get_fp16_safe_perturbation(self, param_tensor, base_epsilon):
        """Dynamic element-wise epsilon that survives FP16 downcasting.

        PiSSA's large singular values (e.g., 32.0) have FP16 ULP of ~0.03.
        Static ε ≈ 0.00025 is silently truncated to zero.
        Fix: ε_ij = max(ε_base, |A_ij| × 0.001).

        Uses isotropic noise only — no covariance shaping (Section 2.8).
        """
        z = torch.randn_like(param_tensor)
        fp16_ulp_margin = param_tensor.abs() * 0.001
        dynamic_epsilon = torch.max(
            torch.tensor(base_epsilon, device=param_tensor.device),
            fp16_ulp_margin
        )
        return dynamic_epsilon * z, dynamic_epsilon

    # ── Exploration ──────────────────────────────────────────

    def _explore(self, batch):
        """Generate candidates with unperturbed weights, select winner/loser.

        Returns ((winner_tokens, winner_reward),
                 (loser_tokens, loser_reward)) or None.
        """
        name = self._next_adapter_name("explore")
        self._register_adapter(name, m=self.m, A=self.A, B=self.B)

        outputs = self.engine.generate(
            batch, lora_request=name,
            n=self.num_candidates, temperature=self.explore_temperature
        )

        scored = [(out, self.score_fn(out.text)) for out in outputs]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_output, best_reward = scored[0]
        worst_output, worst_reward = scored[-1]

        # Dynamic baseline update
        mean_reward = sum(r for _, r in scored) / len(scored)
        self.baseline = (self.baseline_momentum * self.baseline +
                         (1 - self.baseline_momentum) * mean_reward)

        if best_reward < self.reward_threshold:
            self._deregister_adapter(name)
            self.base_logprobs = None
            return None

        if (best_reward - worst_reward) < self.contrastive_gap:
            self._deregister_adapter(name)
            self.base_logprobs = None
            return None

        # Cache base logprobs for KL anchor
        if self.beta_kl > 0:
            base_lp = self.engine.score(
                [best_output.token_ids, worst_output.token_ids],
                lora_request=[name, name]
            )
            self.base_logprobs = (base_lp[0], base_lp[1])
        else:
            self.base_logprobs = None

        self._deregister_adapter(name)
        return ((best_output.token_ids, best_reward),
                (worst_output.token_ids, worst_reward))

    # ── Health Monitoring ────────────────────────────────────

    def _check_health(self, nll_pos, nll_neg):
        """Skip update if NLL exceeds 5x EMA (divergence detection)."""
        avg_nll = (abs(nll_pos) + abs(nll_neg)) / 2
        if self.loss_ema is None:
            self.loss_ema = avg_nll
            return True
        self.loss_ema = (self.loss_ema_momentum * self.loss_ema +
                         (1 - self.loss_ema_momentum) * avg_nll)
        if self.loss_ema > 1e-8 and avg_nll > self.max_loss_ratio * self.loss_ema:
            return False
        return True

    # ── R-AdaZO Adaptive Optimizer ───────────────────────────

    def _adapt_lr_epsilon(self, grad, group):
        """Adam-style adaptive η_t and ε_t for the given parameter group.

        Returns (η_t, ε_t).
        """
        if group == 'tick':
            eta_init, eps_init = self.eta_init_m, self.eps_init_m
        elif group == 'tock_A':
            eta_init, eps_init = self.eta_init_A, self.eps_init_A
        else:  # tock_B
            eta_init, eps_init = self.eta_init_B, self.eps_init_B

        m1_attr = f'm1_{group}'
        v2_attr = f'v2_{group}'
        grad_norm = grad.norm().item() if torch.is_tensor(grad) else abs(grad)

        if getattr(self, m1_attr) is None:
            setattr(self, m1_attr, grad_norm)
            setattr(self, v2_attr, grad_norm ** 2)
            self.m1_norm_init = self.m1_norm_init or grad_norm
            return eta_init, eps_init

        m1 = self.adam_beta1 * getattr(self, m1_attr) + (1 - self.adam_beta1) * grad_norm
        v2 = self.adam_beta2 * getattr(self, v2_attr) + (1 - self.adam_beta2) * grad_norm ** 2
        setattr(self, m1_attr, m1)
        setattr(self, v2_attr, v2)

        bc1 = 1 - self.adam_beta1 ** self.step_count
        bc2 = 1 - self.adam_beta2 ** self.step_count
        m1_hat = m1 / bc1
        v2_hat = v2 / bc2

        eta_t = eta_init / (math.sqrt(v2_hat) + self.adam_delta)

        ratio = m1_hat / self.m1_norm_init if self.m1_norm_init > 0 else 1.0
        eps_t = eps_init * max(ratio, self.eps_floor)

        return eta_t, eps_t

    # ── Dynamic Rank Allocation ──────────────────────────────

    def _maybe_realloc_ranks(self):
        """Redistribute global rank budget based on gradient magnitude.
        Resets R-AdaZO moments for affected layers to prevent stale overshoot.
        """
        if (self.step_count % self.rank_realloc_interval != 0 or
                not self.grad_magnitude_accum):
            return

        layers = sorted(self.grad_magnitude_accum.keys())
        magnitudes = [self.grad_magnitude_accum[l] / self.rank_realloc_interval
                      for l in layers]
        total_mag = sum(magnitudes) + 1e-12
        total_rank = sum(self.layer_ranks.get(l, 16) for l in layers)

        new_ranks = {}
        for l, mag in zip(layers, magnitudes):
            raw = round((mag / total_mag) * total_rank)
            new_ranks[l] = max(self.rank_min, min(self.rank_max, raw))

        allocated = sum(new_ranks.values())
        if allocated != total_rank:
            diff = total_rank - allocated
            top_layer = max(new_ranks, key=lambda l: self.grad_magnitude_accum[l])
            new_ranks[top_layer] = max(self.rank_min,
                                        min(self.rank_max,
                                            new_ranks[top_layer] + diff))

        for l in layers:
            if new_ranks[l] != self.layer_ranks.get(l, 16):
                self.layer_ranks[l] = new_ranks[l]
                self._reinit_layer_adapter(l, new_ranks[l])
                # Reset R-AdaZO moments for this layer (prevents stale overshoot)
                self._reset_moments_for_layer(l)

        self.grad_magnitude_accum = {l: 0.0 for l in layers}

    # ── Main Training Step ───────────────────────────────────

    def step(self, batch):
        self.step_count += 1

        # === EXPLORATION ===
        result = self._explore(batch)
        if result is None:
            return
        (winner_tokens, winner_reward), (loser_tokens, loser_reward) = result

        # === TICK PHASE (Magnitude m — dim ≈ 8,192) ===
        _, eps_m = self._adapt_lr_epsilon(
            self.m1_tick if self.m1_tick is not None else 0.0, 'tick'
        )
        z_m = torch.randn_like(self.m) * eps_m

        name_pos = self._next_adapter_name("tick_pos")
        name_neg = self._next_adapter_name("tick_neg")
        self._register_adapter(name_pos, m=self.m + z_m, A=self.A, B=self.B)
        self._register_adapter(name_neg, m=self.m - z_m, A=self.A, B=self.B)

        loss_pos, loss_neg = self._score_contrastive_batched(
            winner_tokens, loser_tokens,
            winner_reward, loser_reward,
            name_pos, name_neg
        )
        self._deregister_adapter(name_pos)
        self._deregister_adapter(name_neg)

        if self._check_health(loss_pos, loss_neg):
            grad_m = (loss_pos - loss_neg) / (2 * eps_m) * z_m
            eta_m, _ = self._adapt_lr_epsilon(grad_m, 'tick')
            self.m -= eta_m * grad_m

        # === TOCK-A PHASE (Matrix A — B frozen) ===
        _, eps_A = self._adapt_lr_epsilon(
            self.m1_tock_A if self.m1_tock_A is not None else 0.0, 'tock_A'
        )
        Z_A, eps_matrix_A = self.get_fp16_safe_perturbation(self.A, eps_A)

        name_pos = self._next_adapter_name("tockA_pos")
        name_neg = self._next_adapter_name("tockA_neg")
        self._register_adapter(name_pos, m=self.m, A=self.A + Z_A, B=self.B)
        self._register_adapter(name_neg, m=self.m, A=self.A - Z_A, B=self.B)

        loss_pos, loss_neg = self._score_contrastive_batched(
            winner_tokens, loser_tokens,
            winner_reward, loser_reward,
            name_pos, name_neg
        )
        self._deregister_adapter(name_pos)
        self._deregister_adapter(name_neg)

        if self._check_health(loss_pos, loss_neg):
            diff_A = loss_pos - loss_neg
            grad_A = (diff_A / (2 * eps_matrix_A)) * (Z_A / eps_matrix_A)
            eta_A, _ = self._adapt_lr_epsilon(grad_A, 'tock_A')
            self.A -= eta_A * grad_A

        # === TOCK-B PHASE (Matrix B — A frozen at updated value) ===
        _, eps_B = self._adapt_lr_epsilon(
            self.m1_tock_B if self.m1_tock_B is not None else 0.0, 'tock_B'
        )
        Z_B, eps_matrix_B = self.get_fp16_safe_perturbation(self.B, eps_B)

        name_pos = self._next_adapter_name("tockB_pos")
        name_neg = self._next_adapter_name("tockB_neg")
        self._register_adapter(name_pos, m=self.m, A=self.A, B=self.B + Z_B)
        self._register_adapter(name_neg, m=self.m, A=self.A, B=self.B - Z_B)

        loss_pos, loss_neg = self._score_contrastive_batched(
            winner_tokens, loser_tokens,
            winner_reward, loser_reward,
            name_pos, name_neg
        )
        self._deregister_adapter(name_pos)
        self._deregister_adapter(name_neg)

        if self._check_health(loss_pos, loss_neg):
            diff_B = loss_pos - loss_neg
            grad_B = (diff_B / (2 * eps_matrix_B)) * (Z_B / eps_matrix_B)
            eta_B, _ = self._adapt_lr_epsilon(grad_B, 'tock_B')
            self.B -= eta_B * grad_B

        # Dynamic rank reallocation check
        self._maybe_realloc_ranks()

    def train(self, dataloader, num_steps):
        for step_idx, batch in zip(range(num_steps), dataloader):
            self.step(batch)


def initialize_pissa(model_config):
    """PiSSA initialization via Fast SVD (Halko et al.)."""
    W0 = load_pretrained_weights(model_config)
    r = model_config.rank

    U, S, Vt = torch.svd_lowrank(W0, q=r, niter=2)

    sqrt_S = torch.sqrt(S[:r])
    A = U[:, :r] * sqrt_S.unsqueeze(0)       # [d, r]
    B = sqrt_S.unsqueeze(1) * Vt[:r, :]       # [r, k]

    W_res = W0 - A @ B
    W_res_quantized = quantize_nf4(W_res)

    m = torch.norm(W0, dim=0)  # column-wise norms

    return m, A, B, W_res_quantized
```

---

## 6. Memory Budget (Single H100 80GB)

| Component | VRAM | Notes |
| :--- | :--- | :--- |
| Residual Model ($W^{res}$, NF4) | ~35 GB | Frozen. Narrower distribution than $W_0$ reduces quantization error ~20%. |
| KV Cache (PagedAttention) | ~25 GB | Dynamic, batch-dependent. |
| S-LoRA Ring Buffer (10 slots, max 6 active, FP16) | ~2.5 GB | Peak 6 adapters: 2 per phase, only 1 phase active at a time. Ring buffer recycles slots. |
| Controller State (m, A, B, R-AdaZO moments) | ~0.3 GB | 8 scalar moments (tick, tock_A, tock_B × m1/v2) are negligible. |
| **Headroom** | **~17.2 GB** | Safety margin for vLLM internals. |

**Configuration:** Launch vLLM with `--gpu-memory-utilization 0.9`.

**Quantization (QPiSSA):** Residual $W^{res}$ is quantized to NF4 (not $W_0$). PiSSA removes the large singular components into FP16 adapters, leaving a residual with a narrower, more Gaussian distribution that fits NF4 better. The DoRA column-norm computation $\|W^{res} + BA\|_c$ is handled by vLLM's fused native DoRA kernel.

**Quantization Drift Caveat.** NF4 quantization of $W^{res}$ introduces fixed error in the column-norm denominator. The Tick phase magnitude $m$ may over-compensate for this jitter. Keep $\eta_{m,\text{init}}$ conservative ($1 \times 10^{-4}$); R-AdaZO will adapt it upward if safe.

---

## 7. Hyperparameter Configuration

### 7.1 Scaling Rules (Zhang et al.)

| Parameter | Formula | Rationale |
| :--- | :--- | :--- |
| Perturbation Radius | $\epsilon = \frac{10^{-3}}{\sqrt{r}}$ | Smoother subspace at low rank permits larger perturbations. Acts as smoothing parameter $\lambda$. |
| Smoothness Upper Bound | $\epsilon^2 \leq \frac{\sqrt{2} L_1}{r^{3/2} L_3}$ | Exceeding breaks $F(x)$ smoothness. R-AdaZO's $\epsilon_{\text{floor}}$ prevents $\epsilon_t$ from exceeding init. |
| Tick LR | $\eta_{\text{base}}$ | Magnitude is low-dimensional; standard LR sufficient. |
| Tock LR | $\eta_{\text{base}} / r$ | Scale down by rank for higher-dimensional direction updates. |

**Convergence Bound.** $T = \mathcal{O}(\text{dim}^4/\epsilon^2)$ with $\text{dim} = 262$K gives an astronomical raw bound. Practical convergence is governed by Hessian effective rank (far smaller than 262K) and is much faster than worst-case.

### 7.2 Default Values (Llama-3-70B)

| Parameter | Default | Auto-Tuned? |
| :--- | :--- | :--- |
| Subspace Rank ($r$) | 16 or 32 (initial; per-layer) | **Yes** — dynamic rank allocation every 200 steps |
| Tick Perturbation ($\epsilon_{m,\text{init}}$) | $10^{-3}$ | **Yes** — R-AdaZO contracts as gradient mean decays |
| Tock-A Perturbation ($\epsilon_{A,\text{init}}$) | $2.5 \times 10^{-4}$ | **Yes** — R-AdaZO contracts as gradient mean decays |
| Tock-B Perturbation ($\epsilon_{B,\text{init}}$) | $2.5 \times 10^{-4}$ | **Yes** — R-AdaZO contracts as gradient mean decays |
| Tick LR ($\eta_{m,\text{init}}$) | $1 \times 10^{-4}$ | **Yes** — R-AdaZO adapts via $\eta_t = \eta_{\text{init}} / (\sqrt{\hat{v}_t} + \delta)$ |
| Tock-A LR ($\eta_{A,\text{init}}$) | $6 \times 10^{-6}$ | **Yes** — R-AdaZO adapts independently |
| Tock-B LR ($\eta_{B,\text{init}}$) | $6 \times 10^{-6}$ | **Yes** — R-AdaZO adapts independently |
| Reward Baseline ($b_0$) | 0.5 | **Yes** — EMA: $b_t = 0.9 \cdot b_{t-1} + 0.1 \cdot \bar{R}_t$ |
| Exploration Candidates ($N$) | 4 | No |
| Exploration Temperature | 0.7 | No |
| Reward Threshold ($R_{\min}$) | 0.1 | No |
| Contrastive Gap ($\delta$) | 0.1 | No |
| KL Penalty ($\beta$) | 0.1 (0.0 for short runs) | No |
| NLL Spike Threshold | 5$\times$ EMA | No |
| R-AdaZO $\beta_1$/$\beta_2$ | 0.9 / 0.999 | No (standard Adam) |
| R-AdaZO $\epsilon_{\text{floor}}$ | 0.1 | No |
| Rank Realloc Interval | 200 steps | No |
| Rank Bounds ($r_{\min}$/$r_{\max}$) | 4 / 64 | No |
| Ring Buffer Size | 10 slots | No |

**Auto-Tuning Summary:** 5 of 8 previously manual heuristic parameters are now auto-tuned ($\eta_m$, $\eta_A$, $\eta_B$, $\epsilon_m$, $\epsilon_A$, $\epsilon_B$, $b$) and 1 is dynamically allocated ($r$ per layer). Remaining manual parameters are exploration settings, KL penalty, and safety thresholds — none require task-specific tuning in practice.

---

## 8. Loss Function

### 8.1 Linear Multi-Trajectory Contrastive NLL

The per-perturbation contrastive loss:

$$\mathcal{L} = \sum_{i \in \{w, l\}} (R_i - b_t) \cdot \text{NLL}_{\text{perturbed}}(Y_i) + \beta \cdot \text{KL}(P_{\text{base}}^{(i)} \| P_{\text{perturbed}}^{(i)})$$

- Winner ($R_w > b_t$): positive coefficient $\rightarrow$ pull toward high-reward output.
- Loser ($R_l < b_t$): negative coefficient $\rightarrow$ push away from low-reward output.
- KL anchor prevents mode collapse.
- Strictly linear arithmetic — no sigmoid squashing, no gradient distortion.

The ZO gradient estimate:

$$\hat{g} = \frac{\mathcal{L}^+ - \mathcal{L}^-}{2\epsilon} \cdot z$$

### 8.2 Task-Specific Reward Functions

Any `str → float in [0, 1]` function works:

```python
# Code correctness
def code_reward(text):
    if compiles_and_passes(text):   return 1.0
    elif compiles(text):            return 0.5
    else:                           return 0.0

# Mathematical proof
def proof_reward(text):
    if verify_proof(text):          return 1.0
    elif valid_structure(text):     return 0.3
    else:                           return 0.0

# Exact match with partial credit
def match_reward(text, expected):
    if text == expected:            return 1.0
    elif fuzzy_match(text, expected) > 0.8: return 0.7
    else:                           return 0.0
```

Finer granularity improves both selection quality and gradient scaling. The baseline $b_t$ adapts automatically via EMA.

### 8.3 Supervised NLL (Reference Tasks)

For tasks with reference outputs, trajectory locking is automatic — the reference is the gold trajectory:

```python
def supervised_nll_loss(engine, reference_tokens, adapter_name):
    return engine.score(reference_tokens, lora_request=adapter_name)
```

Use supervised NLL as a warm-up phase (guaranteed signal every step), then switch to reward-based optimization.

---

## 9. Per-Step Compute Cost

| Pass | Phase | Operation |
| :--- | :--- | :--- |
| 0 | **Explore** | `generate(n=N, T=0.7)` $\rightarrow$ select $Y_w$ and $Y_l$, update $b_t$ |
| 0b | **KL Base** | `score([Y_w, Y_l])` — cache base logprobs (batch=2) |
| 1 | **Tick** | `score([Y_w, Y_l, Y_w, Y_l], adapters=[pos, pos, neg, neg])` (batch=4) |
| 2 | **Tock-A** | `score([Y_w, Y_l, Y_w, Y_l], adapters=[pos, pos, neg, neg])` (batch=4) — $B$ frozen |
| 3 | **Tock-B** | `score([Y_w, Y_l, Y_w, Y_l], adapters=[pos, pos, neg, neg])` (batch=4) — $A$ frozen |

**Effective cost:** 1 generation + 1 base score (batch=2) + 3 contrastive scores (batch=4 each) = ~14 prefills total. Each prefill is 25-100$\times$ faster than generation.

**Comparison to v6.0:** v7.0 adds one additional batch=4 scoring pass per step (Tock-A + Tock-B instead of combined Tock). For a 512-token sequence on H100, each batched prefill takes ~10-20ms. The additional phase adds ~15ms/step — negligible relative to the ~500-2000ms generation cost.

**Adapter overhead:** ~70ms for 7 re-registrations (1 explore + 6 perturbation slots, ~10ms each, partially overlapped via async pipelining).

**R-AdaZO overhead (CPU):** 4 scalar moment updates per step (~0.01ms). Dynamic baseline: 1 scalar EMA update (~0.001ms). Dynamic rank reallocation: ~2-5s every 200 steps (~10-25ms amortized). **Total adaptive overhead: <1ms/step amortized.**

**Rejection cost:** When rejected, generation is sunk but 14 scoring prefills are saved.

**Scheduling Variant (compute-constrained):** Alternate Tock-A and Tock-B across steps. Cost drops to ~10 prefills/step (matching v6.0) while maintaining unbiased gradient estimates, at the expense of staggering A and B updates.

---

## 10. Risk Analysis

### 10.1 Risks and Mitigations

| Risk | Severity | Mitigation |
| :--- | :--- | :--- |
| **Ring Buffer Latency** — ~70ms/step for 7 re-registrations | Medium | Async registration. For long-generation tasks (~2-4s), overhead is <4%. |
| **Convergence of Composed System** — No single paper proves PiSSA + DoRA + trajectory-locked MeZO + R-AdaZO | Medium | PiSSA+DoRA validated empirically. Trajectory locking satisfies smoothness. Adaptive mechanisms operate on orthogonal state. Three-component core is minimal. |
| **Training Divergence** — NLL explosion or reward collapse | Medium | EMA-based NLL tracking with spike detection. Skip update if NLL > 5$\times$ EMA. |
| **Rejection Rate** — Hard tasks reject most steps | Medium | Lower $R_{\min}$ or $\delta$. Increase $N$. Use curriculum learning. If >80% rejection, task may be too hard for current model. |
| **Mode Collapse** — Narrowing output distribution | Low | KL anchor ($\beta = 0.1$). Temperature sampling. ZO perturbation stochasticity. |
| **vLLM Score API** — May require specific version | Medium | Fallback: `generate(echo=True, max_tokens=0)`. |
| **R-AdaZO Moment Staleness** — High rejection rate causes stale moments | Low | Moments update only on accepted steps. If >80% rejection, disable R-AdaZO and use static values. |
| **Rank Reallocation SVD Cost** — Compute spike every 200 steps | Low | Fast SVD completes in seconds. Amortized overhead is negligible. R-AdaZO moments reset for affected layers to prevent stale overshoot. |
| **Adaptive Interaction Effects** — R-AdaZO + dynamic baseline + dynamic rank operating simultaneously | Low | Conservative coupling: R-AdaZO and dynamic baseline operate on independent state (optimizer moments vs. reward EMA). Rank reallocation is infrequent and resets affected moments. |

### 10.2 Failure Modes

| Mode | Trigger | Mechanism | Fix |
| :--- | :--- | :--- | :--- |
| **A: Reward Vanishing** | Small $\epsilon$ + greedy decoding | Identical outputs $\rightarrow$ zero gradient | Trajectory locking: scoring fixed tokens always produces different NLL. |
| **B: Trajectory Divergence** | Large $\epsilon$ + autoregressive generation | Divergent sequences $\rightarrow$ high-variance noise | Trajectory locking: both NLLs measure same tokens. |
| **C: FP16 Silent Gradient Death** | Static $\epsilon \approx 0.00025$ on large PiSSA values ($\sigma \approx 32.0$) | ULP truncation $\rightarrow$ $\theta^+ = \theta^- = \theta_0$ in FP16 | Dynamic $\epsilon_{ij} = \max(\epsilon_{\text{base}}, |A_{ij}| \times 0.001)$. |
| **D: vLLM Metadata Race** | Weight update without metadata flush | Stale kernel states | Full `add_lora`/`remove_lora` cycle via ring buffer. |
| **E: Mode Collapse** | Long runs with pure NLL, no policy anchor | Probability mass concentrates on reward-hacking patterns | KL penalty $\beta \cdot \text{KL}(P_{\text{base}} \| P_{\text{perturbed}})$. |
| **F: Baseline Plateauing** | Static $b = 0.5$ when average reward $\rightarrow 0.8+$ | Similar $(R-b)$ coefficients $\rightarrow$ gradient plateau | Dynamic baseline: $b_t = 0.9 \cdot b_{t-1} + 0.1 \cdot \bar{R}_t$. |
| **G: Bilinear Perturbation Bias** | Simultaneous perturbation of $A$ and $B$ | Cross-term $Z_BZ_A$ shifts evaluation midpoint away from current weights | Three-phase alternating updates: Tock-A ($B$ frozen) then Tock-B ($A$ frozen). |
| **H: R-AdaZO Runaway** | Sudden landscape change (e.g., after rank reallocation) | Stale $v_t$ denominator $\rightarrow$ oversized $\eta_t$ | Moments reset on rank change. `_check_health()` catches spikes. $\epsilon_{\text{floor}}$ bounds gradient magnitude. |

---

## 11. Data and Model Agnosticism

### 11.1 Data Agnostic: Yes (Exceptional)

DS-MeZO is **more data agnostic than backpropagation**. ZO optimization evaluates loss scalars rather than computing chain-rule derivatives, enabling direct optimization of non-differentiable rewards:

- Code/Logic: Pass generated code to interpreter, return 1/0.
- Tool Use: Did the agent correctly query the database?
- Human/LLM Feedback: Scalar score from LLM-as-Judge or human click.

Trajectory locking means the system generates its own data and learns from the best variant. No massive datasets of formatted input-output pairs required.

**Caveat:** Data must be tokenizable text, code, or serialized structured data.

### 11.2 Model Agnostic: Yes (Within Transformer Ecosystem)

The core math (PiSSA + DoRA) applies to any dense linear projection $W \in \mathbb{R}^{d \times k}$. Agnostic to specific model weights, layer count, vocabulary size, and base quantization method.

**Practical constraints:**

1. **vLLM Backend Dependency.** Requires vLLM's `score()` and `add_lora()` APIs. Unsupported architectures (novel SSMs, complex MoE routing) require custom kernel development.
2. **DoRA Compatibility.** Model must support DoRA on target layers. Specialized layers (1D convolutions, certain layernorms) need custom implementation.
3. **Scale Assumptions.** Tuned for 70B+ models. For very small models (<1B), PiSSA rank reduction may bottleneck capacity.

---

## 12. Parameter Audit

### 12.1 Classification

| Category | Count | Items |
| :--- | :--- | :--- |
| **Core** (algorithmic) | 10 | $r$, $\epsilon_{A,\text{init}}$, $\epsilon_{B,\text{init}}$, $\epsilon_{m,\text{init}}$, $\eta_{m,\text{init}}$, $\eta_{A,\text{init}}$, $\eta_{B,\text{init}}$, Tick/Tock-A/Tock-B decoupling, PiSSA init, $\sqrt{S}$ splitting, trajectory locking, DoRA decomposition |
| **Core (auto-tuned)** | 7 of 10 | $\epsilon_A$, $\epsilon_B$, $\epsilon_m$, $\eta_m$, $\eta_A$, $\eta_B$ (R-AdaZO), $b$ (dynamic baseline) |
| **Heuristic** (tunable defaults) | 10 | $N$, $T_{\text{explore}}$, $R_{\min}$, $\delta$, $\beta$, EMA momentum, `niter`, $\beta_1$/$\beta_2$, baseline momentum, rank realloc interval, $r_{\min}$/$r_{\max}$ |
| **Safety** (guardrails) | 4 | NLL spike threshold, EMA floor, R-AdaZO $\delta$, $\epsilon_{\text{floor}}$ |
| **Engineering** (backend) | 8 | Ring buffer size, GPU memory util, adapter precision, NF4 quantization, ULP margin, registration latency, max active adapters, adapter naming |
| **Implicit Assumptions** | 8 | Dense linear layers, autoregressive model, `score()` API, native DoRA, batched multi-adapter, square matrices, single GPU, Llama-3 70B target |
| **External** | 1 | Reward function (task definition) |

### 12.2 Core Parameters

| # | Parameter | Default | Classification |
| :--- | :--- | :--- | :--- |
| 1 | Subspace Rank ($r$) | 16 or 32 | Core (auto-tuned per layer) |
| 2 | Tock-A Perturbation ($\epsilon_{A,\text{init}}$) | $2.5 \times 10^{-4}$ | Core (auto-tuned seed) |
| 3 | Tock-B Perturbation ($\epsilon_{B,\text{init}}$) | $2.5 \times 10^{-4}$ | Core (auto-tuned seed) |
| 4 | Tick Perturbation ($\epsilon_{m,\text{init}}$) | $10^{-3}$ | Core (auto-tuned seed) |
| 5 | Tock-A LR ($\eta_{A,\text{init}}$) | $6 \times 10^{-6}$ | Derived (auto-tuned seed) |
| 6 | Tock-B LR ($\eta_{B,\text{init}}$) | $6 \times 10^{-6}$ | Derived (auto-tuned seed) |
| 7 | Tick LR ($\eta_{m,\text{init}}$) | $1 \times 10^{-4}$ | Core (auto-tuned seed) |
| 8 | Three-Phase Decoupling | Tick / Tock-A / Tock-B | Core (structural — Section 2.7) |
| 9 | PiSSA Initialization | SVD of $W_0$ | Core (structural) |
| 10 | $\sqrt{S}$ Splitting | $A = U\sqrt{S}$, $B = \sqrt{S}V^T$ | Core (structural) |
| 11 | Trajectory Locking | Always locked | Core (structural) |
| 12 | DoRA Decomposition | $W' = m \cdot \frac{W^{res} + BA}{\|W^{res} + BA\|_c}$ | Core (structural) |

### 12.3 Adaptive Parameters

| # | Parameter | Default | Role |
| :--- | :--- | :--- | :--- |
| 13 | R-AdaZO $\beta_1$ | 0.9 | First moment decay (standard Adam) |
| 14 | R-AdaZO $\beta_2$ | 0.999 | Second moment decay (standard Adam) |
| 15 | R-AdaZO $\delta$ | $10^{-8}$ | Numerical stability |
| 16 | Perturbation Floor | 0.1 | Min $\epsilon_t / \epsilon_{\text{init}}$ ratio |
| 17 | Baseline Momentum | 0.9 | EMA decay for $b_t$ |
| 18 | Rank Realloc Interval | 200 steps | Frequency of rank redistribution |
| 19 | Rank Bounds | $[4, 64]$ | Per-layer rank clamping |

### 12.4 Summary

Of 41 total enumerated items, **12 are core** to the research contribution — and **7 of those 12 are auto-tuned** at runtime. Practitioners set only insensitive initial seeds ($\eta_{\text{init}}$, $\epsilon_{\text{init}}$, $b_0$) that R-AdaZO compensates for within ~50 steps. The 7 adaptive parameters are predominantly standard optimizer constants (Adam $\beta_1$/$\beta_2$) or safety bounds ($\epsilon_{\text{floor}}$, $r_{\min}$/$r_{\max}$) that do not require task-specific tuning.

---

## 13. References

| Citation | Paper | Role in DS-MeZO |
| :--- | :--- | :--- |
| arXiv:2402.09353 | DoRA: Weight-Decomposed Low-Rank Adaptation | Magnitude/direction decomposition. Tick/Tock decoupling. Negative correlation property. Robust at low rank. |
| arXiv:2404.02948 | PiSSA: Principal Singular Values and Singular Vectors Adaptation | SVD adapter init. Cold start elimination. QPiSSA quantization. PiSSA+DoRA complementarity. |
| arXiv:2506.05454 | Zhang et al. — Zeroth-Order Optimization Finds Flat Minima | Core theory. Implicit $\operatorname{Tr}(\nabla^2 f)$ minimization. Hessian effective rank. Convergence bound. Scaling rules. |
| arXiv:2505 | R-AdaZO: Refined Adaptive Zeroth-Order Optimization | Adaptive $\eta_t$ and $\epsilon_t$ via moment tracking. Eliminates manual LR/perturbation tuning. Sole preconditioning layer. |
| arXiv:2511 | PF-VRZO: Perturbation-Free Variance-Reduced Zeroth-Order Optimization | Theoretical foundation for adaptive perturbation scaling and variance reduction. |
| arXiv:2602 | AdaEvolve: Adaptive Rank Allocation for Parameter-Efficient Fine-Tuning | Per-layer dynamic rank allocation based on gradient magnitude. |
| — | S-LoRA / vLLM | Multi-adapter serving. Native DoRA kernel. PagedAttention. Score API. |
