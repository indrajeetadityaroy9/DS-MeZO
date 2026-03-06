# DS-MeZO: Decoupled-Switched Zeroth-Order Optimization

**Unified Design Document (Theory + Implementation)**

**Target Hardware:** Single NVIDIA H100 (80GB HBM3)
**Architecture:** Python/PyTorch Controller + vLLM (S-LoRA Backend, Controller-Side DoRA Fusion)

---

## 1. Executive Summary

DS-MeZO is a gradient-free optimization system for fine-tuning Large Language Models on a single GPU. It eliminates backpropagation entirely, using zeroth-order (ZO) estimation to optimize non-differentiable objectives — code compilation, proof verification, tool use, human feedback — at near-inference memory cost.

| Component | Role |
| :--- | :--- |
| **PiSSA** (arXiv:2404.02948) | SVD-based adapter initialization. Rank-$r$ dimensionality reduction. QPiSSA quantizes the residual. |
| **DoRA** (arXiv:2402.09353) | Magnitude/direction decomposition via $m \cdot \frac{W^{res}+AB}{\|W^{res}+AB\|_c}$. Enables Tick/Tock decoupling. Controller-side optimization; fused to standard LoRA for vLLM serving. |
| **Zhang et al.** (arXiv:2506.05454) | ZO implicitly minimizes $\operatorname{Tr}(\nabla^2 f)$ (flat minima). PL convergence: $T = \mathcal{O}(d \cdot V / \mu^2\epsilon)$. |
| **Trajectory Locking** | Generate with unperturbed weights, score fixed tokens under perturbations. Eliminates trajectory divergence. |

**Pipeline:** Activation-refined perturbation within PiSSA subspace $\to$ LOREN element-wise scaling $\to$ RLOO contrastive selection $\to$ per-token KL-shaped loss $\to$ cyclic three-phase Tick/Tock updates $\to$ noise-calibrated Tensor Adam with FP32 accumulation $\to$ per-layer dynamic rank allocation.

**Design Principle:** The Controller (CPU/PyTorch) handles optimization logic including DoRA decomposition; vLLM (GPU) handles all forward passes via standard LoRA adapters. Any inference server becomes a training server.

---

## 2. The Zeroth-Order Smoothing Objective

For a loss function $f(\theta)$ and isotropic Gaussian perturbation $z \sim \mathcal{N}(0, I)$, the symmetric two-point ZO gradient estimator is:
$$ \hat{g} = \frac{f(\theta + \epsilon z) - f(\theta - \epsilon z)}{2\epsilon} \cdot z $$

This is an unbiased estimator of the gradient of a **Gaussian-smoothed surrogate loss**:
$$ f_\epsilon(\theta) = \mathbb{E}_{z}[f(\theta + \epsilon z)] = f(\theta) + \frac{\epsilon^2}{2} \operatorname{Tr}(\nabla^2 f(\theta)) + \mathcal{O}(\epsilon^4) $$

The $\operatorname{Tr}(\nabla^2 f)$ penalty means ZO optimization inherently acts as a Sharpness-Aware Minimizer, driving parameters toward flat minima that generalize better. Throughout this document, $\epsilon$ denotes the perturbation radius.

## 3. Subspace Design: PiSSA Initialization with Activation Refinement

Under strict convexity, ZO iterations scale as $T = \mathcal{O}(d^4/\epsilon^2)$ — intractable for raw LLM layers ($d_{raw} \approx 6.7 \times 10^7$).

**The Subspace Solution (PiSSA):** Decompose $W_0$ via SVD and constrain optimization to low-rank adapters $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times k}$:
$$d_{eff} = (d \times r) + (r \times k) = 262{,}144 \quad \text{(at rank 16, } d = k = 8192\text{)}$$

**Convergence under Polyak-Lojasiewicz (PL) Conditions:** LLM loss landscapes are non-convex, but locally satisfy PL conditions. The convergence bound becomes:
$$ T = \mathcal{O}\left(\frac{d \cdot V}{\mu^2 \epsilon}\right) $$

Because PiSSA initializes adapters aligned with the principal singular components, the local PL constant $\mu$ is favorable and estimator variance $V$ is bounded along the trajectory, permitting empirical convergence in $\mathcal{O}(10^4)$ steps.

**Subspace quality.** For any linear layer with weight $W_l$ and input activations $H_l$, the gradient factorizes as $\nabla_{W_l} f = Q_l H_l^T$, confining its row-space to $\text{col}(H_l)$ (Park et al., arXiv:2601.17261, Theorem 5.6). The optimal perturbation subspace is therefore data-dependent — the top-$r$ left singular vectors of $H_l$. PiSSA's SVD-initialized subspace approximates this through the weight's intrinsic geometry, giving non-trivial gradient subspace overlap:

$$ \frac{\mathbb{E}[\cos(\hat{g}_{\text{PiSSA}}, g)]}{\mathbb{E}[\cos(\hat{g}_{\text{isotropic}}, g)]} = \sqrt{\frac{d_{in}}{r}} \cdot \frac{\|G \cdot P_{\text{PiSSA}}\|_F}{\|G\|_F} $$

The $\sqrt{d_{in}/r}$ factor captures dimensional reduction ($\sqrt{4096/16} = 16\times$ for typical Transformer layers).

**Activation calibration.** DS-MeZO closes the remaining static-vs-optimal gap via periodic activation calibration: every $K_{calib}$ steps, a calibration batch runs through the model (outside vLLM, via HuggingFace `register_forward_hook()`) to extract per-layer activation bases $A_l$ via power iteration on $H_l H_l^T$. These bases guide the perturbation direction within PiSSA's trainable subspace (Section 6.1). PiSSA defines *which parameters* to optimize; activation calibration refines *which directions within that space* to perturb. The blended perturbation $z' = \alpha_{proj}(z \cdot A_l A_l^T) + (1 - \alpha_{proj})z$ retains an isotropic residual for exploration outside the calibrated subspace.

## 4. Three-Phase Decoupling

DS-MeZO uses Weight-Decomposed Low-Rank Adaptation (DoRA):
$$ W' = m \cdot \frac{W^{res} + AB}{\|W^{res} + AB\|_c} $$

Three independent constraints force a three-phase perturbation structure. The resulting design is a form of block coordinate descent (BCD) for ZO optimization (Park et al., arXiv:2501.19099, Theorem 3.3), with the block decomposition *derived from* DoRA's structure rather than chosen as an engineering convenience.

### 4.1 SNR Isolation (Tick vs Tock)

ZO gradient variance scales linearly with perturbation dimensionality. Magnitude $m$ ($\sim 8$K dims) has $\sim 32\times$ higher SNR than direction $A, B$ ($\sim 131$K dims each). Simultaneous perturbation drowns the magnitude signal. Therefore:

- **Tick Phase:** Perturbs $m$ exclusively.
- **Tock Phase:** Perturbs $A$ or $B$ exclusively.

DoRA exhibits a negative correlation ($-0.31$) between magnitude and directional changes during fine-tuning, further supporting staggered updates. The unequal block sizes with equal cycle time means magnitude receives $\sim 16\times$ more updates per dimension — an implicit importance weighting that exploits the higher-SNR block.

### 4.2 Bilinear Symmetry (Tock-A vs Tock-B)

The symmetric ZO estimator requires midpoint $= \theta_0$. Jointly perturbing $A$ and $B$:
$$ (A + Z_A)(B + Z_B) = AB + (AZ_B + Z_AB) + Z_AZ_B $$
$$ (A - Z_A)(B - Z_B) = AB - (AZ_B + Z_AB) + Z_AZ_B $$
$$ \text{Midpoint} = AB + Z_AZ_B \neq AB $$

The cross-term $Z_AZ_B$ shifts the midpoint by $\mathcal{O}(\epsilon^2)$, injecting per-step variance. Blocks must be chosen to eliminate higher-order cross-terms in the SPSA estimator's midpoint. To restore symmetry:

- **Tock-A:** Freeze $B$, perturb $A \pm Z_A$. Midpoint $= AB$. $\checkmark$
- **Tock-B:** Freeze $A$, perturb $B \pm Z_B$. Midpoint $= AB$. $\checkmark$

### 4.3 Phase Ordering and Convergence

Phase order is randomized via cyclic random permutation: at each 3-step cycle, randomly shuffle $\{\text{Tick}, \text{Tock-A}, \text{Tock-B}\}$, then visit each once. This preserves the marginal distribution $P(p) = 1/3$ while avoiding pathological ordering effects (+3.3% on RTE vs fixed ascending order in BCD ablations, Park et al.).

**Convergence.** The three-phase structure satisfies BCD convergence (Theorem 3.3, Park et al.):
$$ \mathbb{E}[\|\nabla L(\theta_t)\|^2] \leq \mathcal{O}\left(\frac{r^2}{\bar{\rho} \cdot T} + \frac{s^2}{d \cdot T} + \frac{\Delta}{\alpha \cdot T} + \frac{\sigma^2}{B}\right) $$
where $\bar{\rho}$ is the mean subspace alignment. DS-MeZO extends this with a non-uniform subspace alignment analysis under LOREN's element-wise adaptive $\epsilon$ (Section 6.2), which concentrates perturbation energy on high-gradient directions and improves $\bar{\rho}$ relative to the uniform binary case.

## 5. Trajectory Locking and RLOO Contrastive Optimization

### 5.1 Trajectory Locking

Perturbing weights during autoregressive generation causes divergent token sequences and infinite variance. DS-MeZO locks trajectories:

1. Generate $N$ candidates under unperturbed weights $\theta_0$.
2. Select winner ($Y_w$) and loser ($Y_l$) via RLOO (Section 5.2).
3. Score these **fixed sequences** (prefill-only) under $\theta^+$ and $\theta^-$.

The loss becomes a continuous, differentiable function of weights.

### 5.2 RLOO Multi-Trajectory Baseline

Given $N$ candidates with rewards $\{R_1, \ldots, R_N\}$, the RLOO advantage for trajectory $i$:
$$ A_i = R_i - \frac{1}{N-1} \sum_{j \neq i} R_j $$

RLOO is unbiased, minimum-variance, and self-centering ($\sum_i A_i = 0$). It has zero lag and no tunable parameters.

### 5.3 Per-Token KL-Shaped Contrastive Loss

$$ \mathcal{L}(\theta) = \sum_{i \in \{w, l\}} A_i \cdot \text{NLL}_i^{KL}(\theta) $$

where:
$$ \text{NLL}_i^{KL}(\theta) = \frac{1}{T_i} \sum_{t=1}^{T_i} \left[ -(1 + \beta) \log \pi_\theta(y_t | y_{<t}) + \beta \log \pi_{ref}(y_t | y_{<t}) \right] $$

The linear form (not sigmoid DPO) avoids gradient underflow when $\sigma' \approx 0$. Per-token KL provides a dense drift penalty preventing mode collapse. The reference terms $\log \pi_{ref}$ are constant w.r.t. $\theta$ and cancel in the finite difference, but stabilize absolute loss magnitude for health monitoring.

### 5.4 Safety Mechanisms

1. **Per-Step Spike Detection:** Track an EMA of the NLL. If a batch causes NLL $> 5\times$ EMA, skip the step. Update the EMA *only after* a healthy step passes the check.
2. **Reward Rejection:** Skip if best reward $< R_{min}$ or reward gap $< \delta$.

## 6. Perturbation Construction: Activation Projection + LOREN Scaling

### 6.1 Activation-Guided Projection

The activation bases extracted in Section 3 are used to project the random perturbation noise toward the data-dependent gradient subspace:
$$ z' = \alpha_{proj} \cdot (z \cdot A_l A_l^T) + (1 - \alpha_{proj}) \cdot z $$

This projection applies to the $B$ matrix (whose gradient row-space lies directly in $\text{col}(H_l)$) and to the $A$ matrix (weaker but positive benefit via $H_l^T B^T$). The activation subspace drifts slowly under ZO updates ($\eta \approx 10^{-5}$), so a refresh every $K_{calib}$ steps is sufficient.

### 6.2 LOREN Element-Wise Scaling

PiSSA initializes adapters with large singular values (e.g., $32.0$). In FP16, the ULP for $32.0$ is $\approx 0.03$. A scalar $\epsilon \approx 2.5 \times 10^{-4}$ would be truncated to zero — "Silent Gradient Death."

**Solution:** After activation projection, LOREN applies element-wise layer-adaptive scaling. For layer $l$ (zero-indexed):
$$ \epsilon_{l, ij} = \max\!\left(\frac{\epsilon_{base}}{\sqrt{l + 1}},\; |\theta_{ij}| \times 0.001\right) $$

The final perturbation is $Z = E \odot z'$ where $E$ is the element-wise epsilon tensor and $z'$ is the activation-projected noise. The depth scaling creates a trust region that tightens with depth, preventing deep-layer perturbations from causing catastrophic output drift. The magnitude-proportional floor guarantees FP16 survival.

**Unbiasedness Proof:** A non-uniform perturbation $\Delta = E \odot Z$ preserves SPSA unbiasedness when divided element-wise:
$$ \hat{g}_i = \frac{\mathcal{L}(\theta + \Delta) - \mathcal{L}(\theta - \Delta)}{2\epsilon_i} \cdot z_i, \quad z \sim \mathcal{N}(0, I) $$
$$ \mathbb{E}[\hat{g}_i] = \sum_j \frac{\partial \mathcal{L}}{\partial \theta_j} \frac{\epsilon_j}{\epsilon_i} \mathbb{E}[z_j z_i] = \frac{\partial \mathcal{L}}{\partial \theta_i} $$

**Implicit Regularization Shift:** Element-wise $\epsilon_i$ modifies the smoothed objective to $f_E(\theta) = f(\theta) + \frac{1}{2}\sum_i \epsilon_i^2 \frac{\partial^2 f}{\partial \theta_i^2}$, biasing the optimizer toward flatness in high-magnitude directions.

**Non-uniform subspace alignment.** Standard BCD theory (Section 4.3) characterizes expected subspace alignment $\mathbb{E}[\rho]$ only for binary orthogonal projections ($M^2 = M$). LOREN's element-wise $\epsilon_i$ produces a non-uniform diagonal perturbation matrix $M = \text{diag}(\epsilon_1/\epsilon_{\max}, \ldots, \epsilon_d/\epsilon_{\max})$ within each active block. The extended subspace alignment is:

$$ \mathbb{E}[\rho_{\text{LOREN}}] = \frac{\sum_i (\epsilon_i / \epsilon_{\max})^2 \cdot H_{ii}}{\lambda_{\max}(H) \cdot \text{srank}(M)}$$

where $H$ is the local Hessian upper bound. Because LOREN's magnitude-proportional floor sets $\epsilon_i \propto |\theta_i|$ for large parameters, and gradient energy concentrates on high-magnitude parameters, the numerator is weighted toward high-gradient directions — improving subspace alignment relative to the uniform binary case.

## 7. Noise-Calibrated Tensor Adam with FP32 Accumulation

### 7.1 Noise Calibration

At 70B scale with NF4 residuals and FP16 forward passes, quantization noise interacts with perturbation noise. Standard ZO methods collapse under reduced precision (e.g., HiZOO drops from 69.48% to 55.58% on OPT-13B under bf16; Yao et al., arXiv:2601.01452). DS-MeZO addresses this with a per-phase noise tracker using prediction residuals — how much the new gradient surprises Adam's current first-moment estimate:

$$ r_t = \hat{g}_t - m_{t-1} \quad \text{(prediction residual vs. Adam's running mean)} $$
$$ \sigma_e^2 \leftarrow (1 - \alpha_n) \cdot \sigma_e^2 + \alpha_n \cdot \overline{r_t^2} $$
$$ \gamma = \frac{\sigma_s^2}{\sigma_s^2 + \sigma_e^2} \in (0, 1) $$

where $\sigma_s^2$ is the signal prior (initialized from early-step gradient magnitude) and $\alpha_n$ is the noise EMA decay. The gradient entering Adam is $\hat{g}_t^{cal} = \gamma \cdot \hat{g}_t$.

Under high noise (FP16 quantization distortion, NF4 residual error), $\sigma_e^2$ grows, $\gamma$ shrinks toward 0, and updates are automatically damped. Under low noise, $\gamma \to 1$ and the calibration is transparent. Each (layer, phase) pair maintains its own $\sigma_e^2$ tracker, preserving Tick/Tock independence.

### 7.2 Tensor Adam Update

The noise-calibrated gradient $\hat{g}_t^{cal}$ is fed to standard Adam:
$$ m_{t, i} = \beta_1 m_{t-1, i} + (1 - \beta_1) \hat{g}_{t, i}^{cal} $$
$$ v_{t, i} = \beta_2 v_{t-1, i} + (1 - \beta_2) (\hat{g}_{t, i}^{cal})^2 $$
$$ \Delta \theta_{t, i} = \frac{\eta}{\sqrt{\hat{v}_{t, i}} + \delta} \cdot \hat{m}_{t, i} $$

**Adam-ZO Synergy:** Under element-wise $\epsilon_i$, the second moment converges to $\mathbb{E}[\hat{g}_i^2] = 3g_i^2 + C/\epsilon_i^2$. The error term dominates, so Adam's effective step scales as $\Delta\theta_i \propto \epsilon_i \cdot g_i$ — updates proportional to parameter magnitude, naturally resisting FP16 truncation. Noise calibration ensures $v_{t,i}$ converges to true gradient variance rather than noise-inflated variance.

**FP32 Master Accumulation:** Updates are $\approx 10^{-5}$. Master weights and Adam moments are stored in **FP32** on the host to prevent truncation during accumulation. Forward passes remain FP16.

---

## 8. System Architecture

DoRA decomposition ($m$, $A$, $B$) lives entirely in the Controller's optimization logic. vLLM does not natively support DoRA (as of v0.16.x), so the Controller fuses DoRA into standard LoRA format before each vLLM interaction: it computes $W_{eff} = m \cdot \frac{W^{res} + AB}{\|W^{res} + AB\|_c}$, extracts the delta $\Delta W = W_{eff} - W_{base}$, and re-decomposes it into a rank-$r$ LoRA pair $(A_{fused}, B_{fused})$ via truncated SVD. These fused LoRA weights are serialized to disk in PEFT-compatible format and loaded into vLLM via the dynamic LoRA loading API.

**Scoring** uses vLLM's `generate()` with `SamplingParams(max_tokens=1, prompt_logprobs=0)` to extract per-prompt-token logprobs on fixed sequences (prefill-only). vLLM has no dedicated scoring API for generative models; `prompt_logprobs` is the correct mechanism.

```
+----------------------------------------------------------+
|                    DS-MeZO CONTROLLER                    |
|                  (Python / PyTorch / CPU)                 |
|                                                          |
|  +------------------+  +------------------+  +---------+ |
|  |  FP32 Master     |  |  Noise-Calibrated|  | Activ.  | |
|  |  Weights         |  |  Tensor Adam     |  | Calib.  | |
|  |  (m, A, B)       |  |  (m1, v2, gamma  |  | Cache   | |
|  |  DoRA + PiSSA    |  |   per phase)     |  | (A_l)   | |
|  +--------+---------+  +--------+---------+  +---------+ |
|           +----------------+----------------+            |
|                            |                             |
|             +--------------v--------------+              |
|             |  DoRA-to-LoRA Fusion       |              |
|             |  + PEFT Serialization      |              |
|             |  + Dynamic Adapter Reload  |              |
|             +--------------+--------------+              |
+----------------------------+-----------------------------+
                             |  Serialize to disk + load_lora_adapter
                             |  (load_inplace=true)
                             v
+----------------------------------------------------------+
|                    vLLM ENGINE                           |
|              (GPU / Standard LoRA Mode)                   |
|                                                          |
|  +--------------+  +-----------+  +------------------+  |
|  |  Base Model   |  |  S-LoRA   |  |  PagedAttention  |  |
|  |  W_res (NF4)  |  |  Paging   |  |  + KV Cache      |  |
|  +--------------+  +-----------+  +------------------+  |
|                                                          |
|  Output: prompt_logprobs, generated text                  |
+----------------------------------------------------------+
                             |
              +--------------+---------------+
              v                              v
+--------------------------+  +---------------------------+
|   EXPLORATION (once)     |  | OPTIMIZATION (3 phases)   |
|                          |  |                           |
|  generate(th_0, n=N,T>0) |  |  Cyclic random permute:   |
|  RLOO advantages A_i     |  |  {Tick, Tock-A, Tock-B}   |
|  R: code_compiles() ...  |  |  (prefill via prompt_     |
|                          |  |   logprobs, batch=4)      |
+--------------------------+  +---------------------------+
```

**Environment requirement:** `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` must be set to enable dynamic adapter reloading at runtime.

## 9. Execution Workflow

### 9.1 Initialization

1. **Load Backbone:** Load frozen model into vLLM in NF4 quantization via BitsAndBytes.
2. **Compute SVD:** Fast SVD (Halko et al., `niter=2`) of target layer weights:
   $$U, S, V^T = \text{FastSVD}(W_0)$$
3. **Initialize Adapters:**
   - Direction: $A = U[:, :r] \cdot \sqrt{S[:r]}$, $B = \sqrt{S[:r]} \cdot V^T[:r, :]$ (symmetric splitting for gradient balance)
   - Residual (frozen): $W^{res} = W_0 - AB$, quantized to NF4 (~20% less error than quantizing $W_0$ directly)
   - Magnitude: $m = \|W_0\|_c$ (column-wise norms)
4. **Create Adapter Staging Directories:** Create two directories on local storage (`/tmp/ds_mezo/adapter_pos/`, `/tmp/ds_mezo/adapter_neg/`). Fuse initial DoRA weights into standard LoRA format and serialize to PEFT-compatible safetensors. Load both adapters into vLLM via `POST /v1/load_lora_adapter`.
5. **Calibrate Activation Bases:** Run initial calibration batch through the model (outside vLLM) to extract per-layer activation bases $A_l$ via power iteration (Section 3).
6. **Initialize FP32 State:** Store $m$, $A$, $B$ in FP32 on CPU. Initialize per-layer, per-phase Tensor Adam moments and noise calibration state to zero.

### 9.2 Per-Step Loop

For each training step:

1. **Explore:** Generate $N$ candidates with unperturbed weights via `LLM.generate()` with `LoRARequest`. Score with reward function, compute RLOO advantages, select winner/loser. Reject if quality thresholds not met. Cache reference logprobs via `generate()` with `SamplingParams(max_tokens=1, prompt_logprobs=0)`.
2. **Optimize (per layer, per phase):** For each layer $l$ and each phase $p$ in the current cycle's random permutation:
   - Construct activation-projected, LOREN-scaled perturbation $Z, E$ (Section 6)
   - Fuse perturbed DoRA weights ($\theta_p \pm Z$) into LoRA format, serialize to disk, reload adapters with `load_inplace=true`
   - Extract prompt logprobs for $Y_w, Y_l$ under both perturbations via `generate()` with `SamplingParams(max_tokens=1, prompt_logprobs=0)`
   - Health check — skip if NLL spike detected
   - Element-wise SPSA gradient: $\hat{g}_i = \frac{\mathcal{L}^+ - \mathcal{L}^-}{2\epsilon_i} \cdot z_i$
   - Noise-calibrated Tensor Adam update on FP32 master weights (Section 7)
   - Accumulate $\|\hat{g}\|_2$ for rank reallocation
3. **Maintain:** Dynamic rank reallocation check (every 200 steps). Activation calibration refresh (every $K_{calib}$ steps).

| Phase | Target | Frozen | Dims | Adam Group |
| :--- | :--- | :--- | :--- | :--- |
| Tick | $m$ | $A, B$ | $\sim 8$K | `tick` |
| Tock-A | $A$ | $m, B$ | $\sim 131$K | `tock_A` |
| Tock-B | $B$ | $m, A$ (at updated $A_{t+1}$) | $\sim 131$K | `tock_B` |

## 10. Implementation

### 10.1 Adapter Sync Protocol

vLLM does not expose an in-memory weight update API. DS-MeZO uses filesystem-backed adapter synchronization:

1. **DoRA-to-LoRA fusion:** The Controller computes the effective weight $W_{eff}$ from DoRA parameters $(m, A, B, W^{res})$, extracts the delta $\Delta W = W_{eff} - W_{base}$, and decomposes it into a rank-$r$ LoRA pair via truncated SVD.
2. **Serialize:** Write the fused LoRA matrices to disk in PEFT-compatible safetensors format (two staging directories, one per adapter slot).
3. **Reload:** Call `POST /v1/load_lora_adapter` with `load_inplace: true` to hot-swap the adapter without deregistering it. Requires `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`.

**Latency:** ~50-200ms per adapter sync (dominated by disk I/O for serialization). For 2 adapters per phase × 3 phases: ~300ms-1.2s per step. Using tmpfs (RAM-backed `/tmp`) reduces serialization to ~10-30ms per adapter, bringing total overhead to ~60-180ms per step.

**Scoring via prompt_logprobs:** vLLM's `LLM.score()` is a cross-encoder/reranking API, not applicable to generative models. DS-MeZO scores fixed sequences by calling `LLM.generate()` with `SamplingParams(max_tokens=1, prompt_logprobs=0)`, which triggers a prefill pass and returns per-token logprobs without autoregressive generation. The single generated token is discarded.

### 10.2 Complete Implementation

```python
import torch
import random
import os
import json
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from safetensors.torch import save_file


# -- Adapter I/O Helpers --------------------------------------------------

ADAPTER_STAGING_DIR = os.environ.get("DS_MEZO_ADAPTER_DIR", "/tmp/ds_mezo")


def fuse_dora_to_lora(m, A, B, W_res, W_base, rank):
    """Fuse DoRA (m, A, B) into standard LoRA (A_fused, B_fused).

    Computes W_eff = m * normalize(W_res + A@B), extracts delta vs W_base,
    and decomposes into rank-r LoRA pair via truncated SVD.
    """
    direction = W_res.float() + A.float() @ B.float()
    col_norms = direction.norm(dim=0, keepdim=True).clamp(min=1e-12)
    W_eff = m.float().unsqueeze(0) * (direction / col_norms)
    delta = W_eff - W_base.float()
    U, S, Vt = torch.svd_lowrank(delta, q=rank, niter=2)
    sqrt_S = torch.sqrt(S)
    A_fused = (U * sqrt_S.unsqueeze(0)).half()   # [d, r]
    B_fused = (sqrt_S.unsqueeze(1) * Vt).half()  # [r, k]
    return A_fused, B_fused


def save_peft_adapter(A_fused, B_fused, adapter_dir, layer_configs):
    """Serialize fused LoRA weights to PEFT-compatible safetensors format."""
    os.makedirs(adapter_dir, exist_ok=True)
    tensors = {}
    for layer_idx, (A_l, B_l) in enumerate(zip(A_fused, B_fused)):
        prefix = layer_configs[layer_idx]['peft_prefix']
        tensors[f"{prefix}.lora_A.weight"] = A_l
        tensors[f"{prefix}.lora_B.weight"] = B_l
    save_file(tensors, os.path.join(adapter_dir, "adapter_model.safetensors"))

    config = {
        "peft_type": "LORA",
        "r": A_fused[0].shape[1],
        "lora_alpha": A_fused[0].shape[1],  # alpha = r (no scaling)
        "target_modules": [cfg['target_module'] for cfg in layer_configs],
    }
    with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f)


def extract_prompt_logprobs(output):
    """Extract per-token logprobs from a vLLM output with prompt_logprobs."""
    logprobs = []
    if output.prompt_logprobs is None:
        return logprobs
    # prompt_logprobs[0] is None (no preceding token for first token)
    for token_lp in output.prompt_logprobs[1:]:
        if token_lp is not None:
            # Each entry maps token_id -> Logprob; get the actual token's logprob
            for tok_id, lp in token_lp.items():
                logprobs.append(lp.logprob)
                break
    return logprobs


# -- Main Controller -------------------------------------------------------

class DSMeZO_Controller:
    def __init__(self, vllm_engine, model_config, score_fn):
        self.engine = vllm_engine  # vllm.LLM instance
        self.score_fn = score_fn   # Callable: str -> float in [0, 1]
        self.step_count = 0
        self.num_layers = model_config.num_target_layers

        # PiSSA initialization — per-layer master weights stored in FP32
        self.layers = []
        for layer_idx in range(self.num_layers):
            m, A, B, W_res, W_base = initialize_pissa(model_config, layer_idx)
            self.layers.append({
                'm': m.float(), 'A': A.float(), 'B': B.float(),
                'W_res': W_res, 'W_base': W_base,
                'layer_idx': layer_idx,
                'rank': model_config.rank,
                'peft_prefix': model_config.peft_prefix(layer_idx),
                'target_module': model_config.target_module(layer_idx),
            })

        # Adapter staging directories and LoRARequest objects
        self.adapter_dir_pos = os.path.join(ADAPTER_STAGING_DIR, "adapter_pos")
        self.adapter_dir_neg = os.path.join(ADAPTER_STAGING_DIR, "adapter_neg")
        self.lora_pos = LoRARequest("adapter_pos", 1, self.adapter_dir_pos)
        self.lora_neg = LoRARequest("adapter_neg", 2, self.adapter_dir_neg)

        # Initial adapter serialization + registration
        self._sync_adapters_to_vllm()

        # Scoring params: prefill-only via prompt_logprobs
        self.score_params = SamplingParams(
            max_tokens=1, prompt_logprobs=0, temperature=0.0
        )

        # Exploration & selection
        self.num_candidates = 4
        self.explore_temperature = 0.7
        self.reward_threshold = 0.1
        self.contrastive_gap = 0.1

        # Per-token KL penalty
        self.beta_kl = 0.1
        self.ref_logprobs = None

        # Noise-calibrated Tensor Adam hyperparameters (Section 7)
        self.eta = 1e-5
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_delta = 1e-8
        self.noise_alpha = 0.1  # EMA decay for noise tracking

        # Per-layer, per-group state: Adam moments + noise calibration (FP32)
        self.moments = {}     # key: (layer_idx, group) -> {m1, v2, step}
        self.noise_var = {}   # key: (layer_idx, group) -> sigma_e^2
        self.signal_var = {}  # key: (layer_idx, group) -> sigma_s^2

        # Element-wise perturbation base epsilon
        self.eps_base = 2.5e-4

        # Activation-guided perturbation (Section 6.1)
        self.activation_bases = {}  # layer_idx -> [d_in, r_calib]
        self.calib_interval = 1000
        self.calib_blend = 0.8      # alpha_proj
        self._calibrate_activation_bases(model_config)  # initial calibration

        # Per-layer dynamic rank allocation
        self.rank_realloc_interval = 200
        self.rank_min = 4
        self.rank_max = 64
        self.grad_magnitude_accum = {i: 0.0 for i in range(self.num_layers)}

        # Divergence detection
        self.loss_ema = None
        self.loss_ema_momentum = 0.95
        self.max_loss_ratio = 5.0

        # Phase ordering (cyclic random permutation, Section 4.3)
        self._phase_order = [('m', 'tick'), ('A', 'tock_A'), ('B', 'tock_B')]

    # -- Activation Calibration --------------------------------------------

    def _calibrate_activation_bases(self, model_config=None):
        """Extract per-layer activation bases via power iteration (Section 3).

        Runs a calibration batch through the model outside vLLM using
        HuggingFace Transformers with forward hooks. Called at init and
        every calib_interval steps.
        """
        calib_model = load_model_for_calibration(model_config, self.layers)
        calib_data = get_calibration_batch(model_config, n_samples=64)

        hooks = []
        activation_cache = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                H = input[0].reshape(-1, input[0].shape[-1])  # [batch*seq, d_in]
                # Power iteration: 3 steps on H @ H.T
                v = torch.randn(H.shape[1], device=H.device)
                for _ in range(3):
                    v = H.T @ (H @ v)
                    v = v / (v.norm() + 1e-12)
                activation_cache[layer_idx] = v.cpu().float()
            return hook_fn

        for layer_idx in range(self.num_layers):
            h = get_target_module(calib_model, layer_idx).register_forward_hook(
                make_hook(layer_idx)
            )
            hooks.append(h)

        with torch.no_grad():
            calib_model(calib_data)

        for h in hooks:
            h.remove()

        self.activation_bases = activation_cache
        del calib_model

    # -- DoRA-to-LoRA Fusion + Adapter Sync --------------------------------

    def _fuse_all_layers(self, overrides=None):
        """Fuse DoRA parameters into LoRA format for all layers."""
        A_fused_list, B_fused_list = [], []
        for layer in self.layers:
            idx = layer['layer_idx']
            m = overrides.get((idx, 'm'), layer['m']) if overrides else layer['m']
            A = overrides.get((idx, 'A'), layer['A']) if overrides else layer['A']
            B = overrides.get((idx, 'B'), layer['B']) if overrides else layer['B']
            A_f, B_f = fuse_dora_to_lora(
                m, A, B, layer['W_res'], layer['W_base'], layer['rank']
            )
            A_fused_list.append(A_f)
            B_fused_list.append(B_f)
        return A_fused_list, B_fused_list

    def _sync_adapters_to_vllm(self, pos_overrides=None, neg_overrides=None):
        """Fuse DoRA -> LoRA, serialize to disk, reload into vLLM."""
        # Fuse and serialize pos adapter
        A_pos, B_pos = self._fuse_all_layers(pos_overrides)
        save_peft_adapter(A_pos, B_pos, self.adapter_dir_pos, self.layers)

        # Fuse and serialize neg adapter (same as pos if no neg overrides)
        if neg_overrides is not None:
            A_neg, B_neg = self._fuse_all_layers(neg_overrides)
        else:
            A_neg, B_neg = A_pos, B_pos
        save_peft_adapter(A_neg, B_neg, self.adapter_dir_neg, self.layers)

        # Reload into vLLM with in-place swap
        self.engine.load_lora_adapter(
            self.adapter_dir_pos, "adapter_pos", load_inplace=True
        )
        self.engine.load_lora_adapter(
            self.adapter_dir_neg, "adapter_neg", load_inplace=True
        )

    def _write_adapters(self, pos_overrides, neg_overrides):
        """Fuse perturbed DoRA weights to LoRA, sync to vLLM."""
        self._sync_adapters_to_vllm(pos_overrides, neg_overrides)

    # -- Scoring via prompt_logprobs ---------------------------------------

    def _get_prompt_logprobs(self, token_sequences, lora_request):
        """Score fixed sequences by extracting prompt_logprobs via generate().

        vLLM's score() is for cross-encoder models only. For generative models,
        we use generate() with max_tokens=1 and prompt_logprobs=0 to trigger
        a prefill pass that returns per-token logprobs. The single generated
        token is discarded.
        """
        prompts = [{"prompt_token_ids": seq} for seq in token_sequences]
        outputs = self.engine.generate(
            prompts,
            sampling_params=self.score_params,
            lora_request=lora_request,
        )
        return [extract_prompt_logprobs(out) for out in outputs]

    def _score_contrastive_rloo(self, trajectories, advantages):
        """Score winner/loser under pos/neg adapters with per-token KL shaping."""
        winner_tokens, loser_tokens = trajectories
        adv_w, adv_l = advantages

        # Score under pos adapter
        lp_pos = self._get_prompt_logprobs(
            [winner_tokens, loser_tokens], self.lora_pos
        )
        # Score under neg adapter
        lp_neg = self._get_prompt_logprobs(
            [winner_tokens, loser_tokens], self.lora_neg
        )

        def compute_kl_shaped_nll(logprobs, ref_logprobs):
            """Per-token KL-shaped NLL: -(1+beta)*log_pi + beta*log_ref."""
            nll_tokens = [-lp for lp in logprobs]
            if self.beta_kl > 0 and ref_logprobs is not None:
                kl_tokens = [ref - lp for ref, lp in
                             zip(ref_logprobs, logprobs)]
                shaped = [nll + self.beta_kl * kl
                          for nll, kl in zip(nll_tokens, kl_tokens)]
                return sum(shaped) / len(shaped)
            return sum(nll_tokens) / len(nll_tokens)

        ref_w = self.ref_logprobs[0] if self.ref_logprobs else None
        ref_l = self.ref_logprobs[1] if self.ref_logprobs else None

        nll_pos_w = compute_kl_shaped_nll(lp_pos[0], ref_w)
        nll_pos_l = compute_kl_shaped_nll(lp_pos[1], ref_l)
        nll_neg_w = compute_kl_shaped_nll(lp_neg[0], ref_w)
        nll_neg_l = compute_kl_shaped_nll(lp_neg[1], ref_l)

        loss_pos = adv_w * nll_pos_w + adv_l * nll_pos_l
        loss_neg = adv_w * nll_neg_w + adv_l * nll_neg_l
        return loss_pos, loss_neg

    # -- Perturbation: Activation Projection + LOREN Scaling ---------------

    def get_perturbation(self, param_tensor, base_epsilon, layer_idx,
                         target_key):
        """Activation-projected, LOREN-scaled perturbation (Section 6).

        Stage 1: Project random noise onto activation subspace blend.
        Stage 2: Apply element-wise layer-adaptive epsilon.
        """
        z = torch.randn_like(param_tensor)

        # Stage 1: Activation-guided projection (Section 6.1)
        if layer_idx in self.activation_bases:
            A_l = self.activation_bases[layer_idx]
            A_l = A_l.to(device=param_tensor.device)
            z_proj = z @ A_l.unsqueeze(1) @ A_l.unsqueeze(0)
            z = self.calib_blend * z_proj + (1 - self.calib_blend) * z

        # Stage 2: LOREN element-wise scaling (Section 6.2)
        layer_eps = base_epsilon / (layer_idx + 1) ** 0.5
        fp16_ulp_margin = param_tensor.abs() * 0.001
        dynamic_epsilon = torch.max(
            torch.tensor(layer_eps, device=param_tensor.device),
            fp16_ulp_margin
        )
        return dynamic_epsilon * z, dynamic_epsilon

    # -- Noise-Calibrated Tensor Adam Update -------------------------------

    def _noise_calibrated_adam_update(self, param, grad, layer_idx, group):
        """Noise calibration + Tensor Adam as a single unit (Section 7).

        Stage 1: Compute gamma from prediction residuals, scale gradient.
        Stage 2: Feed calibrated gradient to Adam with FP32 accumulation.
        """
        key = (layer_idx, group)

        # --- Stage 1: Noise calibration (Section 7.1) ---
        if key not in self.signal_var:
            # First step: initialize signal prior, no calibration
            self.signal_var[key] = grad.square().mean().item()
            self.noise_var[key] = self.signal_var[key]
        else:
            # Prediction residual against Adam's running mean
            if key in self.moments:
                residual_sq = (grad - self.moments[key]['m1']).square().mean().item()
            else:
                residual_sq = grad.square().mean().item()

            # EMA update of noise variance
            self.noise_var[key] = ((1 - self.noise_alpha) * self.noise_var[key]
                                   + self.noise_alpha * residual_sq)

            # Shrinkage: reduce noise before Adam processes it
            sigma_s = self.signal_var[key]
            sigma_e = self.noise_var[key]
            gamma = sigma_s / (sigma_s + sigma_e + 1e-12)
            grad = gamma * grad

        # --- Stage 2: Tensor Adam with FP32 accumulation (Section 7.2) ---
        if key not in self.moments:
            self.moments[key] = {
                'm1': torch.zeros_like(param),
                'v2': torch.zeros_like(param),
                'step': 0,
            }

        state = self.moments[key]
        state['step'] += 1
        step = state['step']

        m1 = self.adam_beta1 * state['m1'] + (1 - self.adam_beta1) * grad
        v2 = self.adam_beta2 * state['v2'] + (1 - self.adam_beta2) * grad.square()
        state['m1'] = m1
        state['v2'] = v2

        bc1 = 1 - self.adam_beta1 ** step
        bc2 = 1 - self.adam_beta2 ** step
        m1_hat = m1 / bc1
        v2_hat = v2 / bc2

        param -= self.eta / (v2_hat.sqrt() + self.adam_delta) * m1_hat

    # -- Exploration -------------------------------------------------------

    def _explore(self, batch):
        """Generate candidates, compute RLOO advantages, select winner/loser."""
        # Sync unperturbed weights to pos adapter for generation
        self._sync_adapters_to_vllm()

        gen_params = SamplingParams(
            n=self.num_candidates, temperature=self.explore_temperature
        )
        outputs = self.engine.generate(
            batch, sampling_params=gen_params, lora_request=self.lora_pos
        )

        scored = [(out, self.score_fn(out.text)) for out in outputs]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_output, best_reward = scored[0]
        worst_output, worst_reward = scored[-1]

        if best_reward < self.reward_threshold:
            return None
        if (best_reward - worst_reward) < self.contrastive_gap:
            return None

        # RLOO advantages (Section 5.2)
        all_rewards = [r for _, r in scored]
        total_reward = sum(all_rewards)

        adv_w = best_reward - (total_reward - best_reward) / (len(scored) - 1)
        adv_l = worst_reward - (total_reward - worst_reward) / (len(scored) - 1)

        # Cache reference logprobs for per-token KL shaping
        if self.beta_kl > 0:
            ref_lp = self._get_prompt_logprobs(
                [best_output.token_ids, worst_output.token_ids],
                self.lora_pos
            )
            self.ref_logprobs = (ref_lp[0], ref_lp[1])
        else:
            self.ref_logprobs = None

        return ((best_output.token_ids, worst_output.token_ids),
                (adv_w, adv_l))

    # -- Health Monitoring -------------------------------------------------

    def _check_health(self, loss_pos, loss_neg):
        """Skip update if NLL exceeds threshold. Update EMA only on healthy steps."""
        avg_nll = (abs(loss_pos) + abs(loss_neg)) / 2
        if self.loss_ema is None:
            self.loss_ema = avg_nll
            return True
        if self.loss_ema > 1e-8 and avg_nll > self.max_loss_ratio * self.loss_ema:
            return False
        # Update EMA only after passing health check
        self.loss_ema = (self.loss_ema_momentum * self.loss_ema +
                         (1 - self.loss_ema_momentum) * avg_nll)
        return True

    # -- Dynamic Rank Allocation -------------------------------------------

    def _maybe_realloc_ranks(self):
        """Redistribute global rank budget based on accumulated gradient magnitude."""
        if self.step_count % self.rank_realloc_interval != 0:
            return

        magnitudes = [self.grad_magnitude_accum[i] / self.rank_realloc_interval
                      for i in range(self.num_layers)]
        total_mag = sum(magnitudes) + 1e-12
        total_rank = sum(layer['rank'] for layer in self.layers)

        new_ranks = {}
        for i, mag in enumerate(magnitudes):
            raw = round((mag / total_mag) * total_rank)
            new_ranks[i] = max(self.rank_min, min(self.rank_max, raw))

        allocated = sum(new_ranks.values())
        if allocated != total_rank:
            diff = total_rank - allocated
            top_layer = max(new_ranks, key=lambda i: magnitudes[i])
            new_ranks[top_layer] = max(self.rank_min,
                                        min(self.rank_max,
                                            new_ranks[top_layer] + diff))

        for i in range(self.num_layers):
            if new_ranks[i] != self.layers[i]['rank']:
                self.layers[i]['rank'] = new_ranks[i]
                self._reinit_layer_adapter(i, new_ranks[i])
                # Reset optimizer state for affected layer
                for group in ['tick', 'tock_A', 'tock_B']:
                    self.moments.pop((i, group), None)
                    self.noise_var.pop((i, group), None)
                    self.signal_var.pop((i, group), None)

        # Reset accumulators
        self.grad_magnitude_accum = {i: 0.0 for i in range(self.num_layers)}

    def _reinit_layer_adapter(self, layer_idx, new_rank):
        """Re-initialize adapter matrices for a layer at a new rank.

        Truncates (if shrinking) or pads via SVD (if growing) the existing
        A, B matrices to preserve learned information.
        """
        layer = self.layers[layer_idx]
        old_rank = layer['A'].shape[1]

        if new_rank <= old_rank:
            layer['A'] = layer['A'][:, :new_rank].contiguous()
            layer['B'] = layer['B'][:new_rank, :].contiguous()
        else:
            current_approx = layer['A'] @ layer['B']
            W_res_deq = dequantize_nf4(layer['W_res'])
            gap = (W_res_deq + current_approx) - current_approx
            U, S, Vt = torch.svd_lowrank(gap, q=new_rank - old_rank, niter=2)
            sqrt_S = torch.sqrt(S)
            A_ext = U * sqrt_S.unsqueeze(0)
            B_ext = sqrt_S.unsqueeze(1) * Vt
            layer['A'] = torch.cat([layer['A'], A_ext], dim=1)
            layer['B'] = torch.cat([layer['B'], B_ext], dim=0)

    # -- Main Training Step ------------------------------------------------

    def step(self, batch):
        self.step_count += 1

        # Cyclic random permutation of phase order (Section 4.3)
        if self.step_count % 3 == 1:
            random.shuffle(self._phase_order)

        # Periodic activation calibration refresh (Section 3)
        if (self.calib_interval > 0
                and self.step_count % self.calib_interval == 0):
            self._calibrate_activation_bases()

        # === EXPLORATION (RLOO) ===
        result = self._explore(batch)
        if result is None:
            return
        trajectories, advantages = result

        # === PER-LAYER OPTIMIZATION ===
        for layer in self.layers:
            idx = layer['layer_idx']

            for target_key, group in self._phase_order:
                Z, E = self.get_perturbation(
                    layer[target_key], self.eps_base, idx,
                    target_key=target_key
                )

                pos_overrides = {(idx, target_key): layer[target_key] + Z}
                neg_overrides = {(idx, target_key): layer[target_key] - Z}
                self._write_adapters(pos_overrides, neg_overrides)

                loss_pos, loss_neg = self._score_contrastive_rloo(
                    trajectories, advantages
                )

                if self._check_health(loss_pos, loss_neg):
                    diff = loss_pos - loss_neg
                    z = Z / E
                    grad = (diff / (2.0 * E)) * z
                    self._noise_calibrated_adam_update(
                        layer[target_key], grad, idx, group
                    )
                    self.grad_magnitude_accum[idx] += grad.norm().item()

        # Dynamic rank reallocation check
        self._maybe_realloc_ranks()

    def train(self, dataloader, num_steps):
        for step_idx, batch in zip(range(num_steps), dataloader):
            self.step(batch)


def initialize_pissa(model_config, layer_idx):
    """PiSSA initialization via Fast SVD (Halko et al.)."""
    W0 = load_pretrained_weights(model_config, layer_idx)
    r = model_config.rank

    U, S, Vt = torch.svd_lowrank(W0, q=r, niter=2)

    sqrt_S = torch.sqrt(S[:r])
    A = U[:, :r] * sqrt_S.unsqueeze(0)       # [d, r]
    B = sqrt_S.unsqueeze(1) * Vt[:r, :]       # [r, k]

    W_res = W0 - A @ B
    W_res_quantized = quantize_nf4(W_res)

    m = torch.norm(W0, dim=0)  # column-wise norms

    # W_base needed for DoRA-to-LoRA fusion (delta = W_eff - W_base)
    W_base = W0.clone()

    return m, A, B, W_res_quantized, W_base
```

## 11. Memory Budget (Single H100 80GB)

| Component | VRAM | Notes |
| :--- | :--- | :--- |
| Residual Model ($W^{res}$, NF4) | ~35 GB | Frozen. QPiSSA residual quantization via BitsAndBytes. |
| KV Cache (PagedAttention) | ~25 GB | Dynamic, batch-dependent. |
| LoRA Adapter Slots (2 slots, FP16) | ~0.8 GB | Always resident. Standard LoRA format (fused from DoRA). |
| Controller State (FP32 masters + Adam + noise calibration) | ~0.2 GB | Per-layer param groups + moment tensors + noise scalars. |
| Activation Calibration Cache | ~3.3 MB | Per-layer $d_{in}$-vector at $r_{calib}=1$. |
| **Headroom** | **~19 GB** | Safety margin for vLLM internals. |

Launch vLLM with `--gpu-memory-utilization 0.9` and `--enable-lora --max-lora-rank 64`.

**Quantization Drift Caveat:** NF4 quantization of $W^{res}$ introduces fixed error in the DoRA column-norm denominator $\|W^{res} + AB\|_c$. The Tick phase magnitude $m$ may over-compensate. Conservative $\eta$, Tensor Adam's per-coordinate scaling, and noise calibration's $\gamma$ dampening collectively mitigate this.

**DoRA Fusion Overhead:** The Controller stores $W_{base}$ per layer (~0.5 GB total at FP32 for 70B, CPU-resident) for computing fusion deltas. The truncated SVD for re-decomposition runs on CPU and takes <100ms per layer.

---

## 12. Hyperparameter Configuration

### 12.1 Scaling Rules

| Parameter | Formula | Rationale |
| :--- | :--- | :--- |
| Perturbation Radius | $\epsilon_{base} = 10^{-3}/\sqrt{r}$ | Smoother subspace at low rank permits larger perturbations. |
| Layer-Adaptive Scaling | $\epsilon_l = \epsilon_{base}/\sqrt{l + 1}$ | Trust region tightens with depth. |
| Smoothness Bound | $\epsilon^2 \leq \sqrt{2}L_1 / (r^{3/2}L_3)$ | Hard ceiling from Zhang et al. |
| Element-wise Override | $\epsilon_{l,ij} = \max(\epsilon_l, |\theta_{ij}| \times 0.001)$ | FP16 ULP survival. |

### 12.2 Default Values (Llama-3-70B)

| Parameter | Default | Notes |
| :--- | :--- | :--- |
| Subspace Rank ($r$) | 16 or 32 (initial; per-layer) | Dynamic rank allocation every 200 steps |
| Base Perturbation ($\epsilon_{base}$) | $2.5 \times 10^{-4}$ | Layer-adaptive + element-wise override |
| Global Learning Rate ($\eta$) | $1 \times 10^{-5}$ | Tensor Adam provides per-coordinate adaptation |
| Adam $\beta_1$ / $\beta_2$ | 0.9 / 0.999 | |
| Adam $\delta$ | $10^{-8}$ | |
| Noise Calibration $\alpha_n$ | 0.1 | EMA decay for noise tracking |
| Calibration Interval ($K_{calib}$) | 1000 steps | Activation basis refresh |
| Calibration Blend ($\alpha_{proj}$) | 0.8 | Activation projection weight |
| Exploration Candidates ($N$) | 4 | RLOO requires $N \geq 3$ for meaningful baselines |
| Exploration Temperature | 0.7 | |
| Reward Threshold ($R_{min}$) | 0.1 | |
| Contrastive Gap ($\delta$) | 0.1 | |
| KL Penalty ($\beta$) | 0.1 (0.0 for short runs) | |
| NLL Spike Threshold | $5\times$ EMA | |
| Rank Bounds ($r_{min}$ / $r_{max}$) | 4 / 64 | |

The system exposes 2 primary tuning knobs: $\eta$ and $\epsilon_{base}$. All other adaptation is automatic. For the full per-item classification and impact assessment, see `DS_MeZO_Parameter_Audit.md`.

## 13. Per-Step Compute Cost

| Pass | Operation |
| :--- | :--- |
| Explore | `generate(n=N, T=0.7)` $\to$ reward score, RLOO advantages |
| KL Ref | `generate(max_tokens=1, prompt_logprobs=0)` — cache reference logprobs (2 sequences) |
| Tick/Tock-A/Tock-B | `generate(max_tokens=1, prompt_logprobs=0)` per adapter × per layer per phase |

**Effective cost:** 1 generation + 1 reference prefill + 6 scoring prefills per layer per phase (2 adapters × 2 sequences, called twice: once per adapter). Each prefill is 25-100$\times$ faster than generation. Noise calibration and activation projection add negligible CPU compute.

**Adapter sync overhead:** ~60-180ms per step (with tmpfs-backed staging directory). Dominated by DoRA-to-LoRA fusion SVD (~10ms/layer CPU) and safetensors serialization (~20ms). Using tmpfs (`mount -t tmpfs tmpfs /tmp/ds_mezo`) eliminates disk I/O entirely.

**Calibration cost:** One HuggingFace forward pass every $K_{calib}$ steps (~30s for 70B at FP16). Amortized over 1000 steps: <0.03s per step.

**Scheduling variant:** Alternating Tock-A/Tock-B across steps drops per-layer scoring cost by one-third.

## 14. Risk & Failure Mode Analysis

| Failure Mode | Trigger | Mitigation |
| :--- | :--- | :--- |
| **FP16 Silent Gradient Death** | Static $\epsilon$ on large PiSSA values | Element-wise $\epsilon$ with FP16 ULP floor (Section 6.2) |
| **Deep-Layer Logit Divergence** | Uniform $\epsilon$ across all layers | Layer-adaptive $\epsilon_l = \epsilon_{base}/\sqrt{l+1}$ (Section 6.2) |
| **Trajectory Divergence** | Perturbed autoregressive generation | Trajectory locking (Section 5.1) |
| **Mode Collapse** | Long runs with pure NLL, no policy anchor | Per-token KL shaping (Section 5.3) |
| **Bilinear Cross-Term Variance** | Simultaneous perturbation of $A$ and $B$ | Tock-A / Tock-B split (Section 4.2) |
| **FP16 Update Truncation** | Small $\Delta\theta$ added to large $\theta$ | FP32 master accumulation (Section 7.2) |
| **NF4 Noise Amplification** | NF4 residual error + FP16 perturbation at 70B scale | Noise calibration $\gamma$ + FP32 masters + LOREN floor |
| **Static Rank Starvation** | Uniform rank across heterogeneous layers | Dynamic rank allocation via gradient magnitude |
| **Suboptimal Perturbation Direction** | Static PiSSA subspace missing data-dependent structure | Activation-guided projection (Section 6.1) |
| **DoRA Fusion Approximation Error** | Truncated SVD re-decomposition loses information beyond rank $r$ | Rank $r$ captures >99% of the delta energy since updates are inherently low-rank |

**Operational Risks:**

| Risk | Severity | Mitigation |
| :--- | :--- | :--- |
| Adapter sync latency (serialization + reload) | Medium | Use tmpfs for staging directory. Latency: ~60-180ms/step vs ~6-12ms with hypothetical in-memory API. Acceptable for ZO training where prefill dominates. |
| vLLM does not support DoRA natively | Medium | Controller-side DoRA-to-LoRA fusion. No loss of optimization semantics; slight approximation from rank-$r$ SVD re-decomposition of the delta. |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING` required | Low | Set at deployment. Standard requirement for any dynamic LoRA workflow. |
| Convergence theory covers BCD structure but not full composed system | Medium | Three-phase BCD validated by Park et al. Composition is the primary empirical risk. |
| Hard tasks reject most steps | Medium | Lower $R_{min}$/$\delta$, increase $N$, curriculum learning. |
| Calibration requires HuggingFace model load | Low | One-time ~60s load, reused across calibration passes. |

## 15. Constraints & Portability

1. Requires vLLM with `--enable-lora`, dynamic LoRA loading (`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`), and `prompt_logprobs` support in `SamplingParams`.
2. DoRA decomposition is controller-side; vLLM receives standard LoRA adapters. No custom vLLM fork needed.
3. Model must have linear layers compatible with LoRA. Specialized layers (1D convolutions, novel SSMs, complex MoE routing) need custom kernels.
4. Tuned for 70B+ models. For <1B, PiSSA rank reduction may bottleneck capacity.
5. Any `str -> float` reward function works: code correctness, proof validity, tool success, LLM-as-Judge.

---

## 16. References

| Citation | Paper | Role |
| :--- | :--- | :--- |
| arXiv:2402.09353 | DoRA: Weight-Decomposed Low-Rank Adaptation | Magnitude/direction decomposition. Tick/Tock decoupling. Controller-side optimization. |
| arXiv:2404.02948 | PiSSA: Principal Singular Values and Singular Vectors Adaptation | SVD adapter init. QPiSSA quantization. |
| arXiv:2506.05454 | Zhang et al. — ZO Finds Flat Minima | Core theory. Implicit regularization. PL convergence. |
| arXiv:2501.19099 | Park et al. — Elucidating Subspace Perturbation (MeZO-BCD) | Block coordinate descent convergence. Subspace alignment theory. Cyclic random permutation. |
| arXiv:2601.01452 | Yao et al. — BSZO: Bayesian Subspace ZO | Adaptive noise calibration via prediction residuals. Reduced-precision robustness. |
| arXiv:2601.17261 | Park et al. — AGZO: Activation-Guided ZO | Subspace quality bound (Theorem 5.6). Activation-guided perturbation direction. |
| — | RLOO / ReMax (2025) | Unbiased minimum-variance multi-sample baseline. |
| — | LOREN (2025) | Layer-adaptive perturbation scaling for deep Transformers. |
| — | S-LoRA / vLLM | Multi-adapter serving. PagedAttention. Dynamic LoRA loading with `load_inplace`. |
