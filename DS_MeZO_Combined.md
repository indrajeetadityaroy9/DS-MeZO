# DS-MeZO: Zeroth-Order Optimization for LLM Fine-Tuning

**Single NVIDIA H100 80GB HBM3 (SM 9.0 / Hopper) | CUDA 12.8 + PyTorch 2.7 + Triton 3.3 | vLLM (S-LoRA Backend)**

---

## 1. Core Idea

DS-MeZO fine-tunes LLMs without backpropagation using zeroth-order (ZO) gradient estimation, optimizing non-differentiable objectives (code compilation, proof verification, tool use) at near-inference memory cost.

**Pipeline:** PiSSA subspace initialization → per-step activation subspace tracking (power iteration) → AGZO subspace perturbation + momentum-aligned sensitivity masking → FP32 SPSA with adaptive $\epsilon$ → RLOO contrastive selection → gradient-regularized contrastive loss with KL constraint → ZO-Muon spectral update (FP32) → cosine LR + entropy-guided temperature annealing.

The Controller is implemented entirely in OpenAI Triton. vLLM handles all forward passes via standard LoRA adapters. PiSSA adapters are natively LoRA-compatible — no fusion step required.

---

## 2. ZO Gradient Estimation

For loss $f(\theta)$ and perturbation direction $z$, the symmetric two-point SPSA estimator:

$$ \hat{g} = \frac{f(\theta + \epsilon z) - f(\theta - \epsilon z)}{2\epsilon} \cdot z $$

This estimates the gradient of the Gaussian-smoothed surrogate:

$$ f_\epsilon(\theta) = f(\theta) + \frac{\epsilon^2}{2} \operatorname{Tr}(\nabla^2 f(\theta)) + \mathcal{O}(\epsilon^4) $$

The $\operatorname{Tr}(\nabla^2 f)$ penalty drives parameters toward flat minima (implicit SAM, Zhang et al. 2506.05454).

**All-at-once perturbation:** All layers' $A$ and $B$ are perturbed simultaneously. The scalar finite difference $(L^+ - L^-)$ is shared across all layers; per-layer gradients are distinguished by their independent perturbation directions $z_l$. Bilinear cross-terms between $A$ and $B$ are $\mathcal{O}(\epsilon^2)$, same order as the SPSA smoothing bias.

### 2.1 Numerical Stability

The SPSA ratio $(L^+ - L^-) / 2\epsilon$ amplifies noise by $1/(2\epsilon)$. At $r=16$, $\epsilon = 2.5 \times 10^{-4}$, yielding $\sim 2000\times$ noise amplification. Three mitigations (informed by HZO, arXiv:2602.10607):

1. **FP32 loss computation.** All logprob accumulation, NLL reduction, and finite-difference arithmetic is performed in FP32, even though forward passes use BF16. This prevents catastrophic cancellation in $(L^+ - L^-)$ when the loss difference is small. BF16's 8-bit exponent (vs FP16's 5-bit) already reduces overflow risk in intermediate logprobs, and FP32 accumulation eliminates it entirely.

2. **Adaptive $\epsilon$.** Scale $\epsilon$ proportionally to the loss EMA to maintain a stable signal-to-noise ratio:
$$ \epsilon_t = \epsilon_0 \cdot \max\left(\frac{\bar{L}_t}{\bar{L}_0}, \, \epsilon_{floor}\right), \quad \epsilon_0 = \frac{10^{-3}}{\sqrt{r}}, \quad \epsilon_{floor} = 0.1 $$
When losses shrink during training, $\epsilon$ shrinks proportionally, keeping $|L^+ - L^-|$ well above FP32 noise floor. Clamped at $\epsilon_{floor} \cdot \epsilon_0$ to prevent vanishing perturbations.

3. **Directional derivative clipping** (inspired by QZO, arXiv:2505.13430). Clip the raw finite-difference ratio before multiplying by $z$:
$$ \hat{d} = \operatorname{clip}\left(\frac{L^+ - L^-}{2\epsilon}, \, -c, \, c\right), \quad c = 3 \cdot \operatorname{EMA}(|\hat{d}|) $$
This prevents single outlier loss evaluations from producing catastrophic gradient spikes.

---

## 3. Subspace Design: PiSSA + AGZO + Sparse Masking

### 3.1 PiSSA Initialization

Decompose $W_0$ via fast SVD (Halko, `niter=2`):
- $A = U[:, :r] \cdot \sqrt{S[:r]}$, $B = \sqrt{S[:r]} \cdot V^T[:r, :]$
- $W^{res} = W_0 - AB$, quantized to NF4

PiSSA adapters are natively LoRA-compatible: $W' = W^{res} + AB$.

Effective trainable dimensionality: $d_{eff} = (d_{out} \times r) + (r \times d_{in})$ per layer.

### 3.2 AGZO Activation-Guided Subspace Perturbation

The gradient of a linear layer $Y = XW^T$ satisfies $\nabla_W L = (\nabla_Y L)^T \cdot X$. The gradient's row space is confined to $\operatorname{span}(X)$ — the activation subspace (AGZO, arXiv:2601.17261).

**Perturbation for $B$ (r × d_in):** Generate perturbation directly in the activation subspace:
$$ z_B = Z_{coeff} \cdot V_l^T, \quad Z_{coeff} \sim \mathcal{N}(0, I_{r \times r_{calib}}) $$

where $V_l \in \mathbb{R}^{d_{in} \times r_{calib}}$ is the per-layer activation basis. Effective dimensions: $r \times r_{calib} = 128$ (at $r=16$, $r_{calib}=8$).

**Perturbation for $A$ (d_out × r):** The gradient's row space (in the $r$ dimension) lies in $\operatorname{span}(B \cdot V_l)$. Project:
$$ Q = \operatorname{orth}(B \cdot V_l), \quad z_A = Z_{coeff}' \cdot Q^T \odot M_l $$

where $M_l$ is the sparse mask (§3.3) and $Z_{coeff}' \sim \mathcal{N}(0, I_{d_{out} \times \min(r, r_{calib})})$. Effective dimensions after masking: $\sim 0.2 \times d_{out} \times r_{calib}$.

### 3.3 Momentum-Aligned Sensitivity Masking

**Problem with magnitude-based masking:** The original Sparse MeZO (NUS-HPC-AI-Lab) masks large-magnitude parameters to reduce ZO noise. However, PiSSA initializes $A = U \cdot \sqrt{S}$ — large-magnitude entries correspond to top singular values, which are the most important components for the pretrained model. Magnitude-based masking structurally prevents updating the most informative parameters, contradicting PiSSA's design intent.

**Solution:** Replace magnitude-based masking with momentum-gradient alignment masking (Magma, arXiv:2602.15322). Instead of selecting parameters by weight magnitude, select parameters where the current ZO gradient direction is consistent with accumulated momentum — indicating stable, reliable signal rather than noise:

$$ M_l = \mathbf{1}\left[\operatorname{sign}(v_l) = \operatorname{sign}(\hat{g}_l)\right] $$

where $v_l$ is the momentum buffer and $\hat{g}_l$ is the current ZO gradient estimate. Parameters where gradient and momentum agree are receiving consistent signal across steps; parameters where they disagree are dominated by noise.

**Warmup:** During the first $W = 10$ steps (before momentum buffers are populated), use the full unmasked perturbation. After warmup, alignment masking activates automatically.

**Properties:**
- Curvature-aware: implicitly selects parameters in low-curvature directions where ZO estimates are reliable (Magma proves this induces geometric regularization proportional to Hessian eigenvalues).
- SVD-compatible: does not penalize large-magnitude PiSSA components — they receive updates whenever the gradient signal is consistent.
- Dynamic: mask adapts automatically as training progresses, unlike fixed magnitude thresholds.
- No memory overhead: reuses existing momentum buffers; mask is computed on-the-fly.

Applied to $A$ only. $B$'s perturbation is already low-dimensional via AGZO (128 effective dims), so masking is unnecessary.

### 3.4 Effective Dimensionality Summary

| Component | Raw Dims | With AGZO + Masking | Reduction |
|:----------|:---------|:--------------------|:----------|
| $B$ (per layer) | $r \times d_{in} = 131K$ | $r \times r_{calib} = 128$ | 1024× |
| $A$ (per layer) | $d_{out} \times r = 131K$ | $\sim 0.5 \times d_{out} \times r_{calib} \approx 33K$ | 4× |
| **Total (80 layers)** | **20.97M** | **~2.65M** | **8×** |

Note: Momentum-aligned masking typically selects ~50% of parameters (those with gradient-momentum agreement), versus 20% for the original magnitude masking. The reduction is less aggressive but targets signal-bearing parameters instead of systematically excluding the most important ones.

### 3.5 Activation Subspace Tracking

**Problem with fixed-interval calibration:** The original design ran calibration every 100 steps using a fixed sample set. This creates a chicken-and-egg problem: (1) early training changes weights rapidly, so 100 steps of stale bases means perturbations in increasingly misaligned subspaces; (2) the fixed calibration set may not represent the on-policy distribution, which shifts as the model improves. AGZO (arXiv:2601.17261) solves this by recomputing bases every step.

**Per-step power iteration** (following AGZO's actual method): Extract activation bases from the current step's exploration batch, not a fixed calibration set. Use power iteration ($K=3$ steps) instead of SVD — adds only a few matrix multiplications per layer:

$$ V_l^{(k+1)} = \operatorname{normalize}\left(H_l^T (H_l \cdot V_l^{(k)})\right), \quad k = 1, \ldots, K $$

where $H_l \in \mathbb{R}^{B \times d_{in}}$ is the current batch's input activations at layer $l$, and $V_l$ is warm-started from the previous step's basis.

**Warm-starting** ensures continuity: the basis evolves smoothly across steps rather than jumping between independently computed SVDs. Cost: $K=3$ matrix multiplications of size $(B \times d_{in}) \times (d_{in} \times r_{calib})$ per layer — negligible compared to a forward pass.

**Adaptive full recalibration** (inspired by LOTUS, arXiv:2602.01233): Monitor subspace drift via the path-efficiency ratio. When the cosine similarity between successive $V_l$ drops below a threshold $\tau_{drift} = 0.95$, trigger a full SVD recalibration using a larger sample:

$$ \operatorname{drift}_l = 1 - \frac{|\operatorname{tr}(V_l^{(t)T} V_l^{(t-1)})|}{r_{calib}} $$

Full recalibration fires only when the power-iteration-tracked basis diverges significantly, typically during learning rate warmup or after distribution shifts. In practice, this triggers 3-5 times per training run rather than at fixed 100-step intervals.

---

## 4. Trajectory Locking + RLOO

### 4.1 Trajectory Locking
1. Generate $N$ candidates under unperturbed weights $\theta_0$.
2. Select winner/loser via RLOO advantages.
3. Score **fixed sequences** (prefill-only) under $\theta^+$ and $\theta^-$.

### 4.2 RLOO Advantages
$$ A_i = R_i - \frac{1}{N-1} \sum_{j \neq i} R_j $$

Unbiased, minimum-variance, self-centering ($\sum_i A_i = 0$), zero lag, no tunable parameters. RLOO advantages are self-regulating: when all candidates are similar quality, advantages are near-zero, producing near-zero gradients.

### 4.3 Gradient-Regularized Contrastive Loss with KL Constraint

**Problem with KL-shaped loss:** The original $\beta \cdot \text{KL}$ penalty uses fixed reference logprobs $\pi_{ref}$. Since $\pi_{ref}$ is constant w.r.t. $\theta$, the KL term cancels exactly in the SPSA finite difference $(L^+ - L^-)$, providing **zero gradient signal**. Mode collapse prevention relied entirely on RLOO's self-centering — an insufficient safeguard (arXiv:2510.20817 proves standard KL-regularized RL inherently mode-collapses at common hyperparameter settings).

**Solution — two complementary mechanisms:**

**(a) Gradient regularization** (arXiv:2602.18037): Replace the KL penalty with a reward-gradient norm penalty that is explicitly designed for finite-difference estimation and provides actual gradient signal through ZO:

$$ \mathcal{L}(\theta) = \sum_{i \in \{w, l\}} A_i \cdot \text{NLL}_i(\theta) + \lambda_{GR} \cdot \|\hat{\nabla}_\theta R\|^2 $$

The gradient regularization term $\|\hat{\nabla}_\theta R\|^2$ penalizes parameter regions where the reward changes rapidly, biasing toward flat reward regions where the score function is more reliable. Crucially, this term **does not cancel** in finite differences because it depends on $\theta$ through both perturbation directions:

$$ \|\hat{\nabla}_\theta R\|^2 \approx \left(\frac{R(\theta^+) - R(\theta^-)}{2\epsilon}\right)^2 $$

This is computed from the same reward evaluations already available from the exploration phase — no additional forward passes. $\lambda_{GR} = 0.01$.

**(b) KL divergence constraint** (inspired by DPPO, arXiv:2602.04879): Instead of adding KL as a loss penalty (where it cancels), use it as a **hard constraint** on perturbation acceptance. After computing the SPSA gradient, reject the update if the resulting parameter change would move the policy too far from the reference:

$$ \text{KL}_{approx} = \frac{1}{T} \sum_t \left(\log \pi_{\theta_{new}}(y_t) - \log \pi_{\theta_{old}}(y_t)\right)^2 $$

If $\text{KL}_{approx} > \delta_{KL}$, scale down the update: $\eta_t \leftarrow \eta_t \cdot \delta_{KL} / \text{KL}_{approx}$. The constraint $\delta_{KL} = 0.01$ is evaluated using the scoring logprobs already computed — no extra forward passes. This sidesteps finite differences entirely since it operates as a post-hoc filter on the update magnitude.

### 4.4 Safety
**Spike detection:** Skip if NLL $> 5\times$ EMA. Update EMA only after healthy steps.
**Directional derivative clipping:** Clip $(L^+ - L^-) / 2\epsilon$ at $3\times$ its running EMA (§2.1).

---

## 5. Optimizer: ZO-Muon + Cosine Schedule

### 5.1 Motivation: Beyond Weak Adaptivity

MeZO-A³dam (OpenReview:OBIuFjZzmp) shows that full Adam preconditioning hurts ZO optimization — the "weak adaptivity hypothesis." The original DS-MeZO used SGD+momentum (zero adaptivity), but this creates a new problem: when perturbations are drawn from a fixed AGZO subspace (refreshed infrequently), successive gradient estimates are correlated. Momentum applied to correlated estimates amplifies shared bias rather than averaging out noise.

**ZO-Muon** (arXiv:2602.17155) resolves this by applying **spectral orthogonalization** to ZO gradient estimates. Instead of entry-wise momentum accumulation, ZO-Muon extracts informative descent directions via Newton-Schulz orthogonalization within the AGZO subspace. This is neither zero adaptivity (SGD) nor full per-coordinate adaptivity (Adam) — it operates at the **spectral level**, denoising gradient estimates by their singular structure.

ZO-Muon achieves the same performance as MeZO with only 24.7% of the queries (arXiv:2602.17155), directly addressing the correlated-subspace bias concern.

### 5.2 Update Rule

For each matrix parameter $\Theta$ (A or B per layer):

1. **Accumulate momentum** on the ZO gradient estimate:
$$ G_t = \mu \cdot G_{t-1} + (1 - \mu) \cdot \hat{g}_t $$

2. **Newton-Schulz orthogonalization** (5 iterations, no SVD needed):
$$ X_0 = G_t / \|G_t\|_F $$
$$ X_{k+1} = \frac{1}{2} X_k (3I - X_k^T X_k), \quad k = 0, \ldots, 4 $$

3. **Spectral update:**
$$ \Theta_{t+1} = \Theta_t - \eta_t \cdot X_5 $$

The orthogonalization extracts the dominant spectral directions from the noisy gradient, suppressing noise components that would be amplified by naive momentum. For the small matrices in LoRA ($d_{out} \times r$ and $r \times d_{in}$), Newton-Schulz adds negligible compute.

**FP32 master weights and momentum buffers** on GPU prevent accumulation truncation. Forward passes use BF16 (§6.2.1) for Hopper-native throughput with superior dynamic range.

### 5.3 Cosine Learning Rate Schedule

$$ \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T} \pi\right)\right) $$

with $\eta_{min} = \eta_{max} / 100$. No state, no tuning beyond $\eta_{max}$.

---

## 6. System Architecture

```
+----------------------------------------------------------+
|                    DS-MeZO CONTROLLER                    |
|                (Python / OpenAI Triton)                   |
|                                                          |
|  +------------------+  +------------------+  +---------+ |
|  |  FP32 Master     |  |  ZO-Muon         |  | Activ.  | |
|  |  Weights         |  |  (G per layer    |  | Subsp.  | |
|  |  (A, B)          |  |   + Newton-      |  | Tracker | |
|  |  PiSSA           |  |   Schulz orth.)  |  | (V_l)   | |
|  +--------+---------+  +--------+---------+  +---------+ |
|           +----------------+----------------+            |
|             +--------------v--------------+              |
|             |  Direct LoRA Serialization  |              |
|             |  (PEFT-compatible, no SVD)  |              |
|             +--------------+--------------+              |
+----------------------------+-----------------------------+
                             |  Serialize + LoRARequest
                             |  (load_inplace=True)
                             v
+----------------------------------------------------------+
|                    vLLM ENGINE                           |
|              (GPU / Standard LoRA Mode)                   |
|  +--------------+  +-----------+  +------------------+  |
|  |  Base Model   |  |  S-LoRA   |  |  PagedAttention  |  |
|  |  W_res (NF4)  |  |  Paging   |  |  + KV Cache      |  |
|  +--------------+  +-----------+  +------------------+  |
+----------------------------------------------------------+
```

PiSSA adapters are standard LoRA format. Serialization writes $A$ and $B$ matrices directly to PEFT-compatible safetensors. No SVD in the scoring loop.

**Adapter reload:** vLLM's `LoRARequest` with `load_inplace=True` forces re-reading adapter weights from disk on each `generate()` call, enabling hot-swap without server-side API.

**Scoring** uses `generate()` with `SamplingParams(max_tokens=1, prompt_logprobs=1)` — prefill-only logprob extraction on fixed sequences. Note: `prompt_logprobs` must be ≥1; vLLM treats `0` as falsy and returns `None`.

### 6.1 Fused Triton Kernels

| Kernel | Fused Operations |
|:-------|:-----------------|
| `subspace_perturb` | Power iteration basis update → AGZO projection → momentum-aligned masking → $\theta \pm \epsilon z$ |
| `zo_muon_update` | Gradient → momentum accumulation → Newton-Schulz orthogonalization (5 iter) → param update |
| `spsa_gradient` | FP32 $\hat{g} = \operatorname{clip}(\Delta\mathcal{L} / 2\epsilon_t) \cdot z$ with adaptive $\epsilon$ |
| `score_reduce` | Per-token logprob → NLL + gradient regularization → advantage-weighted loss (FP32) |
| `health_monitor` | EMA update → spike detection → KL constraint check → $\epsilon$ adaptation |

### 6.2 H100 Hardware Optimization

The following optimizations exploit specific features of the local hardware: NVIDIA H100 80GB HBM3 (SM 9.0), Intel Xeon Platinum 8480+ (13 cores / 26 threads), 222 GB DDR5, CUDA 12.8, PCIe Gen5 x16.

#### 6.2.1 BF16 Forward Passes (not FP16)

The H100 achieves identical tensor core throughput for BF16 and FP16 (~1979 TFLOPS), but BF16 provides 8 bits of exponent (vs 5 for FP16), preventing overflow/underflow in logprob accumulation across long sequences. All forward passes, adapter weights, and KV cache use **BF16** rather than FP16. This directly improves the SPSA numerical stability (§2.1) at zero throughput cost on Hopper.

```python
# Adapter serialization uses BF16 instead of FP16
tensors[f"{prefix}.lora_A.weight"] = A_l.bfloat16()  # was .half()
tensors[f"{prefix}.lora_B.weight"] = B_l.bfloat16()
```

vLLM launch: `--dtype bfloat16` (default on H100).

#### 6.2.2 Adapter Staging on tmpfs (/dev/shm)

The system has **111 GB tmpfs** at `/dev/shm` — orders of magnitude faster than the NVMe-backed `/tmp`. Adapter serialization (2 writes per step, ~32 MB each at rank 16) on tmpfs eliminates all disk I/O from the training loop:

```bash
export DS_MEZO_ADAPTER_DIR=/dev/shm/ds_mezo
```

Measured: `/dev/shm` write latency is ~0.1ms vs ~5ms for NVMe. At 2 writes/step over 1000 steps, this saves ~10 seconds and eliminates I/O variance. vLLM's `load_inplace=True` reads these files back via mmap, which on tmpfs avoids page cache thrashing.

#### 6.2.3 GPU Clock Locking

Application clocks are **not locked by default**, causing the GPU to frequency-scale between 345 MHz (idle) and 1980 MHz (boost). This introduces variance in loss computation between $\theta^+$ and $\theta^-$ scoring passes — if the first pass triggers boost and the second runs at a lower clock, the SPSA difference includes timing-dependent numerical noise.

```bash
# Lock clocks to max for consistent compute during training
sudo nvidia-smi -lgc 1980,1980    # Lock graphics clock
sudo nvidia-smi -lmc 2619         # Lock memory clock
# Restore after training:
# sudo nvidia-smi -rgc && sudo nvidia-smi -rmc
```

#### 6.2.4 CUDA Graphs for Prefill Scoring

The scoring phase runs 4 identical-structure prefill passes (same sequence lengths, same adapter shape, different weights). vLLM supports CUDA graph capture for prefill, which eliminates kernel launch overhead (~10μs per kernel × hundreds of kernels per prefill = ~2ms per pass):

```python
# vLLM launch with CUDA graphs enabled (default on H100)
llm = LLM(
    model=model_path,
    gpu_memory_utilization=0.92,
    enable_lora=True,
    max_lora_rank=64,
    dtype="bfloat16",
    enforce_eager=False,          # Enable CUDA graphs
    max_num_seqs=8,               # Limit batch for graph capture
)
```

H100's large L2 cache (50 MB) and HBM3 bandwidth (3.35 TB/s) make prefill heavily compute-bound rather than memory-bound, so CUDA graphs primarily help by reducing launch overhead rather than improving occupancy.

#### 6.2.5 Pinned Memory for Adapter Transfer

With 222 GB system RAM, pin adapter staging buffers in CPU memory for faster PCIe Gen5 x16 DMA transfers:

```python
# Pre-allocate pinned staging buffers (one-time, ~64 MB total)
self._pinned_pos = torch.empty(total_adapter_bytes, dtype=torch.uint8,
                                pin_memory=True)
self._pinned_neg = torch.empty(total_adapter_bytes, dtype=torch.uint8,
                                pin_memory=True)
```

PCIe Gen5 x16 theoretical bandwidth: 63 GB/s. A 32 MB adapter transfers in ~0.5ms with pinned memory vs ~2ms without.

#### 6.2.6 CPU Thread Allocation

26 hardware threads available. Allocate explicitly to avoid contention:

| Role | Threads | Notes |
|:-----|:--------|:------|
| vLLM engine (torch) | 16 | `OMP_NUM_THREADS=16` |
| Controller (scoring, RLOO, serialization) | 4 | `DS_MEZO_WORKERS=4` |
| Data loading / tokenization | 4 | `num_workers=4` in dataloader |
| OS / system | 2 | Reserved |

```bash
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=true
```

#### 6.2.7 torch.compile for Controller Kernels

PyTorch 2.7 with `torch.compile` can fuse the controller's non-Triton operations (Newton-Schulz iterations, power iteration, masking) into optimized CUDA kernels via Inductor:

```python
@torch.compile(mode="reduce-overhead", fullgraph=True)
def newton_schulz_orthogonalize(G, num_iters=5):
    norm = torch.norm(G, 'fro')
    if norm < 1e-8:
        return G
    X = G / norm
    I = torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    for _ in range(num_iters):
        X = 0.5 * X @ (3 * I - X.T @ X)
    return X
```

`mode="reduce-overhead"` uses CUDA graphs internally, eliminating Python overhead for repeated calls with the same tensor shapes (which is the case for all per-layer operations).

#### 6.2.8 Memory Utilization Target

With BF16 and /dev/shm staging, increase vLLM's memory utilization from 0.90 to **0.92**, reclaiming ~1.6 GB for larger KV cache (longer sequences or more concurrent candidates):

```bash
--gpu-memory-utilization 0.92
```

---

## 7. Execution

### 7.1 Per-Step Loop

```
1. EXPLORE: Generate N candidates at temperature T_t (annealed) → reward → RLOO advantages
            → select winner/loser. Compute gradient regularization term from reward differences.
            Extract per-layer activations for basis tracking (power iteration, K=3).

2. PERTURB: For each layer, update activation basis V_l via warm-started power iteration.
            Generate subspace perturbation (AGZO for B, momentum-aligned masking for A).
            Construct θ+ and θ- adapters (all layers, adaptive ε_t).

3. SCORE:   Serialize θ+ and θ- adapters to disk.
            Score fixed sequences under θ+ and θ- (4 prefills total, FP32 logprob accumulation).

4. UPDATE:  Health check — skip if NLL spike or directional derivative exceeds clip threshold.
            Per-layer SPSA gradient: ĝ = clip((L+ - L-) / (2ε_t)) · z
            KL constraint check — scale down η if update exceeds δ_KL.
            ZO-Muon spectral update (FP32 masters): momentum → Newton-Schulz → param update.

5. SCHEDULE: Cosine LR update. Adaptive ε update. Temperature anneal.
             Check subspace drift — trigger full SVD recalibration if drift > τ.
```

### 7.2 Per-Step Compute Cost

| Operation | Count |
|:----------|:------|
| Generation | 1 (N candidates) |
| Reference prefill | 1 (2 sequences) |
| Scoring prefills | 4 (2 sequences × 2 perturbation directions) |
| Adapter serializations | 2 (θ+ and θ-) |

**Total prefills per step: 7.** Step time ≈ generation time.

Adapter serialization: 2 writes of PEFT safetensors (~5ms each with tmpfs).

### 7.3 Entropy-Guided Temperature Annealing

**Problem with fixed temperature:** A fixed exploration temperature (0.7) is suboptimal across training. Early training needs diversity to discover reward signal; late training needs focused candidates to refine (arXiv:2505.18573, arXiv:2510.08141).

**Cosine temperature schedule** with entropy floor (inspired by arXiv:2505.18573):

$$ T_t = T_{min} + \frac{1}{2}(T_{max} - T_{min})\left(1 + \cos\left(\frac{t}{T_{total}} \pi\right)\right) $$

with $T_{max} = 1.0$, $T_{min} = 0.3$. This starts with high diversity and smoothly transitions to exploitation.

**Entropy monitoring:** Track the entropy of the sampling distribution at each step. If entropy drops below $H_{floor} = 0.5 \cdot H_0$ (where $H_0$ is the initial entropy), temporarily boost temperature by $1.5\times$ to prevent premature convergence. This implements the core insight of AEPO (arXiv:2510.08141): entropy collapse is the proximate cause of exploration failure in LLM RL.

---

## 8. Implementation

```python
import torch
import math
import os
import json
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from safetensors.torch import save_file


ADAPTER_STAGING_DIR = os.environ.get("DS_MEZO_ADAPTER_DIR", "/dev/shm/ds_mezo")


def save_peft_adapter(A_list, B_list, adapter_dir, layer_configs):
    os.makedirs(adapter_dir, exist_ok=True)
    tensors = {}
    for layer_idx, (A_l, B_l) in enumerate(zip(A_list, B_list)):
        prefix = layer_configs[layer_idx]['peft_prefix']
        tensors[f"{prefix}.lora_A.weight"] = A_l.bfloat16()
        tensors[f"{prefix}.lora_B.weight"] = B_l.bfloat16()
    save_file(tensors, os.path.join(adapter_dir, "adapter_model.safetensors"))
    config = {
        "peft_type": "LORA",
        "r": A_list[0].shape[1],
        "lora_alpha": A_list[0].shape[1],
        "target_modules": [cfg['target_module'] for cfg in layer_configs],
    }
    with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f)


def extract_prompt_logprobs(output, prompt_token_ids):
    logprobs = []
    if output.prompt_logprobs is None:
        return logprobs
    for i, token_lp in enumerate(output.prompt_logprobs[1:], 1):
        if token_lp is not None:
            tok_id = prompt_token_ids[i]
            if tok_id in token_lp:
                logprobs.append(token_lp[tok_id].logprob)
    return logprobs


@torch.compile(mode="reduce-overhead", fullgraph=True)
def newton_schulz_orthogonalize(G, num_iters=5):
    """Newton-Schulz orthogonalization for ZO-Muon (arXiv:2602.17155).
    Extracts spectral structure from noisy ZO gradients without SVD.
    torch.compile fuses iterations into a single CUDA graph on H100."""
    norm = torch.norm(G, 'fro')
    if norm < 1e-8:
        return G
    X = G / norm
    I = torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    for _ in range(num_iters):
        X = 0.5 * X @ (3 * I - X.T @ X)
    return X


class DSMeZO_Controller:
    def __init__(self, vllm_engine, model_config, score_fn,
                 total_steps, calibration_data):
        self.engine = vllm_engine
        self.score_fn = score_fn
        self.step_count = 0
        self.total_steps = total_steps
        self.num_layers = model_config.num_target_layers

        # PiSSA initialization — FP32 master weights
        self.layers = []
        for layer_idx in range(self.num_layers):
            A, B, W_res = initialize_pissa(model_config, layer_idx)
            self.layers.append({
                'A': A.float(), 'B': B.float(),
                'W_res': W_res,
                'layer_idx': layer_idx,
                'rank': model_config.rank,
                'peft_prefix': model_config.peft_prefix(layer_idx),
                'target_module': model_config.target_module(layer_idx),
            })

        # Adapter staging
        self.adapter_dir_pos = os.path.join(ADAPTER_STAGING_DIR, "adapter_pos")
        self.adapter_dir_neg = os.path.join(ADAPTER_STAGING_DIR, "adapter_neg")
        self.lora_pos = LoRARequest(
            "adapter_pos", 1, self.adapter_dir_pos, load_inplace=True
        )
        self.lora_neg = LoRARequest(
            "adapter_neg", 2, self.adapter_dir_neg, load_inplace=True
        )
        self._sync_adapters()

        self.score_params = SamplingParams(
            max_tokens=1, prompt_logprobs=1, temperature=0.0
        )

        # Exploration — entropy-guided temperature annealing (arXiv:2505.18573)
        self.num_candidates = 4
        self.temp_max = 1.0
        self.temp_min = 0.3
        self.explore_temperature = self.temp_max
        self.entropy_floor_ratio = 0.5
        self.initial_entropy = None

        # Gradient regularization (arXiv:2602.18037)
        self.lambda_gr = 0.01
        # KL constraint (inspired by DPPO, arXiv:2602.04879)
        self.delta_kl = 0.01

        # Perturbation — adaptive epsilon (arXiv:2602.10607)
        self.eps_base = 1e-3 / math.sqrt(model_config.rank)
        self.eps = self.eps_base
        self.eps_floor = 0.1
        self.initial_loss_ema = None

        # Optimizer: ZO-Muon (arXiv:2602.17155)
        self.eta_max = 1e-4
        self.eta_min = self.eta_max / 100
        self.eta = self.eta_max
        self.momentum = 0.9
        self.momentum_buffers = {}

        # Masking warmup — use full perturbation for first W steps
        self.mask_warmup_steps = 10

        # Activation subspace tracking — per-step power iteration (AGZO)
        self.r_calib = 8
        self.power_iter_steps = 3
        self.drift_threshold = 0.95
        self._calib_data = calibration_data
        self.activation_bases = {}
        self._calibrate_activation_bases_full()

        # Directional derivative clipping (arXiv:2505.13430)
        self.dd_ema = None
        self.dd_clip_multiplier = 3.0

        # Health monitoring
        self.loss_ema = None
        self.loss_ema_momentum = 0.95
        self.spike_threshold = 5.0

    # -- Activation Subspace Tracking ------------------------------------------

    def _calibrate_activation_bases_full(self):
        """Full SVD calibration — used at init and on drift detection."""
        activations = extract_layer_activations(
            self.engine, self._calib_data, self.num_layers
        )
        for layer_idx in range(self.num_layers):
            H = activations[layer_idx]  # batch*seq_len × d_in
            d_in = H.shape[1]
            r_calib = min(self.r_calib, d_in)
            _, _, Vt = torch.svd_lowrank(H, q=r_calib, niter=2)
            self.activation_bases[layer_idx] = Vt.float()  # d_in × r_calib

    def _update_activation_bases_power_iter(self, activations):
        """Per-step warm-started power iteration (AGZO, arXiv:2601.17261).
        Updates bases from current batch activations without full SVD."""
        needs_full_recalib = False
        for layer_idx in range(self.num_layers):
            if layer_idx not in activations:
                continue
            H = activations[layer_idx].float()  # batch*seq_len × d_in
            V_old = self.activation_bases[layer_idx]
            V = V_old.clone()
            for _ in range(self.power_iter_steps):
                V = H.T @ (H @ V)
                V, _ = torch.linalg.qr(V)
            # Check subspace drift (LOTUS, arXiv:2602.01233)
            alignment = torch.trace(V.T @ V_old).abs() / self.r_calib
            if alignment < self.drift_threshold:
                needs_full_recalib = True
            self.activation_bases[layer_idx] = V
        if needs_full_recalib:
            self._calibrate_activation_bases_full()

    # -- Perturbation ----------------------------------------------------------

    def _get_perturbation(self, layer_idx):
        """Generate AGZO subspace perturbation.
        Bug fix: masking moved to update step — at perturbation time z_A is
        random, so sign comparison with momentum is meaningless."""
        layer = self.layers[layer_idx]
        A, B = layer['A'], layer['B']
        V_l = self.activation_bases[layer_idx].to(device=B.device)

        # B: AGZO subspace perturbation (r × d_in, in span(V_l))
        z_coeff_B = torch.randn(B.shape[0], V_l.shape[1], device=B.device)
        z_B = z_coeff_B @ V_l.T

        # A: projected perturbation (no masking here — applied after scoring)
        BV = B @ V_l  # r × r_calib
        Q, _ = torch.linalg.qr(BV)  # r × min(r, r_calib)
        z_coeff_A = torch.randn(A.shape[0], Q.shape[1], device=A.device)
        z_A = z_coeff_A @ Q.T  # d_out × r

        return z_A * self.eps, z_B * self.eps

    # -- Adapter Sync ----------------------------------------------------------

    def _sync_adapters(self, pos_layers=None, neg_layers=None):
        """Serialize PiSSA adapters directly to PEFT format."""
        def get_AB_list(overrides):
            A_list, B_list = [], []
            for layer in self.layers:
                idx = layer['layer_idx']
                if overrides and idx in overrides:
                    A_l, B_l = overrides[idx]
                else:
                    A_l, B_l = layer['A'], layer['B']
                A_list.append(A_l)
                B_list.append(B_l)
            return A_list, B_list

        A_pos, B_pos = get_AB_list(pos_layers)
        save_peft_adapter(A_pos, B_pos, self.adapter_dir_pos, self.layers)

        if neg_layers is not None:
            A_neg, B_neg = get_AB_list(neg_layers)
        else:
            A_neg, B_neg = A_pos, B_pos
        save_peft_adapter(A_neg, B_neg, self.adapter_dir_neg, self.layers)

    # -- Scoring ---------------------------------------------------------------

    def _get_prompt_logprobs(self, token_sequences, lora_request):
        prompts = [{"prompt_token_ids": seq} for seq in token_sequences]
        outputs = self.engine.generate(
            prompts, sampling_params=self.score_params, lora_request=lora_request,
        )
        return [
            extract_prompt_logprobs(out, seq)
            for out, seq in zip(outputs, token_sequences)
        ]

    def _score_contrastive(self, trajectories, advantages):
        """Score with asymmetric gradient regularization (arXiv:2602.18037).
        Bug fix: GR term is NLL divergence between θ+ and θ-, added only to
        loss_pos so it doesn't cancel in finite differences."""
        winner_tokens, loser_tokens = trajectories
        adv_w, adv_l = advantages
        prompt_len = self.prompt_len

        lp_pos = self._get_prompt_logprobs(
            [winner_tokens, loser_tokens], self.lora_pos
        )
        lp_pos = [lp[prompt_len:] for lp in lp_pos]
        lp_neg = self._get_prompt_logprobs(
            [winner_tokens, loser_tokens], self.lora_neg
        )
        lp_neg = [lp[prompt_len:] for lp in lp_neg]

        def nll(logprobs):
            # FP32 accumulation for numerical stability (arXiv:2602.10607)
            total = sum(float(-lp) for lp in logprobs)
            return total / max(len(logprobs), 1)

        # Advantage-weighted NLL
        loss_pos = float(adv_w) * nll(lp_pos[0]) + float(adv_l) * nll(lp_pos[1])
        loss_neg = float(adv_w) * nll(lp_neg[0]) + float(adv_l) * nll(lp_neg[1])

        # Asymmetric GR: NLL divergence between perturbation directions
        total_tokens = 0
        nll_div = 0.0
        for lps_p, lps_n in zip(lp_pos, lp_neg):
            for p, n in zip(lps_p, lps_n):
                nll_div += (float(-p) - float(-n)) ** 2
                total_tokens += 1
        if total_tokens > 0:
            nll_div /= total_tokens
        loss_pos += self.lambda_gr * nll_div  # Only to loss_pos

        return loss_pos, loss_neg

    # -- Exploration -----------------------------------------------------------

    def _update_temperature(self):
        """Entropy-guided cosine temperature annealing (arXiv:2505.18573)."""
        progress = self.step_count / self.total_steps
        self.explore_temperature = (
            self.temp_min
            + 0.5 * (self.temp_max - self.temp_min)
            * (1 + math.cos(math.pi * progress))
        )

    def _explore(self, batch):
        self._sync_adapters()
        gen_params = SamplingParams(
            n=self.num_candidates, temperature=self.explore_temperature
        )
        request_outputs = self.engine.generate(
            batch, sampling_params=gen_params, lora_request=self.lora_pos
        )
        prompt_token_ids = request_outputs[0].prompt_token_ids

        scored = [(out, self.score_fn(out.text))
                  for out in request_outputs[0].outputs]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_output, best_reward = scored[0]
        worst_output, worst_reward = scored[-1]

        all_rewards = [r for _, r in scored]
        total_reward = sum(all_rewards)
        adv_w = best_reward - (total_reward - best_reward) / (len(scored) - 1)
        adv_l = worst_reward - (total_reward - worst_reward) / (len(scored) - 1)

        # Entropy monitoring (arXiv:2510.08141)
        reward_range = best_reward - worst_reward
        if self.initial_entropy is None and reward_range > 0:
            self.initial_entropy = reward_range
        if (self.initial_entropy and reward_range > 0
                and reward_range < self.entropy_floor_ratio * self.initial_entropy):
            self.explore_temperature = min(
                self.explore_temperature * 1.5, self.temp_max
            )

        winner_full = list(prompt_token_ids) + list(best_output.token_ids)
        loser_full = list(prompt_token_ids) + list(worst_output.token_ids)
        self.prompt_len = len(prompt_token_ids)

        return ((winner_full, loser_full), (adv_w, adv_l))

    # -- ZO-Muon Update (arXiv:2602.17155) ------------------------------------

    def _zo_muon_update(self, param, grad, key):
        """ZO-Muon: momentum + Newton-Schulz spectral orthogonalization."""
        if key not in self.momentum_buffers:
            self.momentum_buffers[key] = torch.zeros_like(param)
        buf = self.momentum_buffers[key]
        buf.mul_(self.momentum).add_((1 - self.momentum) * grad)

        # Newton-Schulz orthogonalization extracts spectral descent directions
        orth_update = newton_schulz_orthogonalize(buf)
        param.sub_(self.eta * orth_update)

    # -- Adaptive Epsilon ------------------------------------------------------

    def _update_eps(self):
        """Scale epsilon proportionally to loss EMA (arXiv:2602.10607)."""
        if self.loss_ema is not None and self.initial_loss_ema is not None:
            ratio = max(self.loss_ema / self.initial_loss_ema, self.eps_floor)
            self.eps = self.eps_base * ratio

    # -- KL Constraint ---------------------------------------------------------

    def _apply_kl_constraint(self, loss_pos, loss_neg):
        """Scale down LR if update exceeds KL budget (arXiv:2602.04879)."""
        kl_approx = abs(loss_pos - loss_neg)
        if kl_approx > self.delta_kl:
            scale = self.delta_kl / kl_approx
            return self.eta * scale
        return self.eta

    # -- Cosine LR -------------------------------------------------------------

    def _update_lr(self):
        progress = self.step_count / self.total_steps
        self.eta = (self.eta_min
                    + 0.5 * (self.eta_max - self.eta_min)
                    * (1 + math.cos(math.pi * progress)))

    # -- Health ----------------------------------------------------------------

    def _check_health(self, loss_pos, loss_neg):
        avg_nll = (abs(loss_pos) + abs(loss_neg)) / 2
        if self.loss_ema is None:
            self.loss_ema = avg_nll
            self.initial_loss_ema = avg_nll
            return True
        if self.loss_ema > 1e-8 and avg_nll > self.spike_threshold * self.loss_ema:
            return False
        self.loss_ema = (self.loss_ema_momentum * self.loss_ema
                         + (1 - self.loss_ema_momentum) * avg_nll)
        return True

    # -- Directional Derivative Clipping (arXiv:2505.13430) --------------------

    def _clip_dd(self, dd):
        """Clip directional derivative to prevent outlier spikes."""
        if self.dd_ema is None:
            self.dd_ema = abs(dd)
            return dd
        clip_val = self.dd_clip_multiplier * self.dd_ema
        clipped = max(-clip_val, min(dd, clip_val))
        self.dd_ema = 0.95 * self.dd_ema + 0.05 * abs(dd)
        return clipped

    # -- Main Training Step ----------------------------------------------------

    def step(self, batch):
        self.step_count += 1

        # Exploration with entropy-guided temperature
        result = self._explore(batch)
        if result is None:
            return
        trajectories, advantages = result

        # Update activation bases via power iteration on current batch
        batch_activations = extract_layer_activations(
            self.engine, batch, self.num_layers
        )
        self._update_activation_bases_power_iter(batch_activations)

        # Generate perturbations for all layers
        perturbations = {}
        for layer in self.layers:
            idx = layer['layer_idx']
            z_A, z_B = self._get_perturbation(idx)
            perturbations[idx] = (z_A, z_B)

        # Construct perturbed adapters (all layers simultaneously)
        pos_layers, neg_layers = {}, {}
        for layer in self.layers:
            idx = layer['layer_idx']
            z_A, z_B = perturbations[idx]
            pos_layers[idx] = (layer['A'] + z_A, layer['B'] + z_B)
            neg_layers[idx] = (layer['A'] - z_A, layer['B'] - z_B)

        # Score (4 prefills total, FP32 accumulation)
        self._sync_adapters(pos_layers, neg_layers)
        loss_pos, loss_neg = self._score_contrastive(
            trajectories, advantages
        )

        # Update with safety checks
        if self._check_health(loss_pos, loss_neg):
            # FP32 directional derivative with clipping
            raw_dd = float(loss_pos - loss_neg) / (2.0 * self.eps)
            dd = self._clip_dd(raw_dd)

            # KL constraint — scale eta if update too large
            effective_eta = self._apply_kl_constraint(loss_pos, loss_neg)
            saved_eta = self.eta
            self.eta = effective_eta

            for layer in self.layers:
                idx = layer['layer_idx']
                z_A, z_B = perturbations[idx]
                grad_A = dd * z_A
                grad_B = dd * z_B

                # Momentum-aligned masking at update step (arXiv:2602.15322)
                # Bug fix: applied to grad_A (not z_A) after scoring
                key_A = (idx, 'A')
                if (self.step_count > self.mask_warmup_steps
                        and key_A in self.momentum_buffers):
                    mask = (grad_A.sign() == self.momentum_buffers[key_A].sign()).float()
                    grad_A = grad_A * mask

                self._zo_muon_update(layer['A'], grad_A, (idx, 'A'))
                self._zo_muon_update(layer['B'], grad_B, (idx, 'B'))

            self.eta = saved_eta

        # Schedule updates
        self._update_lr()
        self._update_eps()
        self._update_temperature()

    def train(self, dataloader, num_steps):
        for step_idx, batch in zip(range(num_steps), dataloader):
            self.step(batch)


def initialize_pissa(model_config, layer_idx):
    W0 = load_pretrained_weights(model_config, layer_idx)
    r = model_config.rank
    U, S, Vt = torch.svd_lowrank(W0, q=r, niter=2)
    sqrt_S = torch.sqrt(S[:r])
    A = U[:, :r] * sqrt_S.unsqueeze(0)
    B = sqrt_S.unsqueeze(1) * Vt[:r, :]
    W_res = W0 - A @ B
    W_res_quantized = quantize_nf4(W_res)
    return A, B, W_res_quantized
```

---

## 9. Memory Budget (Single H100 80GB, BF16)

| Component | VRAM |
|:----------|:-----|
| Residual Model ($W^{res}$, NF4) | ~35 GB |
| KV Cache (PagedAttention, BF16) | ~27 GB |
| LoRA Adapter Slots (2×, BF16) | ~1.6 GB |
| Controller State (FP32 masters + momentum + ZO-Muon G buffers) | ~0.22 GB |
| Activation Subspace Tracker ($V_l$ per layer) | ~26 MB |
| CUDA Graph Capture Overhead | ~0.5 GB |
| **Headroom** | **~15.4 GB** |

Adapter staging on `/dev/shm` (111 GB tmpfs) — zero VRAM cost, zero disk I/O.

Launch:
```bash
sudo nvidia-smi -lgc 1980,1980 && sudo nvidia-smi -lmc 2619  # Lock clocks

export DS_MEZO_ADAPTER_DIR=/dev/shm/ds_mezo
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=true

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.92 \
    --enable-lora \
    --max-lora-rank 64 \
    --enforce-eager False \
    --max-num-seqs 8
```

---

## 10. Hyperparameters

### 10.1 Complete Inventory

| # | Symbol | Code Variable | Default | Class |
|:--|:-------|:--------------|:--------|:------|
| 1 | $\eta_{max}$ | `self.eta_max` | $10^{-4}$ | **Primary** |
| 2 | $r$ | `model_config.rank` | 16 | **Primary** (architecture) |
| 3 | $\epsilon_0$ | `self.eps_base` | $10^{-3}/\sqrt{r}$ | Derived from $r$ |
| 4 | $\epsilon_{floor}$ | `self.eps_floor` | 0.1 | Robust default |
| 5 | $\mu$ | `self.momentum` | 0.9 | Standard |
| 6 | $N$ | `self.num_candidates` | 4 | Robust default |
| 7 | $\lambda_{GR}$ | `self.lambda_gr` | 0.01 | Robust default |
| 8 | $\delta_{KL}$ | `self.delta_kl` | 0.01 | Robust default |
| 9 | $r_{calib}$ | `self.r_calib` | 8 | Fixed constant |
| 10 | $T_{max}$ | `self.temp_max` | 1.0 | Robust default |
| 11 | $T_{min}$ | `self.temp_min` | 0.3 | Robust default |
| 12 | $\tau_{drift}$ | `self.drift_threshold` | 0.95 | Robust default |
| 13 | spike threshold | `self.spike_threshold` | 5.0 | Safety bound |
| 14 | DD clip multiplier | `self.dd_clip_multiplier` | 3.0 | Safety bound |
| 15 | EMA momentum | `self.loss_ema_momentum` | 0.95 | Low-sensitivity |
| 16 | mask warmup | `self.mask_warmup_steps` | 10 | Fixed constant |

**Total: 16 parameters** (2 primary, 1 derived, 13 robust defaults/constants). The increase from 11 to 16 reflects the additional safety mechanisms; no new parameters require tuning.

### 10.2 Scaling Guidelines

| Parameter | Guideline |
|:----------|:----------|
| $r$ | Start at 16. Increase to 32 for >30B models or complex tasks. |
| $\eta_{max}$ | Start at $10^{-4}$. Scale down by $\sqrt{2}$ if loss diverges in first 50 steps. |
| $\epsilon_0$ | Auto-derived: $10^{-3}/\sqrt{r}$. Adapts automatically via loss EMA ratio. |
| $\lambda_{GR}$ | 0.01 default. Increase to 0.1 if reward hacking observed. |

All other parameters use fixed defaults across architectures (Llama, Qwen, Mistral, DeepSeek) and tasks (math, code, instruction following, tool use).

---

## 11. Convergence

With AGZO subspace restriction, ZO convergence depends on subspace alignment $\alpha$ rather than raw dimensionality (Park et al. 2501.19099):

$$ \mathbb{E}[\|\nabla L(\theta_t)\|^2] \leq \mathcal{O}\left(\frac{d_{eff}}{\alpha \cdot T} + \frac{\sigma^2}{B}\right) $$

AGZO achieves $\alpha \approx 1$ (gradient lies in activation subspace for linear layers), so convergence scales with $d_{eff} \approx 2.65M$ rather than raw $d \approx 21M$. Per-step power iteration (§3.5) maintains $\alpha \approx 1$ throughout training by tracking the evolving activation subspace. ZO-Muon's spectral orthogonalization (§5) further improves the effective convergence rate by extracting informative descent directions from noisy gradient estimates, achieving the same accuracy as standard MeZO with 24.7% of the queries (arXiv:2602.17155).

The implicit flat-minima regularization (Zhang et al. 2506.05454) provides:

$$ f_\epsilon(\theta) = f(\theta) + \frac{\epsilon^2}{2}\operatorname{Tr}(\nabla^2 f(\theta)) + \mathcal{O}(\epsilon^4) $$

ZO converges to $(\mathcal{O}(\epsilon/d^2), \epsilon)$-approximate flat minima after $T = \mathcal{O}(d^4/\epsilon^2)$ iterations in the worst case. With subspace restriction, $d$ is replaced by $d_{eff}$.

---

## 12. Risks

| Failure Mode | Mitigation |
|:-------------|:-----------|
| BF16 noise amplification in SPSA | BF16 forward passes (§6.2.1) + FP32 loss computation + adaptive $\epsilon$ + directional derivative clipping (§2.1) |
| High ZO variance (all-at-once) | AGZO subspace reduces $d_{eff}$ by 8×; ZO-Muon spectral denoising (§5) |
| Trajectory divergence | Trajectory locking (generate once, score fixed) |
| Mode collapse | Gradient regularization (actual ZO signal) + KL constraint (§4.3) |
| BF16 update truncation | FP32 master accumulation + BF16 dynamic range (§6.2.1) |
| Stale activation bases | Per-step power iteration + adaptive full recalibration on drift (§3.5) |
| Freezing important PiSSA params | Momentum-aligned masking replaces magnitude masking (§3.3) |
| Momentum bias from correlated perturbations | ZO-Muon spectral orthogonalization (§5) |
| Exploration-exploitation imbalance | Entropy-guided temperature annealing (§7.3) |

**Requirements:**
- GPU: NVIDIA H100 80GB HBM3 (SM 9.0) or BF16-capable GPU (SM ≥ 8.0)
- CUDA ≥ 12.8, PyTorch ≥ 2.7 (for `torch.compile` with `reduce-overhead`)
- `triton >= 3.3`
- vLLM with `enable_lora=True` and `LoRARequest(load_inplace=True)` support (v0.17+)
- vLLM model runner hook for activation extraction
- Linear layers compatible with LoRA
- tmpfs mount (e.g., `/dev/shm`) for adapter staging (recommended ≥ 1 GB)

---

## 13. Future Extensions

- **K-sample SPSA averaging:** Sample $K$ independent perturbation directions per step, average gradient estimates. Costs $K \times 4$ prefills per step. Trades compute for lower variance. Combined with BSZO's Kalman filtering (arXiv:2601.01452), this would provide Bayesian aggregation of multiple gradient observations per step.
- **All-N trajectory scoring:** Score all $N$ candidates (not just winner/loser) with advantage weighting. Doubles prefill cost but may halve convergence steps.
- **James-Stein shrinkage:** When multi-prompt batching ($B \geq 3$) is available, apply James-Stein shrinkage across per-prompt RLOO baselines for variance reduction.
- **TAMPO meta-temperature:** Replace cosine temperature schedule with learned temperature meta-policy (arXiv:2602.11779, ICLR 2026). Outer loop optimizes distribution over candidate temperatures by rewarding those that maximize high-advantage trajectory likelihood.
- **Fisher-information masking:** Replace momentum-aligned masking with empirical Fisher information (arXiv:2406.02913) for provably optimal parameter selection. Compute Fisher from ZO gradient history; select top-$k$ parameters by Fisher magnitude regardless of weight magnitude. More expensive but theoretically grounded.
- **LOTUS adaptive subspace switching:** Replace drift-threshold-based full recalibration with LOTUS's (arXiv:2602.01233) path-efficiency criterion, which provably converges in fewer iterations than any fixed-interval policy (Theorem 3.2).
- **Hierarchical ZO decomposition:** HZO (arXiv:2602.10607) decomposes the network depth dimension via divide-and-conquer, computing ZO gradients hierarchically. Would address residual numerical stability concerns for very deep models (>100 layers).

---

## 14. References

| Paper | Role |
|:------|:-----|
| PiSSA (arXiv:2404.02948) | SVD adapter initialization |
| AGZO (arXiv:2601.17261) | Activation-guided subspace perturbation, per-step power iteration |
| Sparse MeZO (NUS-HPC-AI-Lab) | Original magnitude-based masking (superseded by Magma) |
| Zhang et al. (arXiv:2506.05454) | ZO flat minima theory |
| Park et al. (arXiv:2501.19099) | Subspace alignment convergence |
| BSZO (arXiv:2601.01452) | Bayesian subspace ZO, residual adaptation |
| MeZO-A³dam (OpenReview:OBIuFjZzmp) | Weak adaptivity hypothesis |
| ZO-Muon (arXiv:2602.17155) | **Spectral gradient orthogonalization (now core optimizer)** |
| Zeng et al. (arXiv:2511.03710) | James-Stein shrinkage for RLVR |
| S-LoRA / vLLM | Multi-adapter serving, PagedAttention |
| **Magma (arXiv:2602.15322)** | **Momentum-aligned gradient masking (replaces magnitude masking)** |
| **Gradient Regularization (arXiv:2602.18037)** | **Reward-gradient norm penalty for ZO (replaces KL penalty)** |
| **DPPO (arXiv:2602.04879)** | **Divergence-based trust region / KL constraint design** |
| **LOTUS (arXiv:2602.01233)** | **Adaptive subspace switching criterion** |
| **HZO (arXiv:2602.10607)** | **FP32 minimum precision requirement for ZO; hierarchical decomposition** |
| **QZO (arXiv:2505.13430)** | **Directional derivative clipping for ZO stability** |
| **Entropy annealing (arXiv:2505.18573)** | **Temperature schedule for GRPO-style exploration** |
| **AEPO (arXiv:2510.08141)** | **Entropy collapse diagnosis and control in LLM RL** |
| **TAMPO (arXiv:2602.11779)** | **Learned temperature meta-policy (ICLR 2026)** |
| KL-regularized RL collapse (arXiv:2510.20817) | Theoretical basis for replacing KL penalty |
| Rethinking KL in RLHF (arXiv:2510.01555) | KL gradient properties analysis |
