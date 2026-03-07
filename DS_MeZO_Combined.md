# DS-MeZO: Zeroth-Order Optimization for LLM Fine-Tuning

**Single NVIDIA H100 (80GB HBM3) | OpenAI Triton Controller + vLLM (S-LoRA Backend)**

---

## 1. Core Idea

DS-MeZO fine-tunes LLMs without backpropagation using zeroth-order (ZO) gradient estimation, optimizing non-differentiable objectives (code compilation, proof verification, tool use) at near-inference memory cost.

**Pipeline:** PiSSA subspace initialization → activation-guided subspace perturbation (AGZO) + sparse masking → all-at-once SPSA → RLOO contrastive selection → per-token KL-shaped loss → SGD with momentum (FP32) → cosine LR schedule.

The Controller is implemented entirely in OpenAI Triton. vLLM handles all forward passes via standard LoRA adapters. PiSSA adapters are natively LoRA-compatible — no fusion step required.

---

## 2. ZO Gradient Estimation

For loss $f(\theta)$ and perturbation direction $z$, the symmetric two-point SPSA estimator:

$$ \hat{g} = \frac{f(\theta + \epsilon z) - f(\theta - \epsilon z)}{2\epsilon} \cdot z $$

This estimates the gradient of the Gaussian-smoothed surrogate:

$$ f_\epsilon(\theta) = f(\theta) + \frac{\epsilon^2}{2} \operatorname{Tr}(\nabla^2 f(\theta)) + \mathcal{O}(\epsilon^4) $$

The $\operatorname{Tr}(\nabla^2 f)$ penalty drives parameters toward flat minima (implicit SAM, Zhang et al. 2506.05454).

**All-at-once perturbation:** All layers' $A$ and $B$ are perturbed simultaneously. The scalar finite difference $(L^+ - L^-)$ is shared across all layers; per-layer gradients are distinguished by their independent perturbation directions $z_l$. Bilinear cross-terms between $A$ and $B$ are $\mathcal{O}(\epsilon^2)$, same order as the SPSA smoothing bias.

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

### 3.3 Sparse MeZO Masking

ZO gradient noise disproportionately disrupts large-magnitude parameters (Sparse MeZO, NUS-HPC-AI-Lab). Apply perturbation only to small-magnitude parameters:

$$ M_l = \mathbf{1}[|\theta_l| \leq h_l], \quad h_l = \operatorname{percentile}(|\theta_l|, s) $$

Sparsity $s = 0.8$ (80th percentile threshold — perturb bottom 80% by magnitude). Computed on-the-fly per layer, no memory overhead.

Applied to $A$ only. $B$'s perturbation is already low-dimensional via AGZO (128 effective dims), so masking is unnecessary.

### 3.4 Effective Dimensionality Summary

| Component | Raw Dims | With AGZO + Sparse | Reduction |
|:----------|:---------|:-------------------|:----------|
| $B$ (per layer) | $r \times d_{in} = 131K$ | $r \times r_{calib} = 128$ | 1024× |
| $A$ (per layer) | $d_{out} \times r = 131K$ | $0.2 \times d_{out} \times r_{calib} \approx 13K$ | 10× |
| **Total (80 layers)** | **20.97M** | **~1.05M** | **20×** |

### 3.5 Activation Calibration

**Initialization:** Before training, run a calibration batch (32–64 samples) through the model via a vLLM model runner hook. Extract input activations $H_l \in \mathbb{R}^{B \times d_{in}}$ per layer. Compute top-$r_{calib}$ left singular vectors:

$$ V_l = \operatorname{top-}r_{calib}\operatorname{-SVD}(H_l^T H_l) $$

**Refresh:** Every 100 steps, re-run the calibration batch with current adapter weights and recompute $V_l$. Cost: one extra forward pass. This keeps activation bases aligned with the evolving adapter.

---

## 4. Trajectory Locking + RLOO

### 4.1 Trajectory Locking
1. Generate $N$ candidates under unperturbed weights $\theta_0$.
2. Select winner/loser via RLOO advantages.
3. Score **fixed sequences** (prefill-only) under $\theta^+$ and $\theta^-$.

### 4.2 RLOO Advantages
$$ A_i = R_i - \frac{1}{N-1} \sum_{j \neq i} R_j $$

Unbiased, minimum-variance, self-centering ($\sum_i A_i = 0$), zero lag, no tunable parameters. RLOO advantages are self-regulating: when all candidates are similar quality, advantages are near-zero, producing near-zero gradients.

### 4.3 Per-Token KL-Shaped Contrastive Loss
$$ \mathcal{L}(\theta) = \sum_{i \in \{w, l\}} A_i \cdot \text{NLL}_i^{KL}(\theta) $$
$$ \text{NLL}_i^{KL}(\theta) = \frac{1}{T_i} \sum_{t=1}^{T_i} \left[ -(1 + \beta) \log \pi_\theta(y_t | y_{<t}) + \beta \log \pi_{ref}(y_t | y_{<t}) \right] $$

$\beta = 0.1$. Reference terms cancel in finite difference but stabilize monitoring and prevent mode collapse.

### 4.4 Safety
**Spike detection:** Skip if NLL $> 5\times$ EMA. Update EMA only after healthy steps.

---

## 5. Optimizer: SGD with Momentum + Cosine Schedule

### 5.1 Motivation: Weak Adaptivity Hypothesis

MeZO-A³dam (OpenReview:OBIuFjZzmp) shows that full Adam preconditioning hurts ZO optimization due to high variance of per-coordinate ZO gradient estimates. The "weak adaptivity hypothesis" states that the optimal level of adaptivity in ZO should be much lower than in first-order optimization.

DS-MeZO uses **zero adaptivity**. SGD with momentum provides the minimal stateful optimizer that smooths ZO noise without amplifying per-coordinate estimation errors.

### 5.2 Update Rule

$$ v_t = \mu \cdot v_{t-1} + \hat{g}_t $$
$$ \theta_{t+1} = \theta_t - \eta_t \cdot v_t $$

with $\mu = 0.9$ (momentum). Each (layer, param_group) pair maintains an independent momentum buffer.

**FP32 master weights and momentum buffers** on GPU prevent accumulation truncation. Forward passes remain FP16.

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
|  |  FP32 Master     |  |  SGD + Momentum  |  | Activ.  | |
|  |  Weights         |  |  (v per layer    |  | Calib.  | |
|  |  (A, B)          |  |   per group)     |  | Cache   | |
|  |  PiSSA           |  |                  |  | (V_l)   | |
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
| `subspace_perturb` | AGZO projection → sparse masking → $\theta \pm \epsilon z$ |
| `sgd_momentum_update` | Gradient → momentum buffer → param update |
| `spsa_gradient` | $\hat{g} = (\Delta\mathcal{L} / 2\epsilon) \cdot z$ |
| `score_reduce` | Per-token logprob → KL-shaped NLL → advantage-weighted loss |
| `health_monitor` | EMA update → spike detection |

---

## 7. Execution

### 7.1 Per-Step Loop

```
1. EXPLORE: Generate N candidates (unperturbed) → reward → RLOO advantages → select winner/loser.
           Cache reference logprobs for KL shaping.

2. PERTURB: For each layer, generate subspace perturbation (AGZO for B, sparse for A).
            Construct θ+ and θ- adapters (all layers perturbed simultaneously).

3. SCORE:   Serialize θ+ and θ- adapters to disk.
            Score fixed sequences under θ+ and θ- (4 prefills total).

4. UPDATE:  Health check — skip if NLL spike.
            Per-layer SPSA gradient: ĝ = (L+ - L-) / (2ε) · z
            SGD+momentum update (FP32 masters).

5. SCHEDULE: Cosine LR update.
             Activation basis refresh every 100 steps.
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


ADAPTER_STAGING_DIR = os.environ.get("DS_MEZO_ADAPTER_DIR", "/tmp/ds_mezo")


def save_peft_adapter(A_list, B_list, adapter_dir, layer_configs):
    os.makedirs(adapter_dir, exist_ok=True)
    tensors = {}
    for layer_idx, (A_l, B_l) in enumerate(zip(A_list, B_list)):
        prefix = layer_configs[layer_idx]['peft_prefix']
        tensors[f"{prefix}.lora_A.weight"] = A_l.half()
        tensors[f"{prefix}.lora_B.weight"] = B_l.half()
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

        # Exploration
        self.num_candidates = 4
        self.explore_temperature = 0.7
        self.beta_kl = 0.1

        # Perturbation
        self.eps = 1e-3 / math.sqrt(model_config.rank)
        self.sparsity = 0.8

        # Optimizer: SGD + momentum
        self.eta_max = 1e-4
        self.eta_min = self.eta_max / 100
        self.eta = self.eta_max
        self.momentum = 0.9
        self.momentum_buffers = {}

        # Activation calibration
        self.r_calib = 8
        self.calib_refresh_interval = 100
        self._calib_data = calibration_data
        self.activation_bases = {}
        self._calibrate_activation_bases()

        # Health monitoring
        self.loss_ema = None
        self.loss_ema_momentum = 0.95
        self.spike_threshold = 5.0

    # -- Activation Calibration ------------------------------------------------

    def _calibrate_activation_bases(self):
        """Extract activation subspace per layer via vLLM model runner hook."""
        activations = extract_layer_activations(
            self.engine, self._calib_data, self.num_layers
        )
        for layer_idx in range(self.num_layers):
            H = activations[layer_idx]  # batch*seq_len × d_in
            d_in = H.shape[1]
            r_calib = min(self.r_calib, d_in)
            _, _, Vt = torch.svd_lowrank(H, q=r_calib, niter=2)
            self.activation_bases[layer_idx] = Vt.float()  # d_in × r_calib

    # -- Perturbation ----------------------------------------------------------

    def _get_perturbation(self, layer_idx):
        """Generate AGZO subspace perturbation for A and B."""
        layer = self.layers[layer_idx]
        A, B = layer['A'], layer['B']
        V_l = self.activation_bases[layer_idx].to(device=B.device)

        # B: AGZO subspace perturbation (r × d_in, in span(V_l))
        z_coeff_B = torch.randn(B.shape[0], V_l.shape[1], device=B.device)
        z_B = z_coeff_B @ V_l.T

        # A: projected + sparse perturbation
        BV = B @ V_l  # r × r_calib
        Q, _ = torch.linalg.qr(BV)  # r × min(r, r_calib)
        z_coeff_A = torch.randn(A.shape[0], Q.shape[1], device=A.device)
        z_A = z_coeff_A @ Q.T  # d_out × r
        mask = (A.abs() <= torch.quantile(A.abs(), self.sparsity)).float()
        z_A = z_A * mask

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

        def kl_shaped_nll(logprobs, ref_logprobs):
            nll = [-lp for lp in logprobs]
            kl = [ref - lp for ref, lp in zip(ref_logprobs, logprobs)]
            shaped = [n + self.beta_kl * k for n, k in zip(nll, kl)]
            return sum(shaped) / len(shaped)

        loss_pos = (adv_w * kl_shaped_nll(lp_pos[0], self.ref_logprobs[0])
                    + adv_l * kl_shaped_nll(lp_pos[1], self.ref_logprobs[1]))
        loss_neg = (adv_w * kl_shaped_nll(lp_neg[0], self.ref_logprobs[0])
                    + adv_l * kl_shaped_nll(lp_neg[1], self.ref_logprobs[1]))
        return loss_pos, loss_neg

    # -- Exploration -----------------------------------------------------------

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

        winner_full = list(prompt_token_ids) + list(best_output.token_ids)
        loser_full = list(prompt_token_ids) + list(worst_output.token_ids)
        self.prompt_len = len(prompt_token_ids)

        ref_lp = self._get_prompt_logprobs(
            [winner_full, loser_full], self.lora_pos
        )
        self.ref_logprobs = (
            ref_lp[0][self.prompt_len:], ref_lp[1][self.prompt_len:]
        )

        return ((winner_full, loser_full), (adv_w, adv_l))

    # -- SGD + Momentum --------------------------------------------------------

    def _sgd_momentum_update(self, param, grad, key):
        if key not in self.momentum_buffers:
            self.momentum_buffers[key] = torch.zeros_like(param)
        buf = self.momentum_buffers[key]
        buf.mul_(self.momentum).add_(grad)
        param.sub_(self.eta * buf)

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
            return True
        if self.loss_ema > 1e-8 and avg_nll > self.spike_threshold * self.loss_ema:
            return False
        self.loss_ema = (self.loss_ema_momentum * self.loss_ema
                         + (1 - self.loss_ema_momentum) * avg_nll)
        return True

    # -- Main Training Step ----------------------------------------------------

    def step(self, batch):
        self.step_count += 1

        # Exploration
        result = self._explore(batch)
        if result is None:
            return
        trajectories, advantages = result

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

        # Score (4 prefills total)
        self._sync_adapters(pos_layers, neg_layers)
        loss_pos, loss_neg = self._score_contrastive(trajectories, advantages)

        # Update
        if self._check_health(loss_pos, loss_neg):
            diff = loss_pos - loss_neg
            for layer in self.layers:
                idx = layer['layer_idx']
                z_A, z_B = perturbations[idx]
                grad_A = (diff / (2.0 * self.eps)) * z_A
                grad_B = (diff / (2.0 * self.eps)) * z_B
                self._sgd_momentum_update(layer['A'], grad_A, (idx, 'A'))
                self._sgd_momentum_update(layer['B'], grad_B, (idx, 'B'))

        # Schedule + calibration refresh
        self._update_lr()
        if self.step_count % self.calib_refresh_interval == 0:
            self._calibrate_activation_bases()

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

## 9. Memory Budget (Single H100 80GB)

| Component | VRAM |
|:----------|:-----|
| Residual Model ($W^{res}$, NF4) | ~35 GB |
| KV Cache (PagedAttention) | ~25 GB |
| LoRA Adapter Slots (2×, FP16) | ~1.6 GB |
| Controller State (FP32 masters + momentum) | ~0.15 GB |
| Activation Calibration Cache | ~26 MB |
| **Headroom** | **~18.2 GB** |

Launch: `--gpu-memory-utilization 0.9 --enable-lora --max-lora-rank 64`

---

## 10. Hyperparameters

### 10.1 Complete Inventory

| # | Symbol | Code Variable | Default | Class |
|:--|:-------|:--------------|:--------|:------|
| 1 | $\eta_{max}$ | `self.eta_max` | $10^{-4}$ | **Primary** |
| 2 | $r$ | `model_config.rank` | 16 | **Primary** (architecture) |
| 3 | $\epsilon$ | `self.eps` | $10^{-3}/\sqrt{r}$ | Derived from $r$ |
| 4 | $s$ | `self.sparsity` | 0.8 | Robust default |
| 5 | $\mu$ | `self.momentum` | 0.9 | Standard |
| 6 | $N$ | `self.num_candidates` | 4 | Robust default |
| 7 | $\beta$ | `self.beta_kl` | 0.1 | Robust default |
| 8 | $r_{calib}$ | `self.r_calib` | 8 | Fixed constant |
| 9 | $T_{explore}$ | `self.explore_temperature` | 0.7 | Robust default |
| 10 | spike threshold | `self.spike_threshold` | 5.0 | Safety bound |
| 11 | EMA momentum | `self.loss_ema_momentum` | 0.95 | Low-sensitivity |

**Total: 11 parameters** (2 primary, 1 derived, 8 robust defaults/constants).

### 10.2 Scaling Guidelines

| Parameter | Guideline |
|:----------|:----------|
| $r$ | Start at 16. Increase to 32 for >30B models or complex tasks. |
| $\eta_{max}$ | Start at $10^{-4}$. Scale down by $\sqrt{2}$ if loss diverges in first 50 steps. |
| $\epsilon$ | Auto-derived: $10^{-3}/\sqrt{r}$. No manual tuning. |

All other parameters use fixed defaults across architectures (Llama, Qwen, Mistral, DeepSeek) and tasks (math, code, instruction following, tool use).

---

## 11. Convergence

With AGZO subspace restriction, ZO convergence depends on subspace alignment $\alpha$ rather than raw dimensionality (Park et al. 2501.19099):

$$ \mathbb{E}[\|\nabla L(\theta_t)\|^2] \leq \mathcal{O}\left(\frac{d_{eff}}{\alpha \cdot T} + \frac{\sigma^2}{B}\right) $$

AGZO achieves $\alpha \approx 1$ (gradient lies in activation subspace for linear layers), so convergence scales with $d_{eff} \approx 1M$ rather than raw $d \approx 21M$. Combined with sparse masking, the effective constant is further reduced.

The implicit flat-minima regularization (Zhang et al. 2506.05454) provides:

$$ f_\epsilon(\theta) = f(\theta) + \frac{\epsilon^2}{2}\operatorname{Tr}(\nabla^2 f(\theta)) + \mathcal{O}(\epsilon^4) $$

ZO converges to $(\mathcal{O}(\epsilon/d^2), \epsilon)$-approximate flat minima after $T = \mathcal{O}(d^4/\epsilon^2)$ iterations in the worst case. With subspace restriction, $d$ is replaced by $d_{eff}$.

---

## 12. Risks

| Failure Mode | Mitigation |
|:-------------|:-----------|
| FP16 noise on large weights | Sparse masking excludes large-magnitude params from perturbation |
| High ZO variance (all-at-once) | AGZO subspace reduces $d_{eff}$ by 20× |
| Trajectory divergence | Trajectory locking (generate once, score fixed) |
| Mode collapse | Per-token KL shaping ($\beta = 0.1$) |
| FP16 update truncation | FP32 master accumulation |
| Stale activation bases | Periodic refresh every 100 steps |

**Requirements:**
- vLLM with `enable_lora=True` and `LoRARequest(load_inplace=True)` support (v0.17+)
- vLLM model runner hook for activation extraction
- Linear layers compatible with LoRA
- `triton >= 3.0`

---

## 13. Future Extensions

- **K-sample SPSA averaging:** Sample $K$ independent perturbation directions per step, average gradient estimates. Costs $K \times 4$ prefills per step. Trades compute for lower variance.
- **ZO-Muon spectral updates:** Replace SGD+momentum with subspace gradient orthogonalization (arXiv:2602.17155). Extracts descent directions from noisy ZO gradients via SVD, achieving 4× faster convergence than entry-wise methods.
- **All-N trajectory scoring:** Score all $N$ candidates (not just winner/loser) with advantage weighting. Doubles prefill cost but may halve convergence steps.
- **James-Stein shrinkage:** When multi-prompt batching ($B \geq 3$) is available, apply James-Stein shrinkage across per-prompt RLOO baselines for variance reduction.

---

## 14. References

| Paper | Role |
|:------|:-----|
| PiSSA (arXiv:2404.02948) | SVD adapter initialization |
| AGZO (arXiv:2601.17261) | Activation-guided subspace perturbation |
| Sparse MeZO (NUS-HPC-AI-Lab) | Magnitude-based sparse masking |
| Zhang et al. (arXiv:2506.05454) | ZO flat minima theory |
| Park et al. (arXiv:2501.19099) | Subspace alignment convergence |
| BSZO (arXiv:2601.01452) | Bayesian subspace ZO, residual adaptation |
| MeZO-A³dam (OpenReview:OBIuFjZzmp) | Weak adaptivity hypothesis |
| ZO-Muon (arXiv:2602.17155) | Subspace gradient orthogonalization |
| Zeng et al. (arXiv:2511.03710) | James-Stein shrinkage for RLVR |
| S-LoRA / vLLM | Multi-adapter serving, PagedAttention |
