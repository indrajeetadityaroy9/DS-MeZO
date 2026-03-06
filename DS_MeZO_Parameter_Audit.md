# DS-MeZO Parameter Audit

**Built against:** `DS_MeZO_Master.md` (post-5-fix rewrite: RLOO, per-token KL, layer-adaptive eps, in-place weights, functional rank allocation)

**Audit date:** 2026-03-06

---

## Classification Key

| Tag | Meaning |
|:----|:--------|
| **CORE-THEORY** | Derived from mathematical proofs in Part I. Changing breaks theoretical guarantees. |
| **CORE-DESIGN** | Architectural decisions central to the DS-MeZO contribution. Not derived from a single equation but essential to the system's identity. |
| **TUNABLE-SEED** | Primary user-facing hyperparameters that control optimization behavior. Few in number by design. |
| **AUTO-ADAPTED** | Computed automatically from data, weights, or other parameters. No user tuning required. |
| **HEURISTIC** | Reasonable defaults from practitioner experience. Task-insensitive but adjustable. |
| **SAFETY** | Guards against divergence, NaN, or reward collapse. |
| **ENGINEERING** | Implementation choices (memory layout, API usage, quantization format). No theoretical impact. |
| **IMPLICIT** | Unstated assumptions baked into the design. Not exposed as knobs. |

---

## 1. Zeroth-Order Estimation

### 1.1 Perturbation Radius

| # | Parameter | Value | Source | Classification | Role & Impact |
|---|-----------|-------|--------|----------------|---------------|
| 1 | `eps_base` (perturbation radius) | `2.5e-4` | Sec 13.1 L701: `eps_base = 1e-3 / sqrt(r)` at r=16; Code L355 | **TUNABLE-SEED** | Sets the scale of the finite-difference probe. Too small: FP16 truncation kills the gradient (Failure Mode C). Too large: breaks smoothness bound, causes deep-layer divergence (Failure Mode D). This is the single most sensitive scalar — it controls gradient signal strength, implicit Hessian regularization weight, and the bias-variance tradeoff. |
| 2 | Element-wise override formula | `eps_ij = max(eps_l, abs(theta_ij) * 0.001)` | Sec 7 L147; Code L446-451 | **CORE-THEORY** | Proven to preserve SPSA unbiasedness (Sec 7, L137-139). The `0.001` multiplier is a **SAFETY** sub-component guaranteeing perturbation exceeds FP16 ULP. |
| 3 | ULP margin multiplier | `0.001` | Sec 7 L135; Code L446 | **SAFETY** | Conservative heuristic. Hard requirement is `eps_ij > ULP(theta_ij)` in FP16. For FP16's dynamic range, 0.001 satisfies this for all values with magnitude > 0.03. |
| 4 | Smoothness upper bound | `eps^2 <= sqrt(2)*L1 / (r^(3/2)*L3)` | Sec 13.1 L703 | **CORE-THEORY** | Theoretical constraint from Zhang et al. Exceeding invalidates $f_\lambda(\theta)$ smoothness. Design-time constraint, not enforced in code. |
| 5 | Perturbation distribution | `z ~ N(0, I)` (isotropic Gaussian) | Sec 2 L40; Code L444 | **CORE-THEORY** | Required for `E[z_i * z_j] = delta_ij` in unbiasedness proof and `E[z^T H z] = Tr(H)` for implicit regularization. Rademacher preserves unbiasedness but loses trace interpretation. |
| 6 | Two-point symmetric estimator | `(f(theta+delta) - f(theta-delta)) / (2*lambda)` | Sec 2 L41 | **CORE-THEORY** | Halves variance vs. one-point; ensures odd-order Taylor terms cancel. Central to the $f_\lambda$ surrogate derivation. |
| 7 | Layer-adaptive scaling (LOREN) | `eps_l = eps_base / sqrt(l+1)` | Sec 7 L143-144; Code L445 | **CORE-DESIGN** | Trust region tightens with depth. Prevents deep-layer logit divergence (Failure Mode D). Without this, uniform eps is either too small for early layers or too large for deep layers. |

### 1.2 Perturbation Structure

| # | Parameter | Value | Source | Classification | Role & Impact |
|---|-----------|-------|--------|----------------|---------------|
| 8 | Three-phase decoupling (Tick / Tock-A / Tock-B) | Structural | Sec 4-5; Sec 10.3-10.5; Sec 10.6 L292-294 | **CORE-DESIGN** | Forced by: (a) dimensionality gap — m (~8K) vs A,B (~131K each) would drown magnitude signal; (b) bilinear cross-term $Z_AZ_B$ shifts midpoint by $\mathcal{O}(\lambda^2)$; (c) DoRA negative correlation ($-0.31$). |
| 9 | Tick perturbs m only | Structural | Sec 10.3 L248-261 | **CORE-DESIGN** | Isolates 8K-dimensional magnitude subspace for high-SNR estimation. |
| 10 | Tock-A perturbs A only, B frozen | Structural | Sec 10.4 L262-276 | **CORE-DESIGN** | Eliminates bilinear cross-term. Midpoint = AB exactly. |
| 11 | Tock-B perturbs B only, A frozen (at updated value) | Structural | Sec 10.5 L277-290 | **CORE-DESIGN** | Same cross-term elimination. Uses $A_{t+1}$ so B sees freshest A. |
| 12 | Phase ordering: Tick -> Tock-A -> Tock-B | Structural | Sec 10.3-10.5 | **HEURISTIC** | No proof this ordering is optimal. Tock-B uses updated A from Tock-A (sequential dependency), but Tick could run after Tock phases without violating any proof. |
| 13 | Scheduling variant: alternate Tock-A/Tock-B | Optional | Sec 15 L781 | **HEURISTIC** | Drops per-layer scoring cost by one-third. Staggers A/B updates. Compute/convergence tradeoff. |

---

## 2. Adapter Initialization (PiSSA)

| # | Parameter | Value | Source | Classification | Role & Impact |
|---|-----------|-------|--------|----------------|---------------|
| 14 | PiSSA SVD decomposition | $W_0 = U S V^T$ | Sec 3 L58; Sec 10.1 L226-227 | **CORE-THEORY** | Constrains optimization to rank-$r$ subspace. $d_{eff} = 262$K at r=16 vs $d_{raw} \approx 6.7 \times 10^7$. Directly enters convergence bound. |
| 15 | Symmetric $\sqrt{S}$ splitting | $A = U[:,:r]\sqrt{S}$, $B = \sqrt{S}V^T[:r,:]$ | Sec 10.1 L229; Code L667-669 | **CORE-DESIGN** | Balanced gradient norms between Tock-A and Tock-B. Asymmetric splitting creates gradient imbalance. |
| 16 | Subspace rank $r$ (initial) | 16 or 32 | Sec 13.2 L712; Code L329 | **TUNABLE-SEED** | Initial per-layer rank, dynamically reallocated. Higher = more capacity but larger $d_{eff}$, slower convergence, more memory. |
| 17 | Fast SVD (Halko et al.) | `niter=2` | Sec 10.1 L227; Code L665 | **ENGINEERING** | Randomized SVD. niter=2 sufficient for top-$r$ components. Full SVD exact but slower. |
| 18 | QPiSSA: quantize $W^{res}$, not $W_0$ | NF4 | Sec 10.1 L230; Sec 12 L683, L691 | **CORE-DESIGN** | Residual has narrower distribution → NF4 fits ~20% better. |
| 19 | Residual quantization format | NF4 | Sec 10.1 L225; Sec 12 L683 | **ENGINEERING** | 4-bit quantization. Alternatives: GPTQ, AWQ, FP8. NF4 chosen for ecosystem compatibility. |

---

## 3. DoRA Decomposition

| # | Parameter | Value | Source | Classification | Role & Impact |
|---|-----------|-------|--------|----------------|---------------|
| 20 | DoRA formula | $W' = m \cdot \frac{W^{res}+AB}{\|W^{res}+AB\|_c}$ | Sec 4 L68 | **CORE-THEORY** | Weight decomposition into magnitude and direction. Enables Tick/Tock decoupling. |
| 21 | Column-wise norms for magnitude | $m = \|W_0\|_c$ | Sec 10.1 L231; Code L674 | **IMPLICIT** | Must match vLLM's native DoRA kernel convention. Row-wise would break compatibility. |
| 22 | Negative correlation ($-0.31$) | Empirical | Sec 4 L78 | **IMPLICIT** | From DoRA paper. Supplementary justification for Tick/Tock. May not hold for all architectures. Decoupling is independently justified by SNR gap. |

---

## 4. Trajectory Locking & Contrastive Optimization

### 4.1 Trajectory Locking

| # | Parameter | Value | Source | Classification | Role & Impact |
|---|-----------|-------|--------|----------------|---------------|
| 23 | Trajectory locking protocol | Generate $\theta_0$, score $\theta^\pm$ | Sec 6 L101-105; Code L489-497 | **CORE-DESIGN** | Eliminates trajectory divergence (Failure Mode B) and reward vanishing (Failure Mode A). Makes loss a continuous function of weights. |

### 4.2 RLOO Contrastive Loss

| # | Parameter | Value | Source | Classification | Role & Impact |
|---|-----------|-------|--------|----------------|---------------|
| 24 | RLOO advantage formula | $A_i = R_i - \frac{1}{N-1}\sum_{j \neq i} R_j$ | Sec 6 L110-111; Code L514-515 | **CORE-DESIGN** | Unbiased, minimum-variance baseline. Self-centering ($\sum A_i = 0$). Zero lag. No tunable momentum parameter. Replaces EMA baseline. |
| 25 | RLOO advantages (runtime) | Computed per step | Code L511-515 | **AUTO-ADAPTED** | $A_w, A_l$ derived from batch rewards. Fully self-adapting. |
| 26 | Linear contrastive loss (not sigmoid) | $\mathcal{L} = \sum_i A_i \cdot \text{NLL}_i^{KL}$ | Sec 6 L113-114; Code L432-433 | **CORE-DESIGN** | Sigmoid DPO causes gradient underflow ($\sigma' \approx 0$) in ZO. Linear form avoids this but is unbounded, requiring safety mechanisms. |
| 27 | Per-token KL reward shaping | $\text{NLL}^{KL} = \frac{1}{T}\sum_t[-(1+\beta)\log\pi_\theta + \beta\log\pi_{ref}]$ | Sec 6 L117-120; Code L413-422 | **CORE-DESIGN** | Dense per-token drift penalty. Prevents mode collapse (Failure Mode E) without off-policy sequence-level KL error. Reference terms cancel in finite difference but stabilize absolute loss for health monitoring. |
| 28 | KL penalty $\beta$ | 0.1 (0.0 for short runs) | Sec 13.2 L721; Code L342 | **TUNABLE-SEED** | Strength of per-token anchor. 0.0 = no drift penalty (fast but risks collapse). Too high = over-regularized. |

### 4.3 Exploration

| # | Parameter | Value | Source | Classification | Role & Impact |
|---|-----------|-------|--------|----------------|---------------|
| 29 | Exploration candidates $N$ | 4 | Sec 13.2 L717; Code L336 | **HEURISTIC** | RLOO requires $N \geq 3$ for meaningful baselines. Higher = better advantage estimates but more generation cost. |
| 30 | Exploration temperature | 0.7 | Sec 13.2 L718; Code L337 | **HEURISTIC** | Diversity of candidates. Too low → identical outputs; too high → incoherent. |
| 31 | Reward threshold $R_{min}$ | 0.1 | Sec 13.2 L719; Code L338 | **HEURISTIC** | Minimum best-of-N reward to proceed. Too high → excessive rejection. |
| 32 | Contrastive gap $\delta$ | 0.1 | Sec 13.2 L720; Code L339 | **HEURISTIC** | Minimum reward gap between winner/loser. Too high → rejection; too low → noisy signal. |
| 33 | Reference logprob caching | Single batched `score()` call, batch=2 | Sec 10.2 L246; Code L518-523 | **ENGINEERING** | Cached once per step for per-token KL shaping. Uses unperturbed weights as reference. |

---

## 5. Tensor Adam Optimizer

| # | Parameter | Value | Source | Classification | Role & Impact |
|---|-----------|-------|--------|----------------|---------------|
| 34 | Global learning rate $\eta$ | $1 \times 10^{-5}$ | Sec 13.2 L714; Code L346 | **TUNABLE-SEED** | Scales all Adam updates. Primary tuning knob alongside $\epsilon_{base}$. Too high → divergence; too low → no progress. |
| 35 | Adam $\beta_1$ | 0.9 | Sec 13.2 L715; Code L347 | **HEURISTIC** | Gradient momentum. Standard default. |
| 36 | Adam $\beta_2$ | 0.999 | Sec 13.2 L715; Code L348 | **HEURISTIC** | Variance tracking. Standard default. |
| 37 | Adam $\delta$ (epsilon) | $10^{-8}$ | Sec 13.2 L716; Code L349 | **HEURISTIC** | Numerical stability floor. Standard default. |
| 38 | FP32 master weight accumulation | All masters + moments in FP32 | Sec 8 L164; Code L456 | **CORE-DESIGN** | Updates $\Delta\theta \approx 10^{-5}$ survive FP16 truncation (Failure Mode G). Non-negotiable. |
| 39 | First moment $m_1$ | EMA of gradient | Sec 8 L155; Code L469 | **AUTO-ADAPTED** | $\beta_1 m_{t-1} + (1-\beta_1)\hat{g}_t$. Fully automatic. |
| 40 | Second moment $v_2$ | EMA of squared gradient | Sec 8 L156; Code L470 | **AUTO-ADAPTED** | $\beta_2 v_{t-1} + (1-\beta_2)\hat{g}_t^2$. Fully automatic. |
| 41 | Bias correction | $1/(1-\beta^t)$ | Sec 8 L157; Code L474-475 | **AUTO-ADAPTED** | Corrects initialization bias. Determined by step count. |
| 42 | Per-phase separate moments | Dict keyed by `(layer_idx, group)` | Code L352 | **CORE-DESIGN** | Tick, Tock-A, Tock-B have independent Adam states per layer. Mixing moments across phases with different dimensionalities would corrupt variance estimates. |

---

## 6. Dynamic Rank Allocation

| # | Parameter | Value | Source | Classification | Role & Impact |
|---|-----------|-------|--------|----------------|---------------|
| 43 | Gradient magnitude accumulator | Per-layer running sum | Code L361, L614, L632, L650 | **AUTO-ADAPTED** | $\text{accum}[l] += \|\hat{g}\|_2$ after each phase. Feeds rank reallocation. Reset every interval. |
| 44 | Rank reallocation interval | 200 steps | Sec 13.2 L723; Code L358 | **HEURISTIC** | How often to redistribute ranks. Too frequent → instability from moment resets; too rare → slow adaptation. |
| 45 | Rank minimum $r_{min}$ | 4 | Sec 13.2 L724; Code L359 | **SAFETY** | Floor on per-layer rank. Prevents complete rank starvation (Failure Mode H). |
| 46 | Rank maximum $r_{max}$ | 64 | Sec 13.2 L724; Code L360 | **SAFETY** | Ceiling on per-layer rank. Prevents single layer from consuming all capacity. |
| 47 | Global rank budget conservation | `total_rank = sum(layer['rank'])` preserved | Code L557 | **IMPLICIT** | Total rank budget is constant. Reallocation only redistributes, never grows or shrinks total. |
| 48 | Adam moments reset on rank change | Moments deleted for affected layers | Code L577-578 | **CORE-DESIGN** | Prevents stale moment tensors of wrong shape. Causes temporary learning rate spike from bias correction restart. |
| 49 | Residual correction (greedy) | Difference dumped into highest-gradient layer | Code L567-570 | **HEURISTIC** | When rounding doesn't preserve total rank, excess goes to one layer. Simple but can hit $r_{max}$. |

---

## 7. Safety Mechanisms

| # | Parameter | Value | Source | Classification | Role & Impact |
|---|-----------|-------|--------|----------------|---------------|
| 50 | NLL spike detection threshold | $5\times$ EMA | Sec 13.2 L722; Code L538, L366 | **SAFETY** | Skip update if NLL exceeds threshold. Single anomalous batch could cause catastrophic weight update. |
| 51 | Health-check EMA ordering | Update after check, not before | Code L540-542 | **SAFETY** | EMA updated only on healthy steps. Updating before check allows sustained moderate spikes to inflate threshold — previous bug, now fixed. |
| 52 | Loss EMA momentum | 0.95 | Code L365 | **HEURISTIC** | Smoothing for health-check EMA. Higher → slower to adapt; lower → noisier threshold. |
| 53 | Reward rejection (threshold + gap) | $R_{min}=0.1$, $\delta=0.1$ | Code L505-508 | **SAFETY** | Skip step if no useful signal. Learning from uniformly low or indistinguishable rewards injects noise. |

---

## 8. Infrastructure & Engineering

| # | Parameter | Value | Source | Classification | Role & Impact |
|---|-----------|-------|--------|----------------|---------------|
| 54 | vLLM inference backend | vLLM with S-LoRA, PagedAttention | Sec 9 L193-203 | **ENGINEERING** | Requires `score()`, `add_lora()`, `update_lora_weights()` APIs. Alternatives: SGLang, TGI. |
| 55 | 2 static adapter slots | `adapter_pos`, `adapter_neg` | Sec 10.1 L232; Code L333, L370-374 | **ENGINEERING** | Registered once, overwritten in-place. Minimum 2 required for symmetric estimation. |
| 56 | In-place weight update API | `engine.update_lora_weights()` | Sec 11.1 L304; Code L393-394 | **ENGINEERING** | Direct FP16 tensor overwrite. Fallback: full `add_lora`/`remove_lora` cycle (~70ms vs ~6-12ms). |
| 57 | Contrastive scoring batch size | 4 (Y_w, Y_l × pos, neg) | Sec 10.3-10.5 | **ENGINEERING** | Batches all scoring into one call. Could be 2 sequential calls. |
| 58 | GPU memory utilization | 0.9 | Sec 12 L689 | **ENGINEERING** | vLLM `--gpu-memory-utilization` flag. Range: 0.85-0.95. |
| 59 | CPU controller / GPU inference split | Structural | Sec 9 L174-176 | **CORE-DESIGN** | Controller logic on CPU, all forward passes on GPU. Enables "inference server = training server." |
| 60 | PCIe / IPC weight transfer | ~1-2ms per update | Sec 11.1 L307; Sec 9 L191 | **ENGINEERING** | Transfer FP16 adapter tensors CPU → GPU. NVLink for multi-GPU. |

---

## 9. Implicit Assumptions

| # | Assumption | Source | Impact if Violated |
|---|-----------|--------|-------------------|
| 61 | Gaussian perturbation distribution | Sec 2 L40; Code L444 | Rademacher ±1 preserves unbiasedness but loses $\text{Tr}(\nabla^2 f)$ interpretation. Uniform breaks proof. |
| 62 | PiSSA-favorable PL constant $\mu$ | Sec 3 L63 | If fine-tuning moves far from pretrained initialization, $\mu$ degrades and convergence slows. |
| 63 | DoRA negative correlation $-0.31$ | Sec 4 L78 | Empirical from DoRA paper on specific models. Tick/Tock independently justified by SNR gap. |
| 64 | Reward function range $[0, 1]$ | Code L318 | RLOO advantages scale with reward range. Values outside [0,1] work but may need $\eta$ adjustment. |
| 65 | All target layers are dense linear projections | Sec 17.2 L827 | MoE routing, 1D convolutions, specialized layernorms need custom adapters. |
| 66 | Single GPU (H100 80GB) | Header; Sec 12 | Memory budget assumes single device. Multi-GPU changes communication but not core math. |
| 67 | Reference logprobs from unperturbed weights | Code L490-491, L519-522 | KL shaping uses $\theta_0$ as reference. By design, ref = current unperturbed. |
| 68 | Tock-B uses updated $A_{t+1}$ | Sec 10.5 L279 | Sequential dependency. Tock-B sees a slightly different landscape than Tock-A. |
| 69 | Global rank budget conservation | Code L557 | If total rank changes, memory budget and convergence properties change. |
| 70 | Adam moments reset on rank change | Code L577-578 | Correct (prevents shape mismatch) but causes temporary bias correction spike. |

---

## Summary Counts

| Classification | Count | Items |
|:---------------|:------|:------|
| CORE-THEORY | 6 | #1-6 |
| CORE-DESIGN | 14 | #7-11, #15, #18, #23-24, #26-27, #38, #42, #48, #59 |
| TUNABLE-SEED | 4 | #1, #16, #28, #34 |
| AUTO-ADAPTED | 6 | #25, #39-41, #43 |
| HEURISTIC | 12 | #12-13, #29-32, #35-37, #44, #49, #52 |
| SAFETY | 6 | #3, #45-46, #50-51, #53 |
| ENGINEERING | 8 | #17, #19, #33, #54-58, #60 |
| IMPLICIT | 10 | #21-22, #47, #61-70 |
| **Total** | **70** | |

*Note: Some items carry dual classifications (e.g., #1 is TUNABLE-SEED but also has CORE-THEORY constraints via #4; #2 is CORE-THEORY with a SAFETY sub-component #3). The primary classification reflects the dominant role.*

---

## Known Issues

### Resolved (from previous audit)

| Issue | Resolution |
|:------|:-----------|
| Dead `grad_magnitude_accum` — never populated | Gradient norms accumulated after each phase (Code L614, L632, L650) |
| Single-layer implementation | `self.layers` list with per-layer loop in `step()` (Code L595-653) |
| Health-check EMA ordering bug | EMA updated only after passing threshold check (Code L540-542) |
| EMA baseline lag/noise | Replaced with RLOO advantages (Code L510-515) |
| Ring buffer latency (~70ms) | In-place weight updates via static adapter slots (~6-12ms) |

### Open

| # | Issue | Severity | Details |
|:--|:------|:---------|:--------|
| O1 | **No warm-start after rank reallocation** | Medium | `_reinit_layer_adapter()` (Code L575) is called but not defined. If it re-initializes from scratch rather than truncating/padding existing adapter, layer progress is lost. Adam moments are correctly reset. |
| O2 | **DoRA negative correlation unverified broadly** | Low | The $-0.31$ value (Sec 4 L78) is empirical from specific models. Tick/Tock is independently justified by SNR gap, so this is supplementary. |
| O3 | **RLOO $N=2$ degeneracy** | Low | At $N=2$, advantage = $(R_w - R_l)/2$ for both — equivalent to simple gap. Default $N=4$ avoids this, but code doesn't enforce $N \geq 3$. Documented in Sec 16.1 L796. |
| O4 | **Reference logprob staleness within a step** | Low | Ref logprobs cached once per step (Code L519-522), used for all 3 phases across all layers. After updates within the step, "reference" no longer matches current state. Drift is negligible ($\Delta\theta \approx 10^{-5}$). |
| O5 | **Full adapter rebuild per write** | Low | `_build_adapter_from_layers` rebuilds all layers even when only one parameter group of one layer changed. Correct but wasteful if `update_lora_weights` supports partial updates. |
| O6 | **Quantization drift in DoRA column-norm** | Low | NF4 quantization of $W^{res}$ introduces fixed error in $\|W^{res}+AB\|_c$. Documented in Sec 12 L693. Mitigated by conservative $\eta$ and Adam's per-coordinate scaling. |
| O7 | **Greedy rank residual correction** | Low | When rounding doesn't preserve total rank, difference goes entirely to highest-gradient layer (Code L567-570). Can hit $r_{max}$ while deficit remains. Proportional correction would be more robust. |

---

## Delta from Previous Audit

| Change | Old | New |
|:-------|:----|:----|
| Total items | 78 | 70 |
| Removed | EMA baseline (`baseline`, `baseline_momentum`), ring buffer (`ring_size`, `ring_idx`) | Replaced by RLOO (no tunable params) and static adapter slots |
| Added | — | RLOO advantage (#24-25), per-token KL (#27-28), layer-adaptive scaling (#7), in-place weight protocol (#55-56), functional gradient accum (#43) |
| Fixed issues | 7 open | 5 resolved, 7 open |
| Line numbers | Stale (pre-fix doc) | Updated to current 855-line document |
