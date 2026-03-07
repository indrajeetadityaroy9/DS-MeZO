# DS-MeZO Parameter Audit

Companion to `DS_MeZO_Combined.md` — consistency verification and open design considerations. For the authoritative parameter inventory and scaling guidelines, see §10 of the main doc.

---

## 1. Consistency Verification: Math ↔ Code

| Formula (Section) | Code Implementation | Match? |
|:-------------------|:--------------------|:-------|
| $\hat{g} = \frac{\Delta\mathcal{L}}{2\epsilon} \cdot z$ (§2) | `grad = (diff / (2.0 * self.eps)) * z` | Yes |
| $z_B = Z_{coeff} \cdot V_l^T$ (§3.2) | `z_B = z_coeff_B @ V_l.T` | Yes |
| $Q = \operatorname{orth}(B \cdot V_l)$ for A (§3.2) | `Q, _ = torch.linalg.qr(B @ V_l)` | Yes |
| Sparse mask $M = \mathbf{1}[|\theta| \leq h]$ (§3.3) | `mask = (A.abs() <= quantile(A.abs(), sparsity)).float()` | Yes |
| $v_t = \mu v_{t-1} + \hat{g}_t$ (§5.2) | `buf.mul_(momentum).add_(grad)` | Yes |
| $\theta_{t+1} = \theta_t - \eta_t v_t$ (§5.2) | `param.sub_(self.eta * buf)` | Yes |
| Cosine LR (§5.3) | `eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * t/T))` | Yes |
| RLOO advantage (§4.2) | `adv = reward - (total - reward) / (N - 1)` | Yes |
| KL-shaped NLL (§4.3) | `nll + beta * (ref - lp)` = $-(1+\beta)\log\pi + \beta\log\pi_{ref}$ | Yes |

---

## 2. Open Design Considerations

### 2.1 All-at-Once Variance

Joint perturbation of all layers shares a single scalar $(L^+ - L^-)$. With $d_{eff} \approx 1M$ (after AGZO + sparse), the ZO gradient variance scales as $\mathcal{O}(d_{eff} \cdot \sigma^2)$. If variance is too high in practice:

- **Mitigation 1:** K-sample SPSA averaging (K=4 → 16 prefills).
- **Mitigation 2:** Increase rank of activation subspace ($r_{calib} = 16$).

### 2.2 Activation Calibration Implementation

AGZO requires extracting per-layer activations from the model. The vLLM model runner hook is the mandatory implementation path: modify `model_runner` to capture activations during scoring prefills. Calibration data is a required constructor parameter and activation bases are refreshed every 100 steps.

### 2.3 `_explore` Single-Prompt Assumption

The code assumes `request_outputs[0]` (single prompt per batch). When multi-prompt support is added, iterate over all request outputs and compute per-prompt RLOO advantages.
