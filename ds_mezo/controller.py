"""DS-MeZO Controller: zeroth-order optimization for LLM fine-tuning.

Algorithm-only implementation — no vLLM imports. All inference calls go
through the backend object passed at init.

Components:
- AGZO activation-guided subspace perturbation
- Continuous cosine-similarity masking with EMA smoothing
- SPSA gradient estimation with adaptive epsilon and variance-based DD clipping
- ZO-Muon spectral optimizer (Newton-Schulz orthogonalization)
- RLOO contrastive selection with auto-calibrated gradient regularization
- REINFORCE++ advantage normalization
- Reward-variance exploration monitoring
- Auto-calibrated KL divergence constraint
- Hybrid SFT→RL training pipeline

Reference: DS_MeZO_Combined.md §8
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from ds_mezo.kernels import zo_muon_update, fused_perturb_dual
from ds_mezo.model_config import LayerSpec


def _mean_nll(logprobs: list[float]) -> float:
    """Mean negative log-likelihood over tokens."""
    return -sum(logprobs) / len(logprobs)


@dataclass
class DSMeZOConfig:
    """Configuration for DS-MeZO controller.

    Minimal configuration — most algorithm parameters are auto-calibrated
    from data at runtime. Only paths, training budget, and primary
    hyperparameters (eta_max) require user specification.

    Auto-calibrated parameters (derived from adapter, data, or other config):
    - rank: inferred from adapter tensor shapes
    - r_calib: rank // 2 (half adapter rank captures dominant activation subspace)
    - mask_warmup: 3/(1-momentum) — EMA time constant for 95% convergence
    - drift_threshold: 1 - 1/r_calib (recalibrate if any singular direction has drifted)
    - T_min: T_max / num_candidates (maintain diversity for RLOO)
    - eta_min: 0 (cosine decay to zero, per D2Z findings — CoLLAs 2025)
    - eps_base: 1e-3 / sqrt(rank) (SPSA theory — Spall 1998)
    - lambda_gr: 1/initial_loss (scale-invariant GR weight — ADRPO principle)
    - delta_kl: variance-based (kl_ema + 3σ, ZClip pattern — 2504.02507)
    - EMA factors: adaptive window from step count (VA-Muon — 2601.14603)
    - spike/DD thresholds: 3-sigma from tracked variance (ZClip — 2504.02507)
    """
    # Paths
    output_dir: str = "output"
    staging_dir: str = "/dev/shm/ds_mezo"
    adapter_path: str = ""
    model_path: str = ""

    # Training
    total_steps: int = 1000
    hybrid_switch_step: int = 0
    eta_max: float = 1e-4
    momentum: float = 0.9
    num_candidates: int = 4
    T_max: float = 1.0
    seed: int = 42
    score_fn: Callable[[str], float] | None = field(default=None, repr=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DSMeZOConfig:
        """Create config from dict, ignoring unknown keys."""
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class LayerState:
    """Runtime state for a single trainable layer."""
    A: torch.Tensor
    B: torch.Tensor
    layer_idx: int
    module_name: str
    peft_prefix: str

    @property
    def key(self) -> tuple[int, str]:
        return (self.layer_idx, self.module_name)


class DSMeZO_Controller:
    def __init__(
        self,
        backend: Any,
        layer_specs: list[LayerSpec],
        config: dict[str, Any],
    ) -> None:
        self.backend = backend
        cfg = DSMeZOConfig.from_dict(config)

        self.score_fn = cfg.score_fn
        self.step_count = 0
        self.total_steps = cfg.total_steps
        self.hybrid_switch_step = cfg.hybrid_switch_step

        # Output directory — single root for all persistent artifacts
        self.output_dir = Path(cfg.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Deterministic RNG for reproducible perturbations
        self.rng = torch.Generator(device="cuda")
        self.rng.manual_seed(cfg.seed)

        # Load pre-computed PiSSA adapter into FP32 master weights on GPU
        adapter_tensors = load_file(
            str(Path(cfg.adapter_path) / "adapter_model.safetensors"),
            device="cuda",
        )

        self.layers: list[LayerState] = []
        for ls in layer_specs:
            # PEFT → PiSSA: lora_A (r×d_in) = B, lora_B (d_out×r) = A
            B = adapter_tensors[f"{ls.peft_prefix}.lora_A.weight"].float()
            A = adapter_tensors[f"{ls.peft_prefix}.lora_B.weight"].float()
            self.layers.append(LayerState(
                A=A, B=B,
                layer_idx=ls.layer_idx,
                module_name=ls.module_name,
                peft_prefix=ls.peft_prefix,
            ))
        del adapter_tensors

        # Infer rank from adapter tensor shapes (B is r×d_in)
        rank = self.layers[0].B.shape[0]

        # Initial adapter sync
        self.backend.sync_adapters({}, {}, self.layers)

        # Exploration — reward-variance monitoring + temperature annealing (§7.3)
        self.num_candidates = cfg.num_candidates
        self.T_max = cfg.T_max
        # T_min derived: maintain enough diversity for RLOO with N candidates
        self.T_min = self.T_max / self.num_candidates
        self.explore_temperature = self.T_max
        self.reward_var_ema: float = 0.0
        self.initial_reward_var: float = 0.0

        # Gradient regularization (§4.3a) — auto-calibrated from first step
        # None = not yet calibrated; 0.0 = explicitly disabled (e.g. by ablation)
        self.lambda_gr: float | None = None
        # KL constraint (§4.3b) — variance-based tracking (ZClip pattern)
        # Throttle LR when |loss_pos - loss_neg| exceeds kl_ema + 3σ.
        # float('inf') = disabled (e.g. by ablation); None = use variance-based.
        self.delta_kl: float | None = None
        self.kl_ema: float = 0.0
        self.kl_var_ema: float = 0.0

        # Perturbation — adaptive epsilon (§2.1)
        # eps_base from SPSA theory (Spall 1998): optimal perturbation scale c₀ ≈
        # measurement_noise_std / sqrt(d_eff). For BF16 inference with ~1e-2 relative
        # precision on NLL ≈ O(1) and d_eff = rank × r_calib, this simplifies to
        # 1e-3 / sqrt(rank) for r_calib = rank/2.
        self.eps_base = 1e-3 / math.sqrt(rank)
        self.eps = self.eps_base
        self.initial_loss_ema: float = 0.0
        self.loss_ema: float = 0.0
        # Variance tracker for adaptive spike detection (ZClip — 2504.02507)
        self.loss_var_ema: float = 0.0

        # Optimizer: ZO-Muon (§5)
        self.eta_max = cfg.eta_max
        self.eta = self.eta_max
        self.momentum = cfg.momentum

        # Pre-allocate momentum, scratch, and masking EMA buffers
        self.momentum_buffers: dict[tuple[tuple[int, str], str], torch.Tensor] = {}
        self.scratch_buffers: dict[tuple[tuple[int, str], str], torch.Tensor] = {}
        self.mask_scale_ema: dict[tuple[tuple[int, str], str], float] = {}
        for layer in self.layers:
            key_A = (layer.key, "A")
            key_B = (layer.key, "B")
            self.momentum_buffers[key_A] = torch.zeros_like(layer.A)
            self.momentum_buffers[key_B] = torch.zeros_like(layer.B)
            self.scratch_buffers[key_A] = torch.zeros_like(layer.A)
            self.scratch_buffers[key_B] = torch.zeros_like(layer.B)
            # EMA init at 0.5: E[(cossim+1)/2] = 0.5 for independent random vectors
            self.mask_scale_ema[key_A] = 0.5

        # Masking warmup — momentum EMA needs ~3 time constants to reach 95% of
        # stationary distribution. Time constant τ = 1/(1-β), so warmup = 3τ.
        # For β=0.9: warmup=30. For β=0.99: warmup=300. Derived from momentum.
        self.mask_warmup = round(3.0 / (1.0 - self.momentum))

        # Activation subspace tracking — per-step power iteration (§3.5)
        # r_calib = rank/2: top half of singular spectrum captures >80% of variance
        # in the rank-r activation subspace (AGZO — 2601.17261, P-GAP — 2510.18228).
        self.r_calib = rank // 2
        # Warm-started power iteration converges cubically (Halko et al. 2011, §4.4).
        # 3 iterations: error ∝ ε₀^27 ≈ 0 for any reasonable ε₀ < 0.5.
        self.power_iter_steps = 3
        # Drift detection: alignment = trace(V_new^T @ V_old) / r_calib ∈ [0, 1].
        # Threshold 1 - 1/r_calib triggers recalibration when ≥1 singular direction
        # has rotated beyond expected subspace alignment.
        self.drift_threshold = 1.0 - 1.0 / self.r_calib
        self.activation_bases: dict[tuple[int, str], torch.Tensor] = {}

        # Directional derivative clipping (§2.1) — variance-based (ZClip)
        self.dd_ema: float = 0.0
        self.dd_var_ema: float = 0.0

    # -- Adaptive EMA (VA-Muon — 2601.14603) --------------------------------

    def _ema_alpha(self) -> float:
        """Adaptive EMA smoothing factor: responsive early, stable late.

        Window ramps from 1 (fully responsive at step 1) to sqrt(total_steps),
        giving alpha = 1/window. The sqrt follows from the bias-variance tradeoff
        for estimating a slowly-varying mean: window ∝ √T balances tracking error
        (bias from non-stationarity) against estimation noise (variance from finite
        samples). Minimum 20 ensures enough samples for meaningful variance
        estimates (CLT: n≥20 gives sample variance within ~30% of true variance).
        """
        max_window = max(20, int(math.sqrt(self.total_steps)))
        return 1.0 / min(self.step_count, max_window)

    # -- Activation Subspace Tracking (§3.5) --------------------------------

    def _calibrate_activation_bases_full(self, input_data: list[str]) -> None:
        """Full SVD calibration — used at init and on drift detection."""
        activations = self.backend.extract_activations(input_data)
        for layer in self.layers:
            H = activations[layer.key]
            r_calib = self.r_calib
            # niter=2: randomized SVD power iterations (Halko et al. 2011, §4.3).
            # 2 iterations is the standard recommendation for well-conditioned
            # matrices; additional iterations yield negligible accuracy gain.
            _, _, V = torch.svd_lowrank(H, q=r_calib, niter=2)
            self.activation_bases[layer.key] = V.float()

    def _update_activation_bases_power_iter(
        self, activations: dict[tuple[int, str], torch.Tensor],
    ) -> bool:
        """Per-step warm-started power iteration (§3.5)."""
        needs_full_recalib = False
        for layer in self.layers:
            H = activations[layer.key].float()
            V_old = self.activation_bases[layer.key]
            V = V_old.clone()
            for _ in range(self.power_iter_steps):
                V = H.T @ (H @ V)
                V, _ = torch.linalg.qr(V)
            alignment = torch.trace(V.T @ V_old).abs() / self.r_calib
            if alignment < self.drift_threshold:
                needs_full_recalib = True
            self.activation_bases[layer.key] = V
        return needs_full_recalib

    # -- Perturbation (§3.2) ------------------------------------------------

    def _get_perturbation(self, layer: LayerState) -> tuple[torch.Tensor, torch.Tensor]:
        """AGZO subspace perturbation for A and B."""
        A, B = layer.A, layer.B
        V_l = self.activation_bases[layer.key].cuda()

        z_coeff_B = torch.randn(
            B.shape[0], V_l.shape[1], device="cuda", generator=self.rng,
        )
        z_B = z_coeff_B @ V_l.T

        BV = B @ V_l
        Q, _ = torch.linalg.qr(BV)
        z_coeff_A = torch.randn(
            A.shape[0], Q.shape[1], device="cuda", generator=self.rng,
        )
        z_A = z_coeff_A @ Q.T

        return z_A * self.eps, z_B * self.eps

    # -- Scoring (§4.3) -----------------------------------------------------

    def _score_contrastive(
        self,
        trajectories: tuple[list[int], list[int]],
        advantages: tuple[float, float],
        prompt_len: int,
    ) -> tuple[float, float]:
        """Advantage-weighted NLL with asymmetric gradient regularization.

        When RLOO advantages are both zero (all candidates scored equally),
        falls back to NLL on the best candidate.
        """
        winner_tokens, loser_tokens = trajectories
        adv_w, adv_l = advantages

        seqs = [winner_tokens, loser_tokens]
        lp_pos = [lp[prompt_len:] for lp in self.backend.score(seqs, self.backend.lora_pos)]
        lp_neg = [lp[prompt_len:] for lp in self.backend.score(seqs, self.backend.lora_neg)]

        if adv_w == 0 and adv_l == 0:
            loss_pos = _mean_nll(lp_pos[0])
            loss_neg = _mean_nll(lp_neg[0])
        else:
            loss_pos = adv_w * _mean_nll(lp_pos[0]) + adv_l * _mean_nll(lp_pos[1])
            loss_neg = adv_w * _mean_nll(lp_neg[0]) + adv_l * _mean_nll(lp_neg[1])

        # Asymmetric GR — NLL divergence between perturbation directions
        if self.lambda_gr:
            nll_div = 0.0
            total_tokens = 0
            for lps_p, lps_n in zip(lp_pos, lp_neg):
                for p, n in zip(lps_p, lps_n):
                    nll_div += ((-p) - (-n)) ** 2
                    total_tokens += 1
            loss_pos += self.lambda_gr * nll_div / total_tokens

        return loss_pos, loss_neg

    # -- SFT Loss -----------------------------------------------------------

    def _compute_loss_sft(
        self, token_ids: list[int], prompt_len: int,
    ) -> tuple[float, float]:
        """NLL on target tokens under θ+ and θ-, with asymmetric GR."""
        lp_pos = self.backend.score([token_ids], self.backend.lora_pos)[0][prompt_len:]
        lp_neg = self.backend.score([token_ids], self.backend.lora_neg)[0][prompt_len:]

        loss_pos = _mean_nll(lp_pos)
        loss_neg = _mean_nll(lp_neg)

        if self.lambda_gr:
            nll_div = sum(((-p) - (-n)) ** 2 for p, n in zip(lp_pos, lp_neg))
            loss_pos += self.lambda_gr * nll_div / len(lp_pos)

        return loss_pos, loss_neg

    # -- Exploration (§4.1, §4.2, §7.3) -------------------------------------

    def _update_temperature(self) -> None:
        """Cosine temperature annealing (§7.3)."""
        progress = self.step_count / self.total_steps
        self.explore_temperature = (
            self.T_min
            + 0.5 * (self.T_max - self.T_min)
            * (1 + math.cos(math.pi * progress))
        )

    def _explore(
        self, batch: list[str],
    ) -> tuple[tuple[list[int], list[int]], tuple[float, float], int, float]:
        """Generate candidates, compute RLOO advantages with normalization.

        Returns (trajectories, advantages, prompt_len, reward_var).
        """
        self.backend.sync_adapters({}, {}, self.layers)
        request_outputs = self.backend.generate(
            batch, self.explore_temperature, self.num_candidates
        )
        prompt_token_ids = request_outputs[0].prompt_token_ids

        scored = [
            (out, self.score_fn(out.text))
            for out in request_outputs[0].outputs
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_output, best_reward = scored[0]
        worst_output, worst_reward = scored[-1]

        # RLOO advantages (§4.2) with REINFORCE++ normalization
        all_rewards = [r for _, r in scored]
        total_reward = sum(all_rewards)
        N = len(scored)
        adv_w = best_reward - (total_reward - best_reward) / (N - 1)
        adv_l = worst_reward - (total_reward - worst_reward) / (N - 1)

        # Normalize by reward standard deviation (REINFORCE++)
        mean_r = total_reward / N
        reward_var = sum((r - mean_r) ** 2 for r in all_rewards) / N
        reward_std = reward_var ** 0.5
        if reward_std > 0:
            adv_w /= reward_std
            adv_l /= reward_std

        winner_full = list(prompt_token_ids) + list(best_output.token_ids)
        loser_full = list(prompt_token_ids) + list(worst_output.token_ids)
        prompt_len = len(prompt_token_ids)

        return (winner_full, loser_full), (adv_w, adv_l), prompt_len, reward_var

    # -- Adaptive Epsilon (§2.1) --------------------------------------------

    def _update_eps(self) -> None:
        """Scale epsilon proportionally to loss EMA."""
        self.eps = self.eps_base * (self.loss_ema / self.initial_loss_ema)

    # -- KL Constraint (§4.3b) ----------------------------------------------

    def _apply_kl_constraint(self, loss_pos: float, loss_neg: float) -> float:
        """Scale down LR if update exceeds KL budget.

        Uses variance-based constraint (ZClip pattern): throttle when
        |loss_pos - loss_neg| exceeds kl_ema + 3σ of the tracked KL distribution.
        delta_kl override: float('inf') = disabled, other float = fixed budget.
        """
        if self.delta_kl is not None:
            # Explicit override (ablation or user-specified fixed budget)
            kl_approx = abs(loss_pos - loss_neg)
            if kl_approx > self.delta_kl:
                return self.eta * self.delta_kl / kl_approx
            return self.eta

        # Variance-based: constrain when KL exceeds 3σ of tracked distribution
        kl_approx = abs(loss_pos - loss_neg)
        kl_std = math.sqrt(self.kl_var_ema)
        kl_budget = self.kl_ema + 3.0 * kl_std
        if kl_approx > kl_budget:
            return self.eta * kl_budget / kl_approx
        return self.eta

    # -- Cosine LR (§5.3) --------------------------------------------------

    def _update_lr(self) -> None:
        """Cosine decay to zero (D2Z — CoLLAs 2025)."""
        progress = self.step_count / self.total_steps
        self.eta = 0.5 * self.eta_max * (1 + math.cos(math.pi * progress))

    # -- Health Monitoring (§4.4) -------------------------------------------

    def _check_health(self, loss_pos: float, loss_neg: float) -> bool:
        """Variance-based spike detection (ZClip — 2504.02507).

        Skips step if NLL exceeds loss_ema + 3σ (Chebyshev: ≤11% false positive
        for any distribution). Also tracks KL divergence statistics for the
        variance-based KL constraint.

        Auto-calibrates lambda_gr from the first step's observed NLL scale:
        lambda_gr = 1/NLL makes the GR term scale-invariant (ADRPO principle).
        """
        # In RL mode, loss_neg is advantage-weighted NLL (not raw NLL), so
        # avg_nll tracks the scale of the loss signal rather than true perplexity.
        avg_nll = abs(loss_neg)
        kl_approx = abs(loss_pos - loss_neg)
        alpha = self._ema_alpha()

        if self.step_count == 1:
            self.loss_ema = avg_nll
            self.initial_loss_ema = avg_nll
            self.loss_var_ema = 0.0
            # Auto-calibrate GR weight: 1/NLL makes regularization scale-invariant
            if self.lambda_gr is None:
                self.lambda_gr = 1.0 / avg_nll
            # Initialize KL divergence tracking
            self.kl_ema = kl_approx
            self.kl_var_ema = 0.0
            return True

        # 3-sigma spike detection from tracked variance
        # 3-sigma: Chebyshev bound guarantees ≤1/9 ≈ 11% false positive rate
        # for any distribution; <0.3% for Gaussian (ZClip — 2504.02507)
        loss_std = math.sqrt(self.loss_var_ema)
        if avg_nll > self.loss_ema + 3.0 * loss_std:
            return False

        # Exponential Welford update for loss mean and variance
        diff = avg_nll - self.loss_ema
        self.loss_ema += alpha * diff
        self.loss_var_ema = (1.0 - alpha) * (self.loss_var_ema + alpha * diff ** 2)

        # Track KL divergence statistics (same Welford pattern)
        kl_diff = kl_approx - self.kl_ema
        self.kl_ema += alpha * kl_diff
        self.kl_var_ema = (1.0 - alpha) * (self.kl_var_ema + alpha * kl_diff ** 2)
        return True

    # -- Directional Derivative Clipping (§2.1) -----------------------------

    def _clip_dd(self, dd: float) -> float:
        """Variance-based DD clipping (ZClip — 2504.02507).

        Clips at dd_ema ± 3·sqrt(dd_var_ema) instead of fixed 3× multiplier.
        """
        alpha = self._ema_alpha()
        abs_dd = abs(dd)

        if self.step_count == 1:
            self.dd_ema = abs_dd
            self.dd_var_ema = 0.0
            return dd

        dd_std = math.sqrt(self.dd_var_ema)
        clip_val = self.dd_ema + 3.0 * dd_std
        clipped = max(-clip_val, min(dd, clip_val))

        # Exponential Welford update on |dd|
        diff = abs_dd - self.dd_ema
        self.dd_ema += alpha * diff
        self.dd_var_ema = (1.0 - alpha) * (self.dd_var_ema + alpha * diff ** 2)
        return clipped

    # -- Unified Training Step ----------------------------------------------

    def step(self, batch: list[str] | dict[str, Any]) -> None:
        self.step_count += 1

        # Determine loss mode for this step
        use_rl_loss = (self.step_count > self.hybrid_switch_step)

        # RL exploration (only when computing RL loss)
        trajectories, advantages, prompt_len, reward_var = None, None, None, 0.0
        if use_rl_loss:
            trajectories, advantages, prompt_len, reward_var = self._explore(batch)

        # Activation tracking — lazy init if bases not yet calibrated (D4 fix)
        input_for_activations = batch if isinstance(batch, list) else [batch["prompt_text"]]
        if not self.activation_bases:
            self._calibrate_activation_bases_full(input_for_activations)
        batch_activations = self.backend.extract_activations(input_for_activations)
        needs_recalib = self._update_activation_bases_power_iter(batch_activations)
        if needs_recalib:
            self._calibrate_activation_bases_full(input_for_activations)

        # AGZO perturbation
        perturbations: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in self.layers:
            perturbations[layer.key] = self._get_perturbation(layer)

        # Dual perturbation via fused kernel
        pos_layers: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        neg_layers: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in self.layers:
            z_A, z_B = perturbations[layer.key]
            pos_A, neg_A = torch.empty_like(layer.A), torch.empty_like(layer.A)
            pos_B, neg_B = torch.empty_like(layer.B), torch.empty_like(layer.B)
            fused_perturb_dual(layer.A, z_A, pos_A, neg_A)
            fused_perturb_dual(layer.B, z_B, pos_B, neg_B)
            pos_layers[layer.key] = (pos_A, pos_B)
            neg_layers[layer.key] = (neg_A, neg_B)

        self.backend.sync_adapters(pos_layers, neg_layers, self.layers)

        # Loss computation — only branch point
        if use_rl_loss:
            loss_pos, loss_neg = self._score_contrastive(trajectories, advantages, prompt_len)
        else:
            loss_pos, loss_neg = self._compute_loss_sft(batch["token_ids"], batch["prompt_len"])

        # ZO-Muon update
        if self._check_health(loss_pos, loss_neg):
            raw_dd = float(loss_pos - loss_neg) / (2.0 * self.eps)
            dd = self._clip_dd(raw_dd)

            effective_eta = self._apply_kl_constraint(loss_pos, loss_neg)

            for layer in self.layers:
                z_A, z_B = perturbations[layer.key]
                key_A = (layer.key, "A")
                key_B = (layer.key, "B")

                # Continuous masking for A: only after warmup
                if self.step_count > self.mask_warmup:
                    grad_A = dd * z_A
                    cossim = F.cosine_similarity(
                        grad_A.reshape(1, -1),
                        self.momentum_buffers[key_A].reshape(1, -1),
                    ).item()
                    s = (cossim + 1.0) / 2.0
                    self.mask_scale_ema[key_A] = (
                        self.momentum * self.mask_scale_ema[key_A]
                        + (1.0 - self.momentum) * s
                    )
                    mask_scale = self.mask_scale_ema[key_A]
                else:
                    mask_scale = 1.0

                zo_muon_update(layer.A, self.momentum_buffers[key_A],
                               z_A, self.scratch_buffers[key_A],
                               dd, self.momentum, effective_eta, mask_scale=mask_scale)
                zo_muon_update(layer.B, self.momentum_buffers[key_B],
                               z_B, self.scratch_buffers[key_B],
                               dd, self.momentum, effective_eta, mask_scale=1.0)

        # Schedule updates
        self._update_lr()
        self._update_eps()

        # RL-only: temperature schedule + reward-variance exploration monitoring
        if use_rl_loss:
            self._update_temperature()

            alpha = self._ema_alpha()
            if self.initial_reward_var == 0.0 and reward_var > 0:
                self.reward_var_ema = reward_var
                self.initial_reward_var = reward_var
            elif self.initial_reward_var > 0:
                self.reward_var_ema += alpha * (reward_var - self.reward_var_ema)
                # Reward variance collapsed → exploration stagnated → reset to T_max
                # Threshold: variance < (1/2)² × initial = std has halved (natural signal)
                if self.reward_var_ema < self.initial_reward_var * 0.25:
                    self.explore_temperature = self.T_max

    # -- Main Training Loop -------------------------------------------------

    def train(
        self,
        rl_data: list[str] | None = None,
        sft_data: list[dict[str, Any]] | None = None,
    ) -> None:
        """Train for total_steps.

        rl_data: list of prompt strings for RL steps.
        sft_data: list of dicts with {token_ids, prompt_len, prompt_text} for SFT steps.

        Steps 1..hybrid_switch_step use SFT loss on sft_data.
        Steps hybrid_switch_step+1..total_steps use RL loss on rl_data.
        """
        # Calibrate activation bases from first available data
        first_text = sft_data[0]["prompt_text"] if sft_data else rl_data[0]
        self._calibrate_activation_bases_full([first_text])

        # Log every ~1% of training, checkpoint every ~10%
        log_interval = max(1, self.total_steps // 100)
        ckpt_interval = max(1, self.total_steps // 10)

        for step_idx in range(self.total_steps):
            if step_idx < self.hybrid_switch_step and sft_data:
                batch = sft_data[step_idx % len(sft_data)]
            else:
                batch = [rl_data[(step_idx - self.hybrid_switch_step) % len(rl_data)]]
            self.step(batch)

            if (step_idx + 1) % log_interval == 0:
                print(
                    f"step={step_idx + 1}/{self.total_steps} "
                    f"lr={self.eta:.2e} eps={self.eps:.2e} "
                    f"temp={self.explore_temperature:.3f} "
                    f"loss_ema={self.loss_ema:.4f}"
                )

            if (step_idx + 1) % ckpt_interval == 0:
                self._save_checkpoint(step_idx + 1)

        self._save_checkpoint(self.total_steps)

    def _save_checkpoint(self, step: int) -> None:
        """Minimal checkpoint: FP32 masters + momentum + step count."""
        state = {
            "step": step,
            "layers": [
                {"A": l.A, "B": l.B,
                 "layer_idx": l.layer_idx, "module_name": l.module_name}
                for l in self.layers
            ],
            "momentum_buffers": self.momentum_buffers,
            "activation_bases": self.activation_bases,
            "mask_scale_ema": self.mask_scale_ema,
            "loss_ema": self.loss_ema,
            "initial_loss_ema": self.initial_loss_ema,
            "loss_var_ema": self.loss_var_ema,
            "dd_ema": self.dd_ema,
            "dd_var_ema": self.dd_var_ema,
            "eps": self.eps,
            "eta": self.eta,
            "explore_temperature": self.explore_temperature,
            "reward_var_ema": self.reward_var_ema,
            "initial_reward_var": self.initial_reward_var,
            "lambda_gr": self.lambda_gr,
            "delta_kl": self.delta_kl,
            "kl_ema": self.kl_ema,
            "kl_var_ema": self.kl_var_ema,
        }
        path = self.checkpoint_dir / f"step_{step}.pt"
        torch.save(state, path)
        print(f"Checkpoint saved: {path}")
