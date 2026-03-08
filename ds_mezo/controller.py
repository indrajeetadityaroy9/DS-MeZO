"""DS-MeZO Controller: zeroth-order optimization for LLM fine-tuning.

Algorithm-only implementation — no vLLM imports. All inference calls go
through the backend object passed at init.

Components:
- AGZO activation-guided subspace perturbation
- Momentum-aligned sensitivity masking (applied at update step)
- SPSA gradient estimation with adaptive epsilon and DD clipping
- ZO-Muon spectral optimizer (Newton-Schulz orthogonalization)
- RLOO contrastive selection with gradient regularization
- Entropy-guided temperature annealing
- KL divergence constraint

Reference: DS_MeZO_Combined.md §8 with Bug 1 (GR cancellation) and Bug 2 (masking timing) fixes.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from safetensors.torch import load_file
from ds_mezo.kernels import zo_muon_update, fused_perturb_dual
from ds_mezo.model_config import LayerSpec


def _default_score_fn(text: str) -> float:
    """Default reward: length of generated text."""
    return float(len(text))


@dataclass
class DSMeZOConfig:
    """Configuration for DS-MeZO controller."""
    adapter_path: str = ""
    rank: int = 16
    total_steps: int = 1000
    eta_max: float = 1e-4
    momentum: float = 0.9
    num_candidates: int = 4
    lambda_gr: float = 0.01
    delta_kl: float = 0.01
    r_calib: int = 8
    T_max: float = 1.0
    T_min: float = 0.3
    drift_threshold: float = 0.95
    eps_floor: float = 0.1
    mask_warmup: int = 10
    score_fn: Callable[[str], float] = field(default=_default_score_fn, repr=False)
    mode: str = "rl"

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
        config: DSMeZOConfig | dict[str, Any],
    ) -> None:
        self.backend = backend

        if isinstance(config, dict):
            cfg = DSMeZOConfig.from_dict(config)
        else:
            cfg = config

        self.mode = cfg.mode
        self.score_fn = cfg.score_fn
        self.step_count = 0
        self.total_steps = cfg.total_steps

        # Deterministic RNG for perturbations
        self.rng = torch.Generator(device="cuda")
        self.rng.manual_seed(42)

        # Load pre-computed PiSSA adapter into FP32 master weights on GPU
        adapter_tensors = load_file(
            os.path.join(cfg.adapter_path, "adapter_model.safetensors"),
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

        self.num_layers = len(self.layers)
        rank = cfg.rank

        # Initial adapter sync
        self._sync_adapters({}, {})

        # Exploration — entropy-guided temperature annealing (§7.3)
        self.num_candidates = cfg.num_candidates
        self.T_max = cfg.T_max
        self.T_min = cfg.T_min
        self.explore_temperature = self.T_max
        self.initial_entropy: float = 0.0
        self._last_reward_range: float = 0.0

        # Gradient regularization (§4.3a)
        self.lambda_gr = cfg.lambda_gr
        # KL constraint (§4.3b)
        self.delta_kl = cfg.delta_kl

        # Perturbation — adaptive epsilon (§2.1)
        # RL: scaled by rank for AGZO subspace; SFT: fixed 1e-3 for full-rank
        self.eps_base = 1e-3 if self.mode == "sft" else 1e-3 / math.sqrt(rank)
        self.eps = self.eps_base
        self.eps_floor = cfg.eps_floor
        self.initial_loss_ema: float = 0.0
        self.loss_ema: float = 0.0

        # Optimizer: ZO-Muon (§5)
        self.eta_max = cfg.eta_max
        self.eta_min = self.eta_max / 100
        self.eta = self.eta_max
        self.momentum = cfg.momentum

        # Pre-allocate momentum and scratch buffers for all layers
        self.momentum_buffers: dict[tuple[tuple[int, str], str], torch.Tensor] = {}
        self.scratch_buffers: dict[tuple[tuple[int, str], str], torch.Tensor] = {}
        for layer in self.layers:
            key_A = (layer.key, "A")
            key_B = (layer.key, "B")
            self.momentum_buffers[key_A] = torch.zeros_like(layer.A)
            self.momentum_buffers[key_B] = torch.zeros_like(layer.B)
            self.scratch_buffers[key_A] = torch.zeros_like(layer.A)
            self.scratch_buffers[key_B] = torch.zeros_like(layer.B)

        # Masking warmup — full perturbation for first N steps (§3.3)
        self.mask_warmup = cfg.mask_warmup

        # Activation subspace tracking — per-step power iteration (§3.5)
        self.r_calib = cfg.r_calib
        self.power_iter_steps = 3
        self.drift_threshold = cfg.drift_threshold
        self.activation_bases: dict[tuple[int, str], torch.Tensor] = {}

        # Directional derivative clipping (§2.1)
        self.dd_ema: float = 0.0

    # -- Activation Subspace Tracking (§3.5) --------------------------------

    def _calibrate_activation_bases_full(self, input_data: list[str]) -> None:
        """Full SVD calibration — used at init and on drift detection."""
        activations = self.backend.extract_activations(input_data)
        for layer in self.layers:
            H = activations[layer.key]
            r_calib = min(self.r_calib, H.shape[1])
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
        V_l = self.activation_bases[layer.key].to(device=B.device)

        z_coeff_B = torch.randn(
            B.shape[0], V_l.shape[1], device=B.device, generator=self.rng,
        )
        z_B = z_coeff_B @ V_l.T

        BV = B @ V_l
        Q, _ = torch.linalg.qr(BV)
        z_coeff_A = torch.randn(
            A.shape[0], Q.shape[1], device=A.device, generator=self.rng,
        )
        z_A = z_coeff_A @ Q.T

        return z_A * self.eps, z_B * self.eps

    # -- Adapter Sync -------------------------------------------------------

    def _sync_adapters(
        self,
        pos_overrides: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]],
        neg_overrides: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Delegate adapter serialization to backend."""
        self.backend.sync_adapters(pos_overrides, neg_overrides, self.layers)

    # -- Scoring (§4.3) -----------------------------------------------------

    def _get_prompt_logprobs(
        self, token_sequences: list[list[int]], lora_request: Any,
    ) -> list[list[float]]:
        """Score token sequences under a LoRA adapter."""
        return self.backend.score(token_sequences, lora_request)

    def _score_contrastive(
        self,
        trajectories: tuple[list[int], list[int]],
        advantages: tuple[float, float],
        prompt_len: int,
    ) -> tuple[float, float]:
        """Advantage-weighted NLL with asymmetric gradient regularization.

        When RLOO advantages are both zero (all candidates scored equally),
        falls back to NLL on the best candidate. This ensures gradient signal
        even when the reward function provides no contrast — making the
        mechanism model-agnostic regardless of task difficulty.
        """
        winner_tokens, loser_tokens = trajectories
        adv_w, adv_l = advantages

        lp_pos = self._get_prompt_logprobs(
            [winner_tokens, loser_tokens], self.backend.lora_pos
        )
        lp_pos = [lp[prompt_len:] for lp in lp_pos]
        lp_neg = self._get_prompt_logprobs(
            [winner_tokens, loser_tokens], self.backend.lora_neg
        )
        lp_neg = [lp[prompt_len:] for lp in lp_neg]

        def nll(logprobs: list[float]) -> float:
            n = max(len(logprobs), 1)
            return sum(float(-lp) for lp in logprobs) / n

        if adv_w == 0 and adv_l == 0:
            loss_pos = nll(lp_pos[0])
            loss_neg = nll(lp_neg[0])
        else:
            loss_pos = float(adv_w) * nll(lp_pos[0]) + float(adv_l) * nll(lp_pos[1])
            loss_neg = float(adv_w) * nll(lp_neg[0]) + float(adv_l) * nll(lp_neg[1])

        # Asymmetric GR — NLL divergence between perturbation directions
        total_tokens = 0
        nll_div = 0.0
        for lps_p, lps_n in zip(lp_pos, lp_neg):
            for p, n in zip(lps_p, lps_n):
                nll_div += (float(-p) - float(-n)) ** 2
                total_tokens += 1
        nll_div /= max(total_tokens, 1)
        loss_pos += self.lambda_gr * nll_div

        return loss_pos, loss_neg

    # -- SFT Loss (§2) ------------------------------------------------------

    def _compute_loss_sft(
        self, token_ids: list[int], prompt_len: int,
    ) -> tuple[float, float]:
        """NLL on target tokens under θ+ and θ-, with asymmetric GR."""
        lp_pos = self._get_prompt_logprobs([token_ids], self.backend.lora_pos)[0]
        lp_neg = self._get_prompt_logprobs([token_ids], self.backend.lora_neg)[0]

        lp_pos = lp_pos[prompt_len:]
        lp_neg = lp_neg[prompt_len:]

        def nll(logprobs: list[float]) -> float:
            n = max(len(logprobs), 1)
            return -sum(float(lp) for lp in logprobs) / n

        loss_pos = nll(lp_pos)
        loss_neg = nll(lp_neg)

        n = max(len(lp_pos), 1)
        nll_div = sum((float(-p) - float(-n_)) ** 2 for p, n_ in zip(lp_pos, lp_neg))
        loss_pos += self.lambda_gr * nll_div / n

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
    ) -> tuple[tuple[list[int], list[int]], tuple[float, float], int]:
        """Generate candidates, compute RLOO advantages.

        Returns (trajectories, advantages, prompt_len).
        """
        self._sync_adapters({}, {})
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

        # RLOO advantages (§4.2)
        all_rewards = [r for _, r in scored]
        total_reward = sum(all_rewards)
        N = len(scored)
        adv_w = best_reward - (total_reward - best_reward) / (N - 1)
        adv_l = worst_reward - (total_reward - worst_reward) / (N - 1)

        self._last_reward_range = best_reward - worst_reward

        winner_full = list(prompt_token_ids) + list(best_output.token_ids)
        loser_full = list(prompt_token_ids) + list(worst_output.token_ids)
        prompt_len = len(prompt_token_ids)

        return (winner_full, loser_full), (adv_w, adv_l), prompt_len

    # -- Adaptive Epsilon (§2.1) --------------------------------------------

    def _update_eps(self) -> None:
        """Scale epsilon proportionally to loss EMA."""
        ratio = max(self.loss_ema / self.initial_loss_ema, self.eps_floor) if self.initial_loss_ema > 0 else 1.0
        self.eps = self.eps_base * ratio

    # -- KL Constraint (§4.3b) ----------------------------------------------

    def _apply_kl_constraint(self, loss_pos: float, loss_neg: float) -> float:
        """Scale down LR if update exceeds KL budget."""
        kl_approx = abs(loss_pos - loss_neg)
        if kl_approx > self.delta_kl:
            return self.eta * self.delta_kl / kl_approx
        return self.eta

    # -- Cosine LR (§5.3) --------------------------------------------------

    def _update_lr(self) -> None:
        progress = self.step_count / self.total_steps
        self.eta = (
            self.eta_min
            + 0.5 * (self.eta_max - self.eta_min)
            * (1 + math.cos(math.pi * progress))
        )

    # -- Health Monitoring (§4.4) -------------------------------------------

    def _check_health(self, loss_pos: float, loss_neg: float) -> bool:
        """Spike detection — skip step if NLL > 5× EMA.

        Uses loss_neg for EMA tracking since it's the unbiased NLL
        (loss_pos includes the asymmetric GR penalty).
        """
        avg_nll = abs(loss_neg)
        if self.step_count == 1:
            self.loss_ema = avg_nll
            self.initial_loss_ema = avg_nll
            return True
        if self.loss_ema > 1e-8 and avg_nll > 5 * self.loss_ema:
            return False
        self.loss_ema = 0.95 * self.loss_ema + 0.05 * avg_nll
        return True

    # -- Directional Derivative Clipping (§2.1) -----------------------------

    def _clip_dd(self, dd: float) -> float:
        """Clip DD at 3× its running EMA."""
        if self.step_count == 1:
            self.dd_ema = abs(dd)
            return dd
        clip_val = 3 * self.dd_ema
        clipped = max(-clip_val, min(dd, clip_val))
        self.dd_ema = 0.95 * self.dd_ema + 0.05 * abs(dd)
        return clipped

    # -- Main Training Step (§7.1) ------------------------------------------

    def step(self, batch: list[str] | dict[str, Any]) -> None:
        self.step_count += 1

        if self.mode == "sft":
            return self._step_sft(batch)

        # RL mode: AGZO + ZO-Muon
        trajectories, advantages, prompt_len = self._explore(batch)

        # Activation tracking
        batch_activations = self.backend.extract_activations(batch)
        needs_recalib = self._update_activation_bases_power_iter(batch_activations)
        if needs_recalib:
            self._calibrate_activation_bases_full(batch)

        # AGZO perturbation
        perturbations: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in self.layers:
            perturbations[layer.key] = self._get_perturbation(layer)

        # Dual perturbation
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

        self._sync_adapters(pos_layers, neg_layers)
        loss_pos, loss_neg = self._score_contrastive(trajectories, advantages, prompt_len)

        # SPSA + ZO-Muon update
        if self._check_health(loss_pos, loss_neg):
            raw_dd = float(loss_pos - loss_neg) / (2.0 * self.eps)
            dd = self._clip_dd(raw_dd)

            effective_eta = self._apply_kl_constraint(loss_pos, loss_neg)
            saved_eta = self.eta
            self.eta = effective_eta

            for layer in self.layers:
                z_A, z_B = perturbations[layer.key]
                key_A = (layer.key, "A")
                key_B = (layer.key, "B")

                do_mask = self.step_count > self.mask_warmup
                zo_muon_update(layer.A, self.momentum_buffers[key_A],
                               z_A, self.scratch_buffers[key_A],
                               dd, self.momentum, self.eta, apply_mask=do_mask)
                zo_muon_update(layer.B, self.momentum_buffers[key_B],
                               z_B, self.scratch_buffers[key_B],
                               dd, self.momentum, self.eta, apply_mask=False)

            self.eta = saved_eta

        # Schedule updates
        self._update_lr()
        self._update_eps()
        self._update_temperature()

        # Entropy monitoring (§7.3)
        reward_range = self._last_reward_range
        if self.initial_entropy == 0 and reward_range > 0:
            self.initial_entropy = reward_range
        if (self.initial_entropy > 0 and reward_range > 0
                and reward_range < 0.5 * self.initial_entropy):
            self.explore_temperature = min(
                self.explore_temperature * 1.5, self.T_max
            )

    def _step_sft(self, batch: dict[str, Any]) -> None:
        """SFT step: vanilla MeZO (random perturbation + SGD).

        AGZO subspace and ZO-Muon orthogonalization are designed for RL's
        high-contrast reward signal. SFT's NLL gradient is lower-variance
        and works better with standard MeZO (Malladi et al. 2023).
        """
        # Random full-rank perturbation (not AGZO subspace)
        perturbations: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in self.layers:
            z_A = torch.randn_like(layer.A, generator=self.rng) * self.eps
            z_B = torch.randn_like(layer.B, generator=self.rng) * self.eps
            perturbations[layer.key] = (z_A, z_B)

        # Dual perturbation
        pos_layers: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        neg_layers: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in self.layers:
            z_A, z_B = perturbations[layer.key]
            pos_layers[layer.key] = (layer.A + z_A, layer.B + z_B)
            neg_layers[layer.key] = (layer.A - z_A, layer.B - z_B)

        self._sync_adapters(pos_layers, neg_layers)
        loss_pos, loss_neg = self._compute_loss_sft(
            batch["token_ids"], batch["prompt_len"]
        )

        # Plain SGD update (no ZO-Muon orthogonalization)
        if self._check_health(loss_pos, loss_neg):
            dd = float(loss_pos - loss_neg) / (2.0 * self.eps)

            for layer in self.layers:
                z_A, z_B = perturbations[layer.key]
                layer.A -= self.eta * dd * z_A
                layer.B -= self.eta * dd * z_B

        self._update_lr()

    # -- Main Training Loop -------------------------------------------------

    def train(self, data: list[str] | list[dict[str, Any]]) -> None:
        """Train for total_steps, cycling through data.

        data: list of strings (RL mode) or list of dicts with
              {token_ids, prompt_len, prompt_text} (SFT mode).
        """
        if self.mode == "rl":
            self._calibrate_activation_bases_full([data[0]])

        for step_idx in range(self.total_steps):
            sample = data[step_idx % len(data)]
            batch = sample if self.mode == "sft" else [sample]
            self.step(batch)

            if (step_idx + 1) % 10 == 0:
                print(
                    f"step={step_idx + 1}/{self.total_steps} "
                    f"lr={self.eta:.2e} eps={self.eps:.2e} "
                    f"temp={self.explore_temperature:.3f} "
                    f"loss_ema={self.loss_ema:.4f}"
                )

            if (step_idx + 1) % 100 == 0:
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
            "loss_ema": self.loss_ema,
            "initial_loss_ema": self.initial_loss_ema,
            "dd_ema": self.dd_ema,
            "eps": self.eps,
            "eta": self.eta,
            "explore_temperature": self.explore_temperature,
            "initial_entropy": self.initial_entropy,
        }
        path = os.path.join(self.backend.checkpoint_dir, f"step_{step}.pt")
        torch.save(state, path)
        print(f"Checkpoint saved: {path}")
