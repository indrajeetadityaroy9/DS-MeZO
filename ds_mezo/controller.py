"""DS-MeZO Controller: zeroth-order optimization for LLM fine-tuning.

Algorithm-only implementation — no vLLM imports. All inference calls go
through the backend object passed at init.

Components:
- AGZO activation-guided subspace perturbation
- SPSA gradient estimation with ZClip z-score clipping on directional derivative
- ZO-Muon spectral optimizer (Newton-Schulz orthogonalization)
- RLOO contrastive selection with REINFORCE++ normalization
- Cosine temperature annealing for exploration
- Hybrid SFT→RL training pipeline

Reference: DS_MeZO_Combined.md §8
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors.torch import load_file, save_file
from ds_mezo.backend import _save_peft_adapter, _write_adapter_config
from ds_mezo.kernels import zo_muon_update, fused_power_iter, fused_agzo_perturbation, fused_perturb_dual
from ds_mezo.model_config import LayerSpec

# Config defaults — matches YAML config surface.
# Auto-calibrated parameters (derived from adapter/data):
#   rank: inferred from adapter tensor shapes
#   r_calib: rank // 2 (dominant activation subspace)
#   T_min: T_max / num_candidates (maintain RLOO diversity)
#   eps: 1e-3 / sqrt(rank) (SPSA theory — Spall 1998)
#   EMA: adaptive window from step count (VA-Muon — 2601.14603)
#   DD clipping: 3-sigma from tracked variance (ZClip — 2504.02507)
_CONFIG_DEFAULTS: dict[str, Any] = {
    "output_dir": "output",
    "staging_dir": "/dev/shm/ds_mezo",
    "adapter_path": "",
    "model_path": "",
    "total_steps": 1000,
    "hybrid_switch_step": 0,
    "eta_max": 1e-4,
    "momentum": 0.9,
    "num_candidates": 4,
    "T_max": 1.0,
    "seed": 42,
    "score_fn": None,
}


def _mean_nll(logprobs: list[float]) -> float:
    """Mean negative log-likelihood over tokens."""
    return -sum(logprobs) / len(logprobs)


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

        # Merge caller config over defaults (unknown keys silently ignored)
        cfg = {k: config.get(k, v) for k, v in _CONFIG_DEFAULTS.items()}

        self.score_fn: Callable[[str], float] | None = cfg["score_fn"]
        self.step_count = 0
        self.total_steps: int = cfg["total_steps"]
        self.hybrid_switch_step: int = cfg["hybrid_switch_step"]

        # Output directory — single root for all persistent artifacts
        self.output_dir = Path(cfg["output_dir"])
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Deterministic RNG for reproducible perturbations
        self.rng = torch.Generator(device="cuda")
        self.rng.manual_seed(cfg["seed"])

        # Load pre-computed PiSSA adapter into FP32 master weights on GPU
        adapter_tensors = load_file(
            str(Path(cfg["adapter_path"]) / "adapter_model.safetensors"),
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

        # Exploration — cosine temperature annealing (§7.3)
        self.num_candidates: int = cfg["num_candidates"]
        self.T_max: float = cfg["T_max"]
        self.T_min = self.T_max / self.num_candidates
        self.explore_temperature = self.T_max

        # Fixed epsilon (FlatZero — 2506.05454: fixed ε preserves flat-minima regularization)
        # SPSA theory (Spall 1998): optimal perturbation scale c₀ ≈ 1e-3 / sqrt(rank)
        self.eps = 1e-3 / math.sqrt(rank)

        # Optimizer: ZO-Muon (§5)
        self.eta_max: float = cfg["eta_max"]
        self.eta = self.eta_max
        self.momentum: float = cfg["momentum"]

        # Pre-allocate momentum, scratch, and perturbation output buffers
        self.momentum_buffers: dict[tuple[tuple[int, str], str], torch.Tensor] = {}
        self.scratch_buffers: dict[tuple[tuple[int, str], str], torch.Tensor] = {}
        self.pos_buffers: dict[tuple[tuple[int, str], str], torch.Tensor] = {}
        self.neg_buffers: dict[tuple[tuple[int, str], str], torch.Tensor] = {}
        for layer in self.layers:
            key_A = (layer.key, "A")
            key_B = (layer.key, "B")
            self.momentum_buffers[key_A] = torch.zeros_like(layer.A)
            self.momentum_buffers[key_B] = torch.zeros_like(layer.B)
            self.scratch_buffers[key_A] = torch.zeros_like(layer.A)
            self.scratch_buffers[key_B] = torch.zeros_like(layer.B)
            self.pos_buffers[key_A] = torch.empty_like(layer.A)
            self.pos_buffers[key_B] = torch.empty_like(layer.B)
            self.neg_buffers[key_A] = torch.empty_like(layer.A)
            self.neg_buffers[key_B] = torch.empty_like(layer.B)

        # Activation subspace tracking — per-step power iteration (§3.5)
        # r_calib = rank/2: top half of singular spectrum captures >80% of variance
        self.r_calib = rank // 2
        # Warm-started power iteration converges cubically (Halko et al. 2011, §4.4).
        # 3 iterations: error ∝ ε₀^27 ≈ 0 for any reasonable ε₀ < 0.5.
        self.power_iter_steps = 3
        self.activation_bases: dict[tuple[int, str], torch.Tensor] = {}

        # ZClip on directional derivative (2504.02507)
        self.dd_ema: float = 0.0
        self.dd_var_ema: float = 0.0

    # -- Adaptive EMA (VA-Muon — 2601.14603) --------------------------------

    def _ema_alpha(self) -> float:
        """Adaptive EMA smoothing factor: responsive early, stable late.

        Window ramps from 1 (fully responsive at step 1) to sqrt(total_steps),
        giving alpha = 1/window. The sqrt follows from the bias-variance tradeoff
        for estimating a slowly-varying mean.
        """
        max_window = int(math.sqrt(self.total_steps))
        return 1.0 / min(self.step_count, max_window)

    # -- Activation Subspace Tracking (§3.5) --------------------------------

    def _calibrate_activation_bases_full(self, input_data: list[str]) -> None:
        """Full SVD calibration — used once at init."""
        activations = self.backend.extract_activations(input_data)
        for layer in self.layers:
            H = activations[layer.key].cuda()
            _, _, V = torch.svd_lowrank(H, q=self.r_calib, niter=2)
            self.activation_bases[layer.key] = V

    def _update_activation_bases(
        self, activations: dict[tuple[int, str], torch.Tensor],
    ) -> None:
        """Warm-started power iteration via fused Triton kernel.

        Replaces 3× (matmul + matmul + QR) = 9 kernel launches per layer
        with a single fused kernel launch per layer.
        """
        for layer in self.layers:
            H = activations[layer.key].cuda()
            V = self.activation_bases[layer.key]
            self.activation_bases[layer.key] = fused_power_iter(
                H, V, num_iters=self.power_iter_steps,
            )

    # -- Perturbation (§3.2) ------------------------------------------------

    def _get_perturbation(self, layer: LayerState) -> tuple[torch.Tensor, torch.Tensor]:
        """AGZO subspace perturbation for A and B via fused Triton kernel.

        B perturbation: project random coefficients into activation subspace V.
        A perturbation: project into B's column space via QR(B @ V).
        Single kernel: matmuls + MGS QR + scaling fused, eliminates cuSOLVER overhead.
        """
        A, B = layer.A, layer.B
        V_l = self.activation_bases[layer.key]

        z_coeff_B = torch.randn(
            B.shape[0], V_l.shape[1], device="cuda", generator=self.rng,
        )
        z_coeff_A = torch.randn(
            A.shape[0], V_l.shape[1], device="cuda", generator=self.rng,
        )

        return fused_agzo_perturbation(B, V_l, z_coeff_B, z_coeff_A, self.eps)

    # -- Scoring (§4.3) -----------------------------------------------------

    def _score_contrastive(
        self,
        trajectories: tuple[list[int], list[int]],
        advantages: tuple[float, float],
        prompt_len: int,
    ) -> tuple[float, float]:
        """Advantage-weighted NLL scoring.

        When RLOO advantages are both zero (all candidates scored equally),
        the weighted sum produces zero loss for both perturbations, giving
        dd = 0 and no parameter update — the correct behavior.
        """
        winner_tokens, loser_tokens = trajectories
        adv_w, adv_l = advantages

        seqs = [winner_tokens, loser_tokens]
        lp_pos = [lp[prompt_len:] for lp in self.backend.score(seqs, self.backend.lora_pos)]
        lp_neg = [lp[prompt_len:] for lp in self.backend.score(seqs, self.backend.lora_neg)]

        loss_pos = adv_w * _mean_nll(lp_pos[0]) + adv_l * _mean_nll(lp_pos[1])
        loss_neg = adv_w * _mean_nll(lp_neg[0]) + adv_l * _mean_nll(lp_neg[1])

        return loss_pos, loss_neg

    # -- SFT Loss -----------------------------------------------------------

    def _compute_loss_sft(
        self, token_ids: list[int], prompt_len: int,
    ) -> tuple[float, float]:
        """NLL on target tokens under θ+ and θ-."""
        lp_pos = self.backend.score([token_ids], self.backend.lora_pos)[0][prompt_len:]
        lp_neg = self.backend.score([token_ids], self.backend.lora_neg)[0][prompt_len:]

        return _mean_nll(lp_pos), _mean_nll(lp_neg)

    # -- Exploration (§4.1, §4.2) -------------------------------------------

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
        """Generate candidates, compute RLOO advantages with normalization.

        Returns (trajectories, advantages, prompt_len).
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
        # Epsilon-stabilized: when all rewards are equal, advantages are already
        # zero, so dividing by ~0 still gives ~0 without NaN.
        mean_r = total_reward / N
        reward_var = sum((r - mean_r) ** 2 for r in all_rewards) / N
        reward_std = reward_var ** 0.5 + 1e-8
        adv_w /= reward_std
        adv_l /= reward_std

        winner_full = list(prompt_token_ids) + list(best_output.token_ids)
        loser_full = list(prompt_token_ids) + list(worst_output.token_ids)
        prompt_len = len(prompt_token_ids)

        return (winner_full, loser_full), (adv_w, adv_l), prompt_len

    # -- ZClip (2504.02507) -------------------------------------------------

    def _zclip(self, dd: float) -> float:
        """ZClip: z-score clipping on directional derivative (2504.02507).

        Clips at dd_ema ± 3·σ (Chebyshev: ≤1/9 false positive for any distribution).
        """
        alpha = self._ema_alpha()
        abs_dd = abs(dd)

        if self.step_count == 1:
            self.dd_ema = abs_dd
            return dd

        dd_std = math.sqrt(self.dd_var_ema)
        clip_bound = self.dd_ema + 3.0 * dd_std
        clipped = max(-clip_bound, min(dd, clip_bound))

        diff = abs_dd - self.dd_ema
        self.dd_ema += alpha * diff
        self.dd_var_ema = (1.0 - alpha) * (self.dd_var_ema + alpha * diff ** 2)

        return clipped

    # -- Cosine LR (§5.3) --------------------------------------------------

    def _update_lr(self) -> None:
        """Cosine decay to zero (D2Z — CoLLAs 2025)."""
        progress = self.step_count / self.total_steps
        self.eta = 0.5 * self.eta_max * (1 + math.cos(math.pi * progress))

    # -- Unified Training Step ----------------------------------------------

    def step(self, batch: list[str] | dict[str, Any]) -> None:
        self.step_count += 1

        # Determine loss mode for this step
        use_rl_loss = (self.step_count > self.hybrid_switch_step)

        # RL exploration (only when computing RL loss)
        trajectories, advantages, prompt_len = None, None, None
        if use_rl_loss:
            trajectories, advantages, prompt_len = self._explore(batch)

        # Activation tracking (bases must be calibrated before first step)
        input_for_activations = batch if isinstance(batch, list) else [batch["prompt_text"]]
        batch_activations = self.backend.extract_activations(input_for_activations)
        self._update_activation_bases(batch_activations)

        # AGZO perturbation
        perturbations: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in self.layers:
            perturbations[layer.key] = self._get_perturbation(layer)

        # Dual perturbation: θ+ = base + z, θ- = base - z (fused kernel)
        pos_layers: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        neg_layers: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in self.layers:
            z_A, z_B = perturbations[layer.key]
            key_A = (layer.key, "A")
            key_B = (layer.key, "B")
            fused_perturb_dual(layer.A, z_A, self.pos_buffers[key_A], self.neg_buffers[key_A])
            fused_perturb_dual(layer.B, z_B, self.pos_buffers[key_B], self.neg_buffers[key_B])
            pos_layers[layer.key] = (self.pos_buffers[key_A], self.pos_buffers[key_B])
            neg_layers[layer.key] = (self.neg_buffers[key_A], self.neg_buffers[key_B])

        self.backend.sync_adapters(pos_layers, neg_layers, self.layers)

        # Loss computation
        if use_rl_loss:
            loss_pos, loss_neg = self._score_contrastive(trajectories, advantages, prompt_len)
        else:
            loss_pos, loss_neg = self._compute_loss_sft(batch["token_ids"], batch["prompt_len"])

        # ZO-Muon update with ZClip
        raw_dd = float(loss_pos - loss_neg) / (2.0 * self.eps)
        dd = self._zclip(raw_dd)
        effective_eta = self.eta

        for layer in self.layers:
            z_A, z_B = perturbations[layer.key]
            key_A = (layer.key, "A")
            key_B = (layer.key, "B")
            zo_muon_update(layer.A, self.momentum_buffers[key_A],
                           z_A, self.scratch_buffers[key_A],
                           dd, self.momentum, effective_eta)
            zo_muon_update(layer.B, self.momentum_buffers[key_B],
                           z_B, self.scratch_buffers[key_B],
                           dd, self.momentum, effective_eta)

        # Schedule updates
        self._update_lr()
        if use_rl_loss:
            self._update_temperature()

    # -- Main Training Loop -------------------------------------------------

    def train(
        self,
        rl_data: list[str],
        sft_data: list[dict[str, Any]] | None = None,
    ) -> None:
        """Train for total_steps.

        rl_data: list of prompt strings for RL steps (required).
        sft_data: list of dicts with {token_ids, prompt_len, prompt_text} for SFT steps.
            Required when hybrid_switch_step > 0.

        Steps 1..hybrid_switch_step use SFT loss on sft_data.
        Steps hybrid_switch_step+1..total_steps use RL loss on rl_data.

        Activation bases must be calibrated via _calibrate_activation_bases_full()
        before calling train().
        """
        # Log every ~1% of training, checkpoint every ~10%
        log_interval = max(1, self.total_steps // 100)
        ckpt_interval = max(1, self.total_steps // 10)

        for step_idx in range(self.step_count, self.total_steps):
            if step_idx < self.hybrid_switch_step:
                batch = sft_data[step_idx % len(sft_data)]
            else:
                batch = [rl_data[(step_idx - self.hybrid_switch_step) % len(rl_data)]]
            self.step(batch)

            if (step_idx + 1) % log_interval == 0:
                print(
                    f"step={step_idx + 1}/{self.total_steps} "
                    f"lr={self.eta:.2e} "
                    f"temp={self.explore_temperature:.3f} "
                    f"dd_ema={self.dd_ema:.4f}"
                )

            if (step_idx + 1) % ckpt_interval == 0:
                self._save_checkpoint(step_idx + 1)

        self._save_checkpoint(self.total_steps)

    def _save_checkpoint(self, step: int) -> None:
        """Save checkpoint as safetensors + JSON (no pickle)."""
        step_dir = self.checkpoint_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        rank = self.layers[0].B.shape[0]
        target_modules = sorted({l.module_name for l in self.layers})

        # PEFT-compatible adapter (BF16)
        A_list = [l.A for l in self.layers]
        B_list = [l.B for l in self.layers]
        _save_peft_adapter(A_list, B_list, step_dir, self.layers)
        _write_adapter_config(step_dir, rank, target_modules)

        # Optimizer tensors (FP32) — flatten tuple keys to dot-separated strings
        # safetensors handles GPU→disk transfer internally via Rust backend
        tensors: dict[str, torch.Tensor] = {}
        for layer in self.layers:
            idx, mod = layer.layer_idx, layer.module_name
            tensors[f"master.layer{idx}.{mod}.A"] = layer.A
            tensors[f"master.layer{idx}.{mod}.B"] = layer.B
            tensors[f"momentum.layer{idx}.{mod}.A"] = self.momentum_buffers[(layer.key, "A")]
            tensors[f"momentum.layer{idx}.{mod}.B"] = self.momentum_buffers[(layer.key, "B")]
            tensors[f"act_basis.layer{idx}.{mod}"] = self.activation_bases[layer.key]
        tensors["rng_state"] = self.rng.get_state()
        save_file(tensors, str(step_dir / "optimizer_state.safetensors"))

        # Scalar state as JSON
        training_state = {
            "step": step,
            "dd_ema": self.dd_ema,
            "dd_var_ema": self.dd_var_ema,
            "eta": self.eta,
            "explore_temperature": self.explore_temperature,
        }
        with open(step_dir / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)

        # Overwrite latest adapter at output_dir/adapter
        adapter_dir = self.output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        _save_peft_adapter(A_list, B_list, adapter_dir, self.layers)
        _write_adapter_config(adapter_dir, rank, target_modules)

        print(f"Checkpoint saved: {step_dir}")

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        """Restore state from a safetensors + JSON checkpoint."""
        with open(checkpoint_path / "training_state.json") as f:
            ts = json.load(f)

        self.step_count = ts["step"]
        self.dd_ema = ts["dd_ema"]
        self.dd_var_ema = ts["dd_var_ema"]
        self.eta = ts["eta"]
        self.explore_temperature = ts["explore_temperature"]

        # Load optimizer tensors
        opt = load_file(
            str(checkpoint_path / "optimizer_state.safetensors"), device="cuda",
        )

        for layer in self.layers:
            idx, mod = layer.layer_idx, layer.module_name
            layer.A.copy_(opt[f"master.layer{idx}.{mod}.A"])
            layer.B.copy_(opt[f"master.layer{idx}.{mod}.B"])
            self.momentum_buffers[(layer.key, "A")].copy_(opt[f"momentum.layer{idx}.{mod}.A"])
            self.momentum_buffers[(layer.key, "B")].copy_(opt[f"momentum.layer{idx}.{mod}.B"])
            self.activation_bases[layer.key] = opt[f"act_basis.layer{idx}.{mod}"]

        # Generator.set_state() requires CPU ByteTensor (PyTorch API constraint)
        self.rng.set_state(opt["rng_state"].cpu())

        self.backend.sync_adapters({}, {}, self.layers)
        print(f"Checkpoint loaded: {checkpoint_path}")
