"""DS-MeZO Controller: BSCO — Bayesian Subspace Contrastive Optimization.

Diagonal Kalman filter replaces momentum EMA for gradient estimation.
Posterior variance drives perturbation sampling. Shrinkage RLOO for
advantage estimation. Full candidate scoring. Dynamic sampling."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from optht import optht
from safetensors.torch import load_file, save_file
from torch.optim.lr_scheduler import CosineAnnealingLR

from ds_mezo.backend import _save_peft_adapter, _write_adapter_config
from ds_mezo.kernels import zo_muon_update, fused_power_iter, fused_agzo_perturbation, fused_perturb_dual
from ds_mezo.model_config import LayerSpec

_CONFIG_DEFAULTS: dict[str, Any] = {
    "output_dir": "output",
    "staging_dir": "/dev/shm/ds_mezo",
    "adapter_path": "",
    "total_steps": 1000,
    "seed": 42,
    "score_fn": lambda text: 0.0,
    "resume_from": None,
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

        cfg = {**_CONFIG_DEFAULTS, **config}

        self.score_fn: Callable[[str], float] = cfg["score_fn"]
        self.step_count = 0
        self.total_steps: int = cfg["total_steps"]

        self.output_dir = Path(cfg["output_dir"])
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.rng = torch.Generator(device="cuda")
        self.rng.manual_seed(cfg["seed"])

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

        rank = self.layers[0].B.shape[0]
        self.backend.sync_adapters({}, {}, self.layers)

        self.num_candidates = 4
        self.temperature = 1.0
        self.reward_ema = 0.0
        self._momentum = 0.0

        # eps = median(‖W‖_F) * eps_machine^(1/3) (Numerical Recipes §5.7)
        eps_machine = torch.finfo(torch.float32).eps
        norms = torch.stack([torch.linalg.norm(l.A) for l in self.layers] +
                             [torch.linalg.norm(l.B) for l in self.layers])
        self.eps = float(torch.median(norms)) * float(eps_machine) ** (1.0 / 3.0)

        # eta_max = eps (trust-region: step ≤ perturbation radius)
        self.eta_max = self.eps
        self.eta = self.eta_max
        self._lr_param = torch.nn.Parameter(torch.empty(0, device="cuda"))
        self._lr_opt = torch.optim.SGD([self._lr_param], lr=self.eta_max)
        self.lr_scheduler = CosineAnnealingLR(
            self._lr_opt, T_max=self.total_steps, eta_min=0.0,
        )

        self.momentum_buffers: dict[tuple[tuple[int, str], str], torch.Tensor] = {}
        self.variance_buffers: dict[tuple[tuple[int, str], str], torch.Tensor] = {}
        self.scratch_buffers: dict[tuple[tuple[int, str], str], torch.Tensor] = {}
        self.pos_buffers: dict[tuple[tuple[int, str], str], torch.Tensor] = {}
        self.neg_buffers: dict[tuple[tuple[int, str], str], torch.Tensor] = {}
        for layer in self.layers:
            key_A = (layer.key, "A")
            key_B = (layer.key, "B")
            self.momentum_buffers[key_A] = torch.zeros_like(layer.A)
            self.momentum_buffers[key_B] = torch.zeros_like(layer.B)
            self.variance_buffers[key_A] = torch.full_like(layer.A, self.eps ** 2)
            self.variance_buffers[key_B] = torch.full_like(layer.B, self.eps ** 2)
            self.scratch_buffers[key_A] = torch.zeros_like(layer.A)
            self.scratch_buffers[key_B] = torch.zeros_like(layer.B)
            self.pos_buffers[key_A] = torch.empty_like(layer.A)
            self.pos_buffers[key_B] = torch.empty_like(layer.B)
            self.neg_buffers[key_A] = torch.empty_like(layer.A)
            self.neg_buffers[key_B] = torch.empty_like(layer.B)

        self._bf16_ckpt: dict[str, torch.Tensor] = {}

        self.r_calib = 0
        self.activation_bases: dict[tuple[int, str], torch.Tensor] = {}

        eps_dtype = torch.finfo(torch.float32).eps

        # Power iteration steps: ceil(log(log(1/eps)/log(2))/log(3))
        self.power_iter_steps = math.ceil(
            math.log(math.log(1.0 / eps_dtype) / math.log(2.0)) / math.log(3.0)
        )
        self.norm_floor = torch.finfo(torch.float32).tiny

        if cfg["resume_from"] is not None:
            self._load_checkpoint(Path(cfg["resume_from"]))

    # -- Checkpoint Resume --------------------------------------------------

    def _load_checkpoint(self, checkpoint_dir: Path) -> None:
        """Restore full training state from a checkpoint directory."""
        tensors = load_file(str(checkpoint_dir / "optimizer_state.safetensors"), device="cuda")

        for layer in self.layers:
            idx, mod = layer.layer_idx, layer.module_name
            layer.A = tensors[f"master.layer{idx}.{mod}.A"]
            layer.B = tensors[f"master.layer{idx}.{mod}.B"]
            self.momentum_buffers[(layer.key, "A")] = tensors[f"momentum.layer{idx}.{mod}.A"]
            self.momentum_buffers[(layer.key, "B")] = tensors[f"momentum.layer{idx}.{mod}.B"]
            self.variance_buffers[(layer.key, "A")] = tensors[f"variance.layer{idx}.{mod}.A"]
            self.variance_buffers[(layer.key, "B")] = tensors[f"variance.layer{idx}.{mod}.B"]
            self.activation_bases[layer.key] = tensors[f"act_basis.layer{idx}.{mod}"]

        self.rng.set_state(tensors["rng_state"])

        with open(checkpoint_dir / "training_state.json") as f:
            state = json.load(f)
        self.step_count = state["step"]
        self.eta = state["eta"]
        self.reward_ema = state["reward_ema"]
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        self.r_calib = self.activation_bases[self.layers[0].key].shape[1]

    # -- Activation Subspace Tracking (§3.5) --------------------------------

    def _calibrate_activation_bases_full(self, input_data: list[str]) -> None:
        """Full SVD calibration — determines r_calib via Gavish-Donoho threshold."""
        activations = self.backend.extract_activations(input_data)
        rank = self.layers[0].B.shape[0]

        gpu_acts: dict[int, torch.Tensor] = {}
        for layer in self.layers:
            act = activations[layer.key]
            if id(act) not in gpu_acts:
                gpu_acts[id(act)] = act.cuda(non_blocking=True)

        ranks_per_activation: list[int] = []
        seen: set[int] = set()
        for layer in self.layers:
            act_id = id(activations[layer.key])
            if act_id in seen:
                continue
            seen.add(act_id)
            H = gpu_acts[act_id]
            sv = torch.linalg.svdvals(H).cpu().numpy()
            m, n = H.shape
            beta = min(m, n) / max(m, n)
            ranks_per_activation.append(optht(beta, sv=sv, sigma=None))
        self.r_calib = int(np.median(ranks_per_activation))

        svd_cache: dict[int, torch.Tensor] = {}
        for layer in self.layers:
            act_id = id(activations[layer.key])
            if act_id not in svd_cache:
                H = gpu_acts[act_id]
                _, _, V = torch.svd_lowrank(H, q=self.r_calib, niter=self.power_iter_steps)
                svd_cache[act_id] = V
            self.activation_bases[layer.key] = svd_cache[act_id]

    def _update_activation_bases(
        self, activations: dict[tuple[int, str], torch.Tensor],
    ) -> None:
        """Warm-started power iteration via fused Triton kernel."""
        processed: dict[int, torch.Tensor] = {}
        for layer in self.layers:
            act = activations[layer.key]
            act_id = id(act)
            if act_id not in processed:
                H = act.cuda(non_blocking=True)
                V = self.activation_bases[layer.key]
                processed[act_id] = fused_power_iter(
                    H, V, num_iters=self.power_iter_steps, norm_floor=self.norm_floor,
                )
            self.activation_bases[layer.key] = processed[act_id]

    # -- Perturbation (§3.2) ------------------------------------------------

    def _get_perturbation(self, layer: LayerState) -> tuple[torch.Tensor, torch.Tensor]:
        """AGZO subspace perturbation with variance-weighted sampling (DAP-optimal)."""
        A, B = layer.A, layer.B
        V_l = self.activation_bases[layer.key]

        # Variance-weighted B perturbation: sample proportional to posterior uncertainty
        var_B = self.variance_buffers[(layer.key, "B")]
        var_B_proj = var_B @ (V_l ** 2)  # (r, r_calib), always non-negative
        z_coeff_B = torch.randn(
            B.shape[0], V_l.shape[1], device="cuda", generator=self.rng,
        ) * torch.sqrt(var_B_proj)

        z_coeff_A = torch.randn(
            A.shape[0], V_l.shape[1], device="cuda", generator=self.rng,
        )

        return fused_agzo_perturbation(B, V_l, z_coeff_B, z_coeff_A, self.eps, self.norm_floor)

    def _score_contrastive(
        self,
        trajectories: list[list[int]],
        advantages: list[float],
        prompt_len: int,
    ) -> tuple[float, float]:
        """Advantage-weighted NLL under θ+ and θ- for all N candidates."""
        lp_pos = [lp[prompt_len:] for lp in self.backend.score(trajectories, self.backend.lora_pos)]
        lp_neg = [lp[prompt_len:] for lp in self.backend.score(trajectories, self.backend.lora_neg)]

        loss_pos = sum(adv * _mean_nll(lp) for adv, lp in zip(advantages, lp_pos))
        loss_neg = sum(adv * _mean_nll(lp) for adv, lp in zip(advantages, lp_neg))

        return loss_pos, loss_neg

    def _explore(
        self, batch: list[str],
    ) -> tuple[list[list[int]], list[float], int]:
        """Generate candidates, compute shrinkage RLOO advantages, return all trajectories."""
        self.backend.sync_adapters({}, {}, self.layers)
        request_outputs = self.backend.generate(
            batch, self.temperature, self.num_candidates
        )
        prompt_token_ids = request_outputs[0].prompt_token_ids

        scored = [
            (out, self.score_fn(out.text))
            for out in request_outputs[0].outputs
        ]

        # Shrinkage RLOO (James-Stein optimal baseline)
        rewards = torch.tensor([r for _, r in scored])
        N = len(scored)
        r_bar = float(rewards.mean())
        baselines_rloo = (rewards.sum() - rewards) / (N - 1)
        reward_var = float(rewards.var())
        lam = reward_var / (reward_var + (r_bar - self.reward_ema) ** 2 + self.norm_floor)
        baselines = (1.0 - lam) * baselines_rloo + lam * self.reward_ema
        advantages = [float(r - b) for r, b in zip(rewards, baselines)]

        # Update reward EMA using current β
        beta = self._momentum
        self.reward_ema = beta * self.reward_ema + (1.0 - beta) * r_bar

        # Build full token sequences for all candidates
        trajectories = [
            list(prompt_token_ids) + list(out.token_ids) for out, _ in scored
        ]
        prompt_len = len(prompt_token_ids)

        return trajectories, advantages, prompt_len

    def _perturb_and_sync(
        self, batch: list[str],
    ) -> dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]]:
        """Compute perturbations, apply dual perturbation, sync to backend."""
        batch_activations = self.backend.extract_activations(batch)
        self._update_activation_bases(batch_activations)

        perturbations = {l.key: self._get_perturbation(l) for l in self.layers}

        pos_layers: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        neg_layers: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in self.layers:
            z_A, z_B = perturbations[layer.key]
            key_A, key_B = (layer.key, "A"), (layer.key, "B")
            fused_perturb_dual(layer.A, z_A, self.pos_buffers[key_A], self.neg_buffers[key_A])
            fused_perturb_dual(layer.B, z_B, self.pos_buffers[key_B], self.neg_buffers[key_B])
            pos_layers[layer.key] = (self.pos_buffers[key_A], self.pos_buffers[key_B])
            neg_layers[layer.key] = (self.neg_buffers[key_A], self.neg_buffers[key_B])

        self.backend.sync_adapters(pos_layers, neg_layers, self.layers)
        return perturbations

    def _update_weights(
        self,
        perturbations: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]],
        dd: float,
    ) -> None:
        """Diagonal Kalman update + ZO-Muon spectral optimization."""
        max_window = int(math.sqrt(self.total_steps))
        self._momentum = 1.0 - 1.0 / min(self.step_count, max_window)
        beta = self._momentum
        q = self.eps ** 2 * (1.0 - beta * beta)

        # Pass 1: Kalman prediction + global reductions (ŷ and S across all layers)
        y_hat = 0.0
        s_sum = 0.0
        for layer in self.layers:
            z_A, z_B = perturbations[layer.key]
            for z_mat, suffix in [(z_A, "A"), (z_B, "B")]:
                key = (layer.key, suffix)
                mu = self.momentum_buffers[key]
                var = self.variance_buffers[key]
                mu.mul_(beta)
                var.mul_(beta * beta).add_(q)
                y_hat += (z_mat * mu).sum().item()
                s_sum += (z_mat * z_mat * var).sum().item()

        S = s_sum + self.norm_floor
        innovation = dd - y_hat
        inv_S = 1.0 / S

        # Pass 2: Kalman observation update + N-S orthogonalization
        for layer in self.layers:
            z_A, z_B = perturbations[layer.key]
            for z_mat, suffix, param in [(z_A, "A", layer.A), (z_B, "B", layer.B)]:
                key = (layer.key, suffix)
                mu = self.momentum_buffers[key]
                var = self.variance_buffers[key]
                K = var * z_mat * inv_S
                mu.add_(innovation * K)
                var.mul_(1.0 - K * z_mat)
                zo_muon_update(param, mu, self.scratch_buffers[key], self.eta, self.norm_floor)

        self.lr_scheduler.step()
        self.eta = self._lr_opt.param_groups[0]["lr"]

    def step(self, batch: list[str]) -> None:
        self.step_count += 1
        trajectories, advantages, prompt_len = self._explore(batch)

        # Dynamic sampling: skip when advantages are below SPSA truncation floor
        if max(abs(a) for a in advantages) < self.eps:
            self.lr_scheduler.step()
            self.eta = self._lr_opt.param_groups[0]["lr"]
            return

        perturbations = self._perturb_and_sync(batch)
        loss_pos, loss_neg = self._score_contrastive(trajectories, advantages, prompt_len)
        dd = float(loss_pos - loss_neg) / (2.0 * self.eps)
        self._update_weights(perturbations, dd)

    def train(self, prompts: list[str]) -> None:
        log_interval = self.total_steps // 100
        ckpt_interval = self.total_steps // 10

        for step_idx in range(self.step_count, self.total_steps):
            batch = [prompts[step_idx % len(prompts)]]
            self.step(batch)

            if (step_idx + 1) % log_interval == 0:
                print(
                    f"step={step_idx + 1}/{self.total_steps} "
                    f"lr={self.eta:.2e}"
                )

            if (step_idx + 1) % ckpt_interval == 0:
                self._save_checkpoint(step_idx + 1)

        self._save_checkpoint(self.total_steps)

    def _save_checkpoint(self, step: int) -> None:
        step_dir = self.checkpoint_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        rank = self.layers[0].B.shape[0]
        target_modules = sorted({l.module_name for l in self.layers})

        A_list = [l.A for l in self.layers]
        B_list = [l.B for l in self.layers]
        _save_peft_adapter(A_list, B_list, step_dir, self.layers, self._bf16_ckpt)
        _write_adapter_config(step_dir, rank, target_modules)

        tensors: dict[str, torch.Tensor] = {}
        for layer in self.layers:
            idx, mod = layer.layer_idx, layer.module_name
            tensors[f"master.layer{idx}.{mod}.A"] = layer.A
            tensors[f"master.layer{idx}.{mod}.B"] = layer.B
            tensors[f"momentum.layer{idx}.{mod}.A"] = self.momentum_buffers[(layer.key, "A")]
            tensors[f"momentum.layer{idx}.{mod}.B"] = self.momentum_buffers[(layer.key, "B")]
            tensors[f"variance.layer{idx}.{mod}.A"] = self.variance_buffers[(layer.key, "A")]
            tensors[f"variance.layer{idx}.{mod}.B"] = self.variance_buffers[(layer.key, "B")]
            tensors[f"act_basis.layer{idx}.{mod}"] = self.activation_bases[layer.key]
        tensors["rng_state"] = self.rng.get_state()
        save_file(tensors, str(step_dir / "optimizer_state.safetensors"))

        training_state = {
            "step": step,
            "eta": self.eta,
            "reward_ema": self.reward_ema,
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        with open(step_dir / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)

        adapter_dir = self.output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        _save_peft_adapter(A_list, B_list, adapter_dir, self.layers, self._bf16_ckpt)
        _write_adapter_config(adapter_dir, rank, target_modules)

        print(f"Checkpoint saved: {step_dir}")

