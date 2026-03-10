"""DS-MeZO Controller: AGZO + SPSA + ZO-Muon zeroth-order optimizer."""

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
from torch.optim._muon import DEFAULT_A, DEFAULT_B, DEFAULT_C
from torch.optim.lr_scheduler import CosineAnnealingLR

from ds_mezo.backend import _save_peft_adapter, _write_adapter_config
from ds_mezo.kernels import zo_muon_update, fused_power_iter, fused_agzo_perturbation, fused_perturb_dual
from ds_mezo.model_config import LayerSpec

_CONFIG_DEFAULTS: dict[str, Any] = {
    "output_dir": "output",
    "staging_dir": "/dev/shm/ds_mezo",
    "adapter_path": "",
    "total_steps": 1000,
    "hybrid_switch_step": 0,
    "seed": 42,
    "score_fn": lambda text: 0.0,
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
        self.hybrid_switch_step: int = cfg["hybrid_switch_step"]

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
        self.explore_temperature = 1.0

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

        self._bf16_ckpt: dict[str, torch.Tensor] = {}

        self.r_calib = 0
        self.activation_bases: dict[tuple[int, str], torch.Tensor] = {}

        eps_dtype = torch.finfo(torch.float32).eps

        # N-S iterations: 3-term scalar simulation for s_min = 1/sqrt(rank)
        self.ns_iterations = self._ns_iters_for_smin(
            1.0 / math.sqrt(rank), float(eps_dtype),
        )

        # Power iteration steps: ceil(log(log(1/eps)/log(2))/log(3))
        self.power_iter_steps = math.ceil(
            math.log(math.log(1.0 / eps_dtype) / math.log(2.0)) / math.log(3.0)
        )
        self.norm_floor = torch.finfo(torch.float32).tiny

    @staticmethod
    def _ns_iters_for_smin(s_min: float, eps_dtype: float) -> int:
        """Simulate scalar N-S map from s_min until convergence."""
        a, b, c = DEFAULT_A, DEFAULT_B, DEFAULT_C
        s = s_min
        for k in range(20):
            s2 = s * s
            s = s * (a + b * s2 + c * s2 * s2)
            if abs(s - 1.0) < eps_dtype:
                return k + 1
        return 20

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
            beta = max(m, n) / min(m, n)
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
        """AGZO subspace perturbation via fused Triton kernel."""
        A, B = layer.A, layer.B
        V_l = self.activation_bases[layer.key]

        z_coeff_B = torch.randn(
            B.shape[0], V_l.shape[1], device="cuda", generator=self.rng,
        )
        z_coeff_A = torch.randn(
            A.shape[0], V_l.shape[1], device="cuda", generator=self.rng,
        )

        return fused_agzo_perturbation(B, V_l, z_coeff_B, z_coeff_A, self.eps, self.norm_floor)

    def _score_contrastive(
        self,
        trajectories: tuple[list[int], list[int]],
        advantages: tuple[float, float],
        prompt_len: int,
    ) -> tuple[float, float]:
        """Advantage-weighted NLL under θ+ and θ-."""
        winner_tokens, loser_tokens = trajectories
        adv_w, adv_l = advantages

        seqs = [winner_tokens, loser_tokens]
        lp_pos = [lp[prompt_len:] for lp in self.backend.score(seqs, self.backend.lora_pos)]
        lp_neg = [lp[prompt_len:] for lp in self.backend.score(seqs, self.backend.lora_neg)]

        loss_pos = adv_w * _mean_nll(lp_pos[0]) + adv_l * _mean_nll(lp_pos[1])
        loss_neg = adv_w * _mean_nll(lp_neg[0]) + adv_l * _mean_nll(lp_neg[1])

        return loss_pos, loss_neg

    def _compute_loss_sft(
        self, token_ids: list[int], prompt_len: int,
    ) -> tuple[float, float]:
        """NLL on target tokens under θ+ and θ-."""
        lp_pos = self.backend.score([token_ids], self.backend.lora_pos)[0][prompt_len:]
        lp_neg = self.backend.score([token_ids], self.backend.lora_neg)[0][prompt_len:]

        return _mean_nll(lp_pos), _mean_nll(lp_neg)

    def _explore(
        self, batch: list[str],
    ) -> tuple[tuple[list[int], list[int]], tuple[float, float], int]:
        """Generate candidates, compute RLOO advantages, return (trajectories, advantages, prompt_len)."""
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

        # RLOO advantages with REINFORCE++ normalization
        rewards = torch.tensor([r for _, r in scored])
        N = len(scored)
        baselines = (rewards.sum() - rewards) / (N - 1)
        advantages = (rewards - baselines) / rewards.std()
        adv_w = float(advantages[0])   # best (sorted descending)
        adv_l = float(advantages[-1])  # worst

        winner_full = list(prompt_token_ids) + list(best_output.token_ids)
        loser_full = list(prompt_token_ids) + list(worst_output.token_ids)
        prompt_len = len(prompt_token_ids)

        return (winner_full, loser_full), (adv_w, adv_l), prompt_len

    def step(self, batch: list[str] | dict[str, Any]) -> None:
        self.step_count += 1
        use_rl_loss = self.step_count > self.hybrid_switch_step

        if use_rl_loss:
            trajectories, advantages, prompt_len = self._explore(batch)

        input_for_activations = batch if isinstance(batch, list) else [batch["prompt_text"]]
        batch_activations = self.backend.extract_activations(input_for_activations)
        self._update_activation_bases(batch_activations)

        perturbations: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in self.layers:
            perturbations[layer.key] = self._get_perturbation(layer)

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

        if use_rl_loss:
            loss_pos, loss_neg = self._score_contrastive(trajectories, advantages, prompt_len)
        else:
            loss_pos, loss_neg = self._compute_loss_sft(batch["token_ids"], batch["prompt_len"])

        dd = float(loss_pos - loss_neg) / (2.0 * self.eps)

        max_window = int(math.sqrt(self.total_steps))
        momentum = 1.0 - 1.0 / min(self.step_count, max_window)

        for layer in self.layers:
            z_A, z_B = perturbations[layer.key]
            key_A = (layer.key, "A")
            key_B = (layer.key, "B")
            zo_muon_update(layer.A, self.momentum_buffers[key_A],
                           z_A, self.scratch_buffers[key_A],
                           dd, momentum, self.eta,
                           self.ns_iterations, self.norm_floor)
            zo_muon_update(layer.B, self.momentum_buffers[key_B],
                           z_B, self.scratch_buffers[key_B],
                           dd, momentum, self.eta,
                           self.ns_iterations, self.norm_floor)

        self.lr_scheduler.step()
        self.eta = self._lr_opt.param_groups[0]["lr"]
        if use_rl_loss:
            progress = self.step_count / self.total_steps
            self.explore_temperature = 0.5 * (1 + math.cos(math.pi * progress))

    def train(
        self,
        rl_data: list[str],
        sft_data: list[dict[str, Any]] | None = None,
    ) -> None:
        log_interval = self.total_steps // 100
        ckpt_interval = self.total_steps // 10

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
                    f"temp={self.explore_temperature:.3f}"
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
            tensors[f"act_basis.layer{idx}.{mod}"] = self.activation_bases[layer.key]
        tensors["rng_state"] = self.rng.get_state()
        save_file(tensors, str(step_dir / "optimizer_state.safetensors"))

        training_state = {
            "step": step,
            "eta": self.eta,
            "explore_temperature": self.explore_temperature,
        }
        with open(step_dir / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)

        adapter_dir = self.output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        _save_peft_adapter(A_list, B_list, adapter_dir, self.layers, self._bf16_ckpt)
        _write_adapter_config(adapter_dir, rank, target_modules)

        print(f"Checkpoint saved: {step_dir}")

