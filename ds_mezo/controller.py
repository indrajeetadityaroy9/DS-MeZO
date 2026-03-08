"""DS-MeZO Controller: zeroth-order optimization for LLM fine-tuning.

Single-file implementation of the DS-MeZO training loop including:
- PiSSA adapter management and vLLM integration
- AGZO activation-guided subspace perturbation
- Momentum-aligned sensitivity masking (applied at update step)
- SPSA gradient estimation with adaptive epsilon and DD clipping
- ZO-Muon spectral optimizer (Newton-Schulz orthogonalization)
- RLOO contrastive selection with gradient regularization
- Entropy-guided temperature annealing
- KL divergence constraint

Reference: DS_MeZO_Combined.md §8 with Bug 1 (GR cancellation) and Bug 2 (masking timing) fixes.
"""

import math
import os
import json
import re
import shutil

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from safetensors.torch import save_file, load_file
from ds_mezo.kernels import zo_muon_update, fused_perturb_dual


ADAPTER_STAGING_DIR = "/dev/shm/ds_mezo"

DEFAULTS = {
    "rank": 16,
    "target_modules": ["q_proj", "v_proj"],
    "total_steps": 1000,
    "eta_max": 1e-4,
    "momentum": 0.9,
    "num_candidates": 4,
    "lambda_gr": 0.01,
    "delta_kl": 0.01,
    "r_calib": 8,
    "T_max": 1.0,
    "T_min": 0.3,
    "drift_threshold": 0.95,
    "eps_floor": 0.1,
    "mask_warmup": 10,
    "score_fn": lambda text: len(text),
}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def write_adapter_config(adapter_dir, rank, target_modules):
    """Write PEFT adapter config (once at init, not every sync)."""
    config = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": rank,
        "target_modules": target_modules,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f)


def save_peft_adapter(A_list, B_list, adapter_dir, layer_configs):
    """Serialize A, B matrices as PEFT-compatible LoRA adapter (BF16).

    PiSSA convention: W = W_res + A @ B where A:(d_out, r), B:(r, d_in)
    PEFT convention:  W = W0 + lora_B @ lora_A where lora_A:(r, d_in), lora_B:(d_out, r)
    Therefore: lora_A.weight = B, lora_B.weight = A"""
    tensors = {}
    for layer_idx, (A_l, B_l) in enumerate(zip(A_list, B_list)):
        prefix = layer_configs[layer_idx]["peft_prefix"]
        tensors[f"{prefix}.lora_A.weight"] = B_l.bfloat16()   # B → lora_A (r × d_in)
        tensors[f"{prefix}.lora_B.weight"] = A_l.bfloat16()   # A → lora_B (d_out × r)
    save_file(tensors, os.path.join(adapter_dir, "adapter_model.safetensors"))


def extract_prompt_logprobs(output, prompt_token_ids):
    """Extract per-token logprobs from vLLM output."""
    logprobs = []
    for i, token_lp in enumerate(output.prompt_logprobs[1:], 1):
        tok_id = prompt_token_ids[i]
        logprobs.append(token_lp[tok_id].logprob)
    return logprobs


def _register_activation_hooks(worker, target_modules):
    """Register forward hooks in the worker process. Called via collective_rpc.

    vLLM merges q/k/v projections into a single qkv_proj module.
    Since q_proj and v_proj share the same input activations, we register
    on qkv_proj and store the activation under all target module keys."""
    model = worker.get_model()
    worker._ds_mezo_hooks = []
    worker._ds_mezo_activations = {}

    for name, module in model.named_modules():
        if "base_layer" in name:
            continue
        match = re.search(r"layers\.(\d+)", name)
        if match is None:
            continue
        layer_idx = int(match.group(1))

        # Match target modules directly or via vLLM's merged qkv_proj.
        # Check qkv_proj first since "q_proj" is a substring of "qkv_proj".
        if "qkv_proj" in name:
            # vLLM merges q_proj+k_proj+v_proj — input activations are shared
            matched_targets = [tm for tm in target_modules
                               if tm in ("q_proj", "v_proj", "k_proj")]
        else:
            matched_targets = [tm for tm in target_modules if tm in name]
        if not matched_targets:
            continue

        keys = [(layer_idx, tm) for tm in matched_targets]

        def hook_fn(mod, inp, out, ks=keys):
            act = inp[0].detach().float().cpu()
            for k in ks:
                worker._ds_mezo_activations[k] = act

        worker._ds_mezo_hooks.append(module.register_forward_hook(hook_fn))
    return len(worker._ds_mezo_hooks)


def _collect_and_remove_hooks(worker):
    """Collect activation data and remove hooks. Called via collective_rpc."""
    activations = worker._ds_mezo_activations
    for h in worker._ds_mezo_hooks:
        h.remove()
    worker._ds_mezo_hooks = []
    worker._ds_mezo_activations = {}
    return activations


def extract_layer_activations(engine, input_data, target_modules):
    """Extract per-layer input activations via forward hooks on vLLM's model.

    vLLM v0.17+ runs the model in a separate worker process.
    Strategy: register hooks → generate (triggers forward pass) → collect."""
    engine.llm_engine.collective_rpc(
        _register_activation_hooks, args=(target_modules,)
    )

    engine.generate(
        input_data,
        SamplingParams(max_tokens=1, temperature=0.0),
    )

    results = engine.llm_engine.collective_rpc(_collect_and_remove_hooks)
    return results[0]  # Single-GPU: first worker's results


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class DSMeZO_Controller:
    def __init__(self, engine, config):
        self.engine = engine
        cfg = {**DEFAULTS, **config}

        self.score_fn = cfg["score_fn"]
        self.step_count = 0
        self.total_steps = cfg["total_steps"]
        self.target_modules = cfg["target_modules"]

        # Load pre-computed PiSSA adapter into FP32 master weights on GPU
        adapter_path = cfg["adapter_path"]
        adapter_tensors = load_file(
            os.path.join(adapter_path, "adapter_model.safetensors"),
            device="cuda",
        )

        # Parse layer structure from adapter keys
        layer_set = set()
        for key in adapter_tensors:
            match = re.search(r"layers\.(\d+)\.self_attn\.(\w+)", key)
            if match:
                layer_set.add((int(match.group(1)), match.group(2)))

        self.layers = []
        for layer_idx, module_name in sorted(layer_set):
            prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{module_name}"
            # PEFT → PiSSA: lora_A (r×d_in) = B, lora_B (d_out×r) = A
            B = adapter_tensors[f"{prefix}.lora_A.weight"].float()  # r × d_in
            A = adapter_tensors[f"{prefix}.lora_B.weight"].float()  # d_out × r
            self.layers.append({
                "A": A, "B": B,
                "layer_idx": layer_idx,
                "module_name": module_name,
                "peft_prefix": prefix,
            })
        del adapter_tensors

        self.num_layers = len(self.layers)
        rank = cfg["rank"]

        # Create fresh staging directories (deterministic clean start)
        shutil.rmtree(ADAPTER_STAGING_DIR, ignore_errors=True)
        self.adapter_dir_pos = os.path.join(ADAPTER_STAGING_DIR, "adapter_pos")
        self.adapter_dir_neg = os.path.join(ADAPTER_STAGING_DIR, "adapter_neg")
        self.checkpoint_dir = os.path.join(ADAPTER_STAGING_DIR, "checkpoints")
        os.makedirs(self.adapter_dir_pos)
        os.makedirs(self.adapter_dir_neg)
        os.makedirs(self.checkpoint_dir)

        # Write adapter config once (not on every sync)
        unique_modules = list({l["module_name"] for l in self.layers})
        write_adapter_config(self.adapter_dir_pos, rank, unique_modules)
        write_adapter_config(self.adapter_dir_neg, rank, unique_modules)

        self.lora_pos = LoRARequest("adapter_pos", 1, self.adapter_dir_pos, load_inplace=True)
        self.lora_neg = LoRARequest("adapter_neg", 2, self.adapter_dir_neg, load_inplace=True)
        self._sync_adapters({}, {})

        self.score_params = SamplingParams(
            max_tokens=1, prompt_logprobs=1, temperature=0.0
        )

        # Exploration — entropy-guided temperature annealing (§7.3)
        self.num_candidates = cfg["num_candidates"]
        self.T_max = cfg["T_max"]
        self.T_min = cfg["T_min"]
        self.explore_temperature = self.T_max
        self.initial_entropy = None
        self._last_reward_range = 0

        # Gradient regularization (§4.3a)
        self.lambda_gr = cfg["lambda_gr"]
        # KL constraint (§4.3b)
        self.delta_kl = cfg["delta_kl"]

        # Perturbation — adaptive epsilon (§2.1)
        self.eps_base = 1e-3 / math.sqrt(rank)  # derived from rank
        self.eps = self.eps_base
        self.eps_floor = cfg["eps_floor"]
        self.initial_loss_ema = None

        # Optimizer: ZO-Muon (§5)
        self.eta_max = cfg["eta_max"]
        self.eta_min = self.eta_max / 100  # derived from eta_max
        self.eta = self.eta_max
        self.momentum = cfg["momentum"]

        # Pre-allocate momentum and scratch buffers for all layers
        self.momentum_buffers = {}
        self.scratch_buffers = {}
        for layer in self.layers:
            key = (layer["layer_idx"], layer["module_name"])
            key_A = (key, "A")
            key_B = (key, "B")
            self.momentum_buffers[key_A] = torch.zeros_like(layer["A"])
            self.momentum_buffers[key_B] = torch.zeros_like(layer["B"])
            self.scratch_buffers[key_A] = torch.zeros_like(layer["A"])
            self.scratch_buffers[key_B] = torch.zeros_like(layer["B"])

        # Masking warmup — full perturbation for first N steps (§3.3)
        self.mask_warmup = cfg["mask_warmup"]

        # Activation subspace tracking — per-step power iteration (§3.5)
        self.r_calib = cfg["r_calib"]
        self.power_iter_steps = 3  # K=3 per §3.5
        self.drift_threshold = cfg["drift_threshold"]
        self.activation_bases = {}

        # Directional derivative clipping (§2.1)
        self.dd_ema = None

        # Health monitoring (§4.4)
        self.loss_ema = None

    # -- Activation Subspace Tracking (§3.5) --------------------------------

    def _calibrate_activation_bases_full(self, input_data):
        """Full SVD calibration — used at init and on drift detection."""
        activations = extract_layer_activations(
            self.engine, input_data, self.target_modules
        )
        for layer in self.layers:
            key = (layer["layer_idx"], layer["module_name"])
            H = activations[key]  # batch*seq_len × d_in
            r_calib = min(self.r_calib, H.shape[1])
            _, _, V = torch.svd_lowrank(H, q=r_calib, niter=2)
            self.activation_bases[key] = V.float()  # d_in × r_calib

    def _update_activation_bases_power_iter(self, activations):
        """Per-step warm-started power iteration (§3.5)."""
        needs_full_recalib = False
        for layer in self.layers:
            key = (layer["layer_idx"], layer["module_name"])
            H = activations[key].float()
            V_old = self.activation_bases[key]
            V = V_old.clone()
            for _ in range(self.power_iter_steps):
                V = H.T @ (H @ V)
                V, _ = torch.linalg.qr(V)
            # Drift check (§3.5 — LOTUS criterion)
            alignment = torch.trace(V.T @ V_old).abs() / self.r_calib
            if alignment < self.drift_threshold:
                needs_full_recalib = True
            self.activation_bases[key] = V
        return needs_full_recalib

    # -- Perturbation (§3.2) ------------------------------------------------

    def _get_perturbation(self, layer):
        """AGZO subspace perturbation for A and B.
        Masking is NOT applied here — Bug 2 fix moves it to update step."""
        A, B = layer["A"], layer["B"]
        key = (layer["layer_idx"], layer["module_name"])
        V_l = self.activation_bases[key].to(device=B.device)

        # B: AGZO subspace perturbation (r × d_in, in span(V_l))
        z_coeff_B = torch.randn(B.shape[0], V_l.shape[1], device=B.device)
        z_B = z_coeff_B @ V_l.T

        # A: projected perturbation
        BV = B @ V_l  # r × r_calib
        Q, _ = torch.linalg.qr(BV)  # r × min(r, r_calib)
        z_coeff_A = torch.randn(A.shape[0], Q.shape[1], device=A.device)
        z_A = z_coeff_A @ Q.T  # d_out × r

        return z_A * self.eps, z_B * self.eps

    # -- Adapter Sync -------------------------------------------------------

    def _sync_adapters(self, pos_overrides, neg_overrides):
        """Serialize PiSSA adapters to /dev/shm for vLLM."""
        def get_AB(overrides):
            A_list, B_list = [], []
            for layer in self.layers:
                k = (layer["layer_idx"], layer["module_name"])
                if k in overrides:
                    A_l, B_l = overrides[k]
                else:
                    A_l, B_l = layer["A"], layer["B"]
                A_list.append(A_l)
                B_list.append(B_l)
            return A_list, B_list

        A_pos, B_pos = get_AB(pos_overrides)
        save_peft_adapter(A_pos, B_pos, self.adapter_dir_pos, self.layers)

        A_neg, B_neg = get_AB(neg_overrides)
        save_peft_adapter(A_neg, B_neg, self.adapter_dir_neg, self.layers)

    # -- Scoring (§4.3) -----------------------------------------------------

    def _get_prompt_logprobs(self, token_sequences, lora_request):
        """Score token sequences under a LoRA adapter via vLLM prefill."""
        prompts = [{"prompt_token_ids": seq} for seq in token_sequences]
        outputs = self.engine.generate(
            prompts, sampling_params=self.score_params, lora_request=lora_request,
        )
        return [
            extract_prompt_logprobs(out, seq)
            for out, seq in zip(outputs, token_sequences)
        ]

    def _score_contrastive(self, trajectories, advantages):
        """Advantage-weighted NLL with asymmetric gradient regularization.

        Bug 1 fix: GR term computed as NLL divergence between θ+ and θ-,
        added only to loss_pos so it doesn't cancel in finite differences."""
        winner_tokens, loser_tokens = trajectories
        adv_w, adv_l = advantages

        lp_pos = self._get_prompt_logprobs(
            [winner_tokens, loser_tokens], self.lora_pos
        )
        lp_pos = [lp[self.prompt_len:] for lp in lp_pos]
        lp_neg = self._get_prompt_logprobs(
            [winner_tokens, loser_tokens], self.lora_neg
        )
        lp_neg = [lp[self.prompt_len:] for lp in lp_neg]

        def nll(logprobs):
            total = sum(float(-lp) for lp in logprobs)
            return total / len(logprobs)

        loss_pos = float(adv_w) * nll(lp_pos[0]) + float(adv_l) * nll(lp_pos[1])
        loss_neg = float(adv_w) * nll(lp_neg[0]) + float(adv_l) * nll(lp_neg[1])

        # Bug 1 fix: asymmetric GR — NLL divergence between perturbation directions
        total_tokens = 0
        nll_div = 0.0
        for lps_p, lps_n in zip(lp_pos, lp_neg):
            for p, n in zip(lps_p, lps_n):
                nll_div += (float(-p) - float(-n)) ** 2
                total_tokens += 1
        nll_div /= total_tokens
        loss_pos += self.lambda_gr * nll_div  # Asymmetric — only loss_pos

        return loss_pos, loss_neg

    # -- Exploration (§4.1, §4.2, §7.3) -------------------------------------

    def _update_temperature(self):
        """Cosine temperature annealing (§7.3)."""
        progress = self.step_count / self.total_steps
        self.explore_temperature = (
            self.T_min
            + 0.5 * (self.T_max - self.T_min)
            * (1 + math.cos(math.pi * progress))
        )

    def _explore(self, batch):
        """Generate candidates, compute RLOO advantages."""
        self._sync_adapters({}, {})
        gen_params = SamplingParams(
            n=self.num_candidates, temperature=self.explore_temperature
        )
        request_outputs = self.engine.generate(
            batch, sampling_params=gen_params, lora_request=self.lora_pos
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

        # Store reward range for entropy monitoring (applied after schedule updates)
        self._last_reward_range = best_reward - worst_reward

        winner_full = list(prompt_token_ids) + list(best_output.token_ids)
        loser_full = list(prompt_token_ids) + list(worst_output.token_ids)
        self.prompt_len = len(prompt_token_ids)

        return (winner_full, loser_full), (adv_w, adv_l)

    # -- Adaptive Epsilon (§2.1) --------------------------------------------

    def _update_eps(self):
        """Scale epsilon proportionally to loss EMA."""
        if self.loss_ema is not None and self.initial_loss_ema is not None:
            ratio = max(self.loss_ema / self.initial_loss_ema, self.eps_floor)
            self.eps = self.eps_base * ratio

    # -- KL Constraint (§4.3b) ----------------------------------------------

    def _apply_kl_constraint(self, loss_pos, loss_neg):
        """Scale down LR if update exceeds KL budget."""
        kl_approx = abs(loss_pos - loss_neg)
        if kl_approx > self.delta_kl:
            return self.eta * self.delta_kl / kl_approx
        return self.eta

    # -- Cosine LR (§5.3) --------------------------------------------------

    def _update_lr(self):
        progress = self.step_count / self.total_steps
        self.eta = (
            self.eta_min
            + 0.5 * (self.eta_max - self.eta_min)
            * (1 + math.cos(math.pi * progress))
        )

    # -- Health Monitoring (§4.4) -------------------------------------------

    def _check_health(self, loss_pos, loss_neg):
        """Spike detection — skip step if NLL > 5× EMA."""
        avg_nll = (abs(loss_pos) + abs(loss_neg)) / 2
        if self.loss_ema is None:
            self.loss_ema = avg_nll
            self.initial_loss_ema = avg_nll
            return True
        if self.loss_ema > 1e-8 and avg_nll > 5 * self.loss_ema:
            return False
        self.loss_ema = 0.95 * self.loss_ema + 0.05 * avg_nll
        return True

    # -- Directional Derivative Clipping (§2.1) -----------------------------

    def _clip_dd(self, dd):
        """Clip DD at 3× its running EMA."""
        if self.dd_ema is None:
            self.dd_ema = abs(dd)
            return dd
        clip_val = 3 * self.dd_ema
        clipped = max(-clip_val, min(dd, clip_val))
        self.dd_ema = 0.95 * self.dd_ema + 0.05 * abs(dd)
        return clipped

    # -- Main Training Step (§7.1) ------------------------------------------

    def step(self, batch):
        self.step_count += 1

        # Exploration with entropy-guided temperature
        trajectories, advantages = self._explore(batch)

        # Update activation bases via power iteration
        batch_activations = extract_layer_activations(
            self.engine, batch, self.target_modules
        )
        needs_recalib = self._update_activation_bases_power_iter(batch_activations)
        if needs_recalib:
            self._calibrate_activation_bases_full(batch)

        # Generate perturbations for all layers
        perturbations = {}
        for layer in self.layers:
            key = (layer["layer_idx"], layer["module_name"])
            perturbations[key] = self._get_perturbation(layer)

        # Construct perturbed adapters (fused dual perturbation kernel)
        pos_layers, neg_layers = {}, {}
        for layer in self.layers:
            key = (layer["layer_idx"], layer["module_name"])
            z_A, z_B = perturbations[key]
            pos_A, neg_A = torch.empty_like(layer["A"]), torch.empty_like(layer["A"])
            pos_B, neg_B = torch.empty_like(layer["B"]), torch.empty_like(layer["B"])
            fused_perturb_dual(layer["A"], z_A, pos_A, neg_A)
            fused_perturb_dual(layer["B"], z_B, pos_B, neg_B)
            pos_layers[key] = (pos_A, pos_B)
            neg_layers[key] = (neg_A, neg_B)

        # Score (4 prefills — 2 sequences × 2 perturbation directions)
        self._sync_adapters(pos_layers, neg_layers)
        loss_pos, loss_neg = self._score_contrastive(trajectories, advantages)

        # Update with safety checks
        if self._check_health(loss_pos, loss_neg):
            raw_dd = float(loss_pos - loss_neg) / (2.0 * self.eps)
            dd = self._clip_dd(raw_dd)

            # KL constraint — scale eta if update too large
            effective_eta = self._apply_kl_constraint(loss_pos, loss_neg)
            saved_eta = self.eta
            self.eta = effective_eta

            for layer in self.layers:
                key = (layer["layer_idx"], layer["module_name"])
                z_A, z_B = perturbations[key]
                key_A = (key, "A")
                key_B = (key, "B")

                # Bug 2 fix: masking on A only, after warmup (§3.3)
                do_mask = self.step_count > self.mask_warmup
                zo_muon_update(layer["A"], self.momentum_buffers[key_A],
                               z_A, self.scratch_buffers[key_A],
                               dd, self.momentum, self.eta, apply_mask=do_mask)
                zo_muon_update(layer["B"], self.momentum_buffers[key_B],
                               z_B, self.scratch_buffers[key_B],
                               dd, self.momentum, self.eta, apply_mask=False)

            self.eta = saved_eta

        # Schedule updates
        self._update_lr()
        self._update_eps()
        self._update_temperature()

        # Entropy monitoring AFTER cosine annealing — boost persists to next step (§7.3)
        reward_range = self._last_reward_range
        if self.initial_entropy is None and reward_range > 0:
            self.initial_entropy = reward_range
        if (self.initial_entropy and reward_range > 0
                and reward_range < 0.5 * self.initial_entropy):
            self.explore_temperature = min(
                self.explore_temperature * 1.5, self.T_max
            )

    # -- Main Training Loop -------------------------------------------------

    def train(self, prompts):
        """Train for total_steps, cycling through prompts."""
        # Initial activation calibration using first prompt
        self._calibrate_activation_bases_full([prompts[0]])

        for step_idx in range(self.total_steps):
            batch = [prompts[step_idx % len(prompts)]]
            self.step(batch)

            if (step_idx + 1) % 10 == 0:
                print(
                    f"step={step_idx + 1}/{self.total_steps} "
                    f"lr={self.eta:.2e} eps={self.eps:.2e} "
                    f"temp={self.explore_temperature:.3f} "
                    f"loss_ema={self.loss_ema:.4f}" if self.loss_ema else
                    f"step={step_idx + 1}/{self.total_steps}"
                )

            # Checkpoint every 100 steps
            if (step_idx + 1) % 100 == 0:
                self._save_checkpoint(step_idx + 1)

        # Final checkpoint
        self._save_checkpoint(self.total_steps)

    def _save_checkpoint(self, step):
        """Minimal checkpoint: FP32 masters + momentum + step count."""
        state = {
            "step": step,
            "layers": [
                {"A": l["A"], "B": l["B"],
                 "layer_idx": l["layer_idx"], "module_name": l["module_name"]}
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
        path = os.path.join(self.checkpoint_dir, f"step_{step}.pt")
        torch.save(state, path)
        print(f"Checkpoint saved: {path}")
