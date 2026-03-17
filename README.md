# Zeroth-Order Post-Training on a Single GPU

RL post-training of large language models without backpropagation. It uses zeroth-order (ZO) gradient estimation via SPSA on PiSSA adapters to optimize non-differentiable reward signals — code execution, proof verification, tool use — at near-inference memory cost on a single GPU.

This enables RL post-training (RLVR, GRPO-style) on hardware where backpropagation-based methods like PPO, GRPO, or DeepSeek-R1 require multi-GPU clusters with 2-4x training memory overhead.

The current RL post-training ecosystem has converged on REINFORCE methods with leave-one-out baselines as the dominant approach for reasoning and code generation:

| Framework | Methods | Execution | Scale |
|:---|:---|:---|:---|
| [TRL (HuggingFace)](https://github.com/huggingface/trl) | RLOOTrainer, GRPOTrainer, PPO | Backprop | Single/multi-GPU |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | PPO, GRPO, REINFORCE++, RLOO | Backprop (Ray) | Multi-GPU clusters |
| [AgentGym-RL](https://github.com/WooooDyy/AgentGym-RL) | PPO, GRPO, RLOO, REINFORCE++ | Backprop | Multi-GPU |
| **DS-MeZO (this work)** | **RLOO + SPSA + ZO-Muon** | **Zeroth-order (no backprop)** | **Single GPU** |

All frameworks share the same RLOO advantage formula:
```
baselines = (rewards.sum() - rewards) / (N - 1)
advantages = rewards - baselines
```

DS-MeZO diverges from the ecosystem in three ways:
- **No backpropagation**: Advantages feed into SPSA contrastive scoring instead of policy gradient backward passes
- **James-Stein estimator shrinkage**: Shrinkage-optimal baseline blending RLOO with reward EMA (not in any existing framework)
- **Zeroth-order optimization**: Diagonal Kalman filter + ZO-Muon spectral update replaces Adam/SGD on gradient estimates

REINFORCE++-baseline (ProRL V2, ScaleRL) and GRPO (DeepSeek-R1) are the closest backprop-based counterparts. DS-MeZO aims to achieves the same RL loop at near-inference memory cost by eliminating the backward pass entirely.
