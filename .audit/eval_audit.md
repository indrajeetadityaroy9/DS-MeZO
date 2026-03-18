# DS-MeZO Evaluation Suite Audit

Audit date: 2026-03-17
Files analyzed:
- `eval/__init__.py`
- `eval/rewards.py`
- `eval/benchmarks.py`
- `eval/rl_bench_eval.py`
- `eval/grpo_baseline.py`

---

## 1. File-by-File Analysis

### eval/__init__.py

Minimal module init. Sets `HF_ALLOW_CODE_EVAL=1` environment variable, which is required by the HuggingFace `code_eval` metric to enable execution of untrusted code. This is a package-level side effect -- importing `eval` anywhere enables code execution globally.

---

### eval/rewards.py

#### Reward Functions

- **`make_exec_reward()`**: Factory returning a `(reward_fn, set_problem_fn)` pair. Uses closure state to hold the current problem's `tests` and `imports`. The reward function extracts code from markdown fences via `extract_code()`, then delegates to `_score_code_solution()`.
- **`_score_code_solution(code, tests, imports)`**: Passes each test as a separate reference to HuggingFace `code_eval.compute()` with `k=[1]`, `num_workers=1`, `timeout=3.0`. Returns fraction of tests passed (continuous [0,1] reward, not binary).

#### Code Extraction

- `extract_code()`: Regex `r"```(?:python)?\s*\n(.*?)```"` extracts content from markdown code fences. Falls back to raw text if no fences found. Joins multiple blocks with newlines.

#### Datasets Supported

1. **MBPP (sanitized)**: `google-research-datasets/mbpp`, sanitized split, train partition. Prompt format: docstring wrapping the problem description and first test case.
2. **APPS (introductory)**: `codeparrot/apps`, train split, filtered to `difficulty == "introductory"`, capped at 7000 problems. Constructs assert-based tests from JSON `input_output` pairs.

#### Issues

1. **APPS test construction is fragile**: `assert solution({repr(inp.strip())}) == {repr(out.strip())}` assumes every APPS problem defines a `solution()` function and that inputs/outputs are simple string-representable values. Many APPS problems use stdin/stdout I/O, not function calls. This will produce systematically wrong tests for a large fraction of the dataset.
2. **Timeout of 3.0s** is hardcoded and relatively tight. Complex APPS introductory problems with loops may time out spuriously.
3. **`num_workers=1`** in `_score_code_solution` means test execution is sequential. Each training step blocks on serial code execution. This is a performance bottleneck but not a correctness issue.
4. **Closure-based state in `make_exec_reward()`** is not thread-safe. If the controller ever ran steps concurrently (it does not currently), tests/imports could be overwritten mid-evaluation.

#### Sandboxing

Relies entirely on HuggingFace `evaluate`'s `code_eval` metric, which uses `multiprocessing` with a timeout. There is **no containerization, seccomp, or namespace isolation**. The `HF_ALLOW_CODE_EVAL=1` flag explicitly acknowledges this. Arbitrary code from model completions runs in the host process's security context.

---

### eval/benchmarks.py

#### Evaluation Metrics

- **pass@k** (k=1 and k=10): Uses HuggingFace `code_eval.compute()` which implements the unbiased estimator from Chen et al. (Codex paper): `pass@k = 1 - C(n-c, k) / C(n, k)`.
- **95% Bootstrap CI**: `scipy.stats.bootstrap` with `n_resamples=10000`, percentile method, seeded at 42. Applied to per-task pass@1 and pass@10 arrays.

#### Datasets Supported (Evaluation)

1. **MBPP (sanitized)**: test split. References include both test imports and test assertions.
2. **HumanEval**: `openai_humaneval`, test split. Uses `prefix_fn` to prepend the prompt to the completion (function-completion format). References append `check(entry_point)` to the canonical test.

#### Statistical Methodology

- **pass@k computation**: Correct. Uses the standard unbiased estimator via `code_eval.compute()`. The manual per-task computation in lines 46-51 (`per_task_pass1 = c/n`) is the **biased** per-task rate used only as input to bootstrap CI -- this is the empirical pass rate, not the unbiased estimator. However, for pass@1 with n samples, `c/n` is already unbiased, so this is fine for pass@1. For pass@10, the code correctly uses the combinatorial formula (line 60-63).
- **Bootstrap CI**: Methodologically sound. Percentile method with 10k resamples is standard. Fixed seed ensures reproducibility.

#### Issues

1. **Stop tokens may truncate valid code**: `_CODE_STOP = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif"]`. Stopping at `\nif` will truncate any solution containing a top-level `if` statement. This is overly aggressive for MBPP/HumanEval where many solutions require conditionals. This likely **depresses pass rates systematically**.
2. **`top_p=0.95` hardcoded** during evaluation sampling. This interacts with the `temperature` parameter. At low temperatures (default 0.2), top_p=0.95 is essentially a no-op, but at higher temperatures it could affect results. Should be documented or configurable.
3. **`max_tokens=512`** is hardcoded. Sufficient for MBPP/HumanEval but could truncate longer APPS solutions if this function is ever reused for APPS evaluation.
4. **pass@10 guard**: Only computed when `n_samples >= 10`. The combinatorial formula `C(n-c, 10)/C(n, 10)` requires `n >= 10`, so this is correct. However, if `n_samples` is exactly 10, the estimator has zero variance (it's deterministic given c), making the bootstrap CI degenerate.
5. **No multiple-comparison correction**: When reporting both MBPP and HumanEval results, no Bonferroni or similar correction is applied to the 95% CIs. This is acceptable for an exploratory proof-of-concept but should be noted.

---

### eval/rl_bench_eval.py

#### Structure

End-to-end RL training + evaluation harness. Orchestrates:
1. Load training data (MBPP or APPS)
2. Build DS-MeZO controller via `build_controller()`
3. Pre-training eval (MBPP + HumanEval)
4. Training loop with periodic evaluation at specified steps
5. Post-training eval (MBPP + HumanEval)
6. JSON results dump

#### Training Data

- Supports `--train-data mbpp` (default) or `--train-data apps`
- Cycles through training problems: `problem = train_data[step_idx % len(train_data)]`

#### Evaluation Protocol

- Pre/post evaluation on both MBPP test and HumanEval test
- Scaling checkpoints: `--eval-at-steps` allows intermediate evaluations
- Calls `backend.sync_adapters()` before each eval to push adapter weights to vLLM

#### Issues

1. **Cyclic data ordering without shuffling**: `train_data[step_idx % len(train_data)]` cycles through problems in dataset order. No shuffling means the model sees problems in the same order every epoch. For MBPP's 374 training problems over 1000 steps, this means ~2.67 epochs with identical ordering. Could introduce order-dependent bias.
2. **`total_steps` override logic is confusing**: If `eval_at_steps` contains a value larger than `--total-steps`, `total_steps` is silently increased (line 54). This means `--total-steps 100 --eval-at-steps 200` trains for 200 steps, which is counterintuitive.
3. **Single-problem batches**: `controller.step([problem["prompt"]])` processes one problem per step. This is inherent to the ZO approach (SPSA perturbation per step) but limits throughput.
4. **No validation set**: Training and evaluation use the same benchmark family (MBPP train / MBPP test). While the splits are different, there's no held-out validation set for hyperparameter tuning vs. final reporting. The `eval_at_steps` checkpoints could be used for model selection, which would invalidate the test set results.
5. **Training log is sparse**: Only records `step`, `eta`, `eps`. Does not log per-step reward, loss, or gradient norm. Makes post-hoc debugging difficult.

---

### eval/grpo_baseline.py

#### GRPO Baseline Setup

Uses TRL's `GRPOTrainer` for a first-order backpropagation baseline comparison against DS-MeZO.

#### Configuration

- `GRPOConfig` with vLLM in colocate mode (`vllm_mode="colocate"`)
- `vllm_gpu_memory_utilization=0.3` (conservative, leaves room for model + optimizer)
- `num_generations=4`, `per_device_train_batch_size=4`
- Gradient checkpointing enabled, BF16
- `max_prompt_length=512`, `max_completion_length=512`
- No saving (`save_strategy="no"`)

#### Reward Function

- `mbpp_exec_reward()`: Thin wrapper calling `_score_code_solution` per completion. Matches TRL's expected signature `(completions, **kwargs) -> List[float]`.

#### Evaluation

- Pre-training eval: MBPP only (no HumanEval, unlike `rl_bench_eval.py`)
- Post-training eval: MBPP only
- Uses vLLM engine for eval (separate from training engine)

#### Comparison Table

Loads DS-MeZO results from `--dsmezo-results` JSON file and prints a side-by-side comparison table with pass@1, delta, time, and peak VRAM.

#### Issues

1. **Asymmetric evaluation**: GRPO baseline evaluates only on MBPP, while DS-MeZO evaluates on both MBPP and HumanEval. This makes the comparison incomplete. HumanEval results are missing from the GRPO side of the comparison table.
2. **DS-MeZO VRAM is hardcoded as `~17 GB`** in the comparison table (line 182), not measured. This undermines the resource comparison claim. Should either measure it or clearly label it as an estimate.
3. **Training data mismatch potential**: GRPO always trains on MBPP (`load_mbpp_train()`), but DS-MeZO can train on either MBPP or APPS. If DS-MeZO was trained on APPS and the GRPO baseline on MBPP, the comparison is invalid.
4. **Different training dynamics**: GRPO uses `num_generations=4` (4 samples per prompt for RLOO/GRPO advantage estimation) while DS-MeZO uses 2-perturbation SPSA. The effective compute per step differs substantially, making wall-clock time comparisons meaningful but step-count comparisons misleading.
5. **No seed control for DS-MeZO**: GRPO sets `seed=42`, but the DS-MeZO run's randomness is not controlled from this script (it reads results from a file). Reproducibility depends on the DS-MeZO run being seeded separately.
6. **`PeftModel.from_pretrained`** loads the PiSSA adapter. After GRPO training, the adapter is saved and re-evaluated via vLLM. This is correct but requires the saved adapter to be compatible with vLLM's LoRA loading (PiSSA adapters stored in LoRA format should work).
7. **Memory tracking is coarse**: `MemoryCallback` samples VRAM only at `on_step_end`. Peak allocation during forward/backward within a step is missed. `torch.cuda.max_memory_allocated()` would be more accurate.

---

## 2. Cross-Cutting Findings

### Correctness

| Issue | Severity | File |
|-------|----------|------|
| APPS test construction assumes `solution()` function | **High** | rewards.py |
| `\nif` stop token truncates valid solutions | **High** | benchmarks.py |
| Cyclic data without shuffling | **Medium** | rl_bench_eval.py |
| Asymmetric eval (MBPP-only for GRPO) | **Medium** | grpo_baseline.py |
| Hardcoded VRAM in comparison | **Low** | grpo_baseline.py |
| No sandboxing beyond HF code_eval | **Low** | rewards.py |

### Statistical Methodology

- **pass@k**: Correctly uses Chen et al. unbiased estimator via HuggingFace `code_eval`.
- **Bootstrap CI**: Sound (10k percentile bootstrap, seeded). Applied per-benchmark, no multiple comparison correction.
- **Missing**: No effect size reporting (Cohen's d), no paired test between pre/post, no comparison significance test between DS-MeZO and GRPO.

### Datasets

| Dataset | Split | Usage | File |
|---------|-------|-------|------|
| MBPP (sanitized) | train | RL training | rewards.py |
| MBPP (sanitized) | test | Evaluation | benchmarks.py |
| APPS (introductory) | train | RL training (optional) | rewards.py |
| HumanEval | test | Evaluation (DS-MeZO only) | benchmarks.py |

### Sandboxing

All code execution uses HuggingFace `evaluate`'s `code_eval` metric, which spawns a subprocess with a timeout. No container, namespace, or seccomp isolation. The `HF_ALLOW_CODE_EVAL=1` in `__init__.py` is required to enable this. Risk is mitigated by the fact that executed code comes from a fine-tuned code model (not adversarial input), but remains a concern for production use.

### GRPO Baseline Comparison

The comparison is structurally reasonable but has several confounds:
1. Asymmetric evaluation scope (MBPP-only vs MBPP+HumanEval)
2. Hardcoded vs measured VRAM
3. Different effective compute per step (4 generations vs 2 perturbations)
4. Potential training data mismatch (MBPP vs APPS)

For a rigorous comparison, both methods should be evaluated on the same benchmarks, with the same training data, and with compute-matched (not step-matched) training budgets.
