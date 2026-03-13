"""DS-MeZO: Zeroth-order LLM fine-tuning via vLLM + Triton."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ds_mezo.controller import DSMeZO_Controller
from ds_mezo.backend import VLLMBackend, create_engine
from ds_mezo.model_config import LayerSpec, discover_layers, load_adapter_config, svd_power_iters


def build_controller(
    model_path: Path | str | None,
    adapter_path: Path | str,
    output_dir: Path | str,
    total_steps: int,
    score_fn: Callable | None = None,
    calibration_prompt: str | None = None,
    engine: Any = None,
    layer_specs: list | None = None,
    rank: int | None = None,
    extra_config: dict[str, Any] | None = None,
) -> tuple[Any, VLLMBackend, DSMeZO_Controller, int, list]:
    """Build vLLM engine (or reuse), create backend + controller, calibrate.
    Returns (engine, backend, controller, rank, layer_specs)."""
    if rank is None:
        rank, target_modules = load_adapter_config(adapter_path)
    else:
        _, target_modules = load_adapter_config(adapter_path)
    if engine is None:
        engine = create_engine(model_path, rank)
    if layer_specs is None:
        layer_specs = discover_layers(model_path, target_modules)
    backend = VLLMBackend(engine, layer_specs, rank)
    config: dict[str, Any] = {
        "output_dir": str(output_dir),
        "adapter_path": str(adapter_path),
        "total_steps": total_steps,
    }
    if score_fn:
        config["score_fn"] = score_fn
    if extra_config:
        config.update(extra_config)
    controller = DSMeZO_Controller(backend, layer_specs, config)
    if calibration_prompt:
        controller._calibrate_activation_bases_full([calibration_prompt])
    return engine, backend, controller, rank, layer_specs


__all__ = [
    "DSMeZO_Controller",
    "VLLMBackend",
    "build_controller",
    "create_engine",
    "LayerSpec",
    "discover_layers",
    "load_adapter_config",
    "svd_power_iters",
]
