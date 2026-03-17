from pathlib import Path

from ds_mezo.controller import DSMeZO_Controller
from ds_mezo.backend import VLLMBackend, create_engine
from ds_mezo.model_config import LayerSpec, discover_layers, load_adapter_config, svd_power_iters


def build_controller(
    model_path=None,
    adapter_path=None,
    output_dir=None,
    total_steps=None,
    score_fn=None,
    calibration_prompt=None,
    engine=None,
    layer_specs=None,
    rank=None,
    extra_config=None,
):
    if rank is None:
        rank, target_modules = load_adapter_config(adapter_path)
    else:
        _, target_modules = load_adapter_config(adapter_path)
    if engine is None:
        engine = create_engine(model_path, rank)
    if layer_specs is None:
        layer_specs = discover_layers(model_path, target_modules)
    backend = VLLMBackend(engine, layer_specs, rank)
    config = {
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
