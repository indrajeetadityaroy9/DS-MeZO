from ds_mezo.config import Config
from ds_mezo.controller import DSMeZO_Controller
from ds_mezo.backend import VLLMBackend, create_engine
from ds_mezo.model_config import LayerSpec, discover_layers, load_adapter_config, svd_power_iters


def build_controller(cfg, score_fn):
    rank, target_modules = load_adapter_config(cfg.model.adapter_path)
    engine = create_engine(cfg.model.path, rank)
    layer_specs = discover_layers(cfg.model.path, target_modules)
    backend = VLLMBackend(engine, layer_specs, rank, cfg.system.staging_dir)
    controller = DSMeZO_Controller(backend, layer_specs, cfg, score_fn)
    return controller, engine


__all__ = [
    "Config",
    "DSMeZO_Controller",
    "VLLMBackend",
    "build_controller",
    "create_engine",
    "LayerSpec",
    "discover_layers",
    "load_adapter_config",
    "svd_power_iters",
]
