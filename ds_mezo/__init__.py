"""DS-MeZO: Zeroth-order LLM fine-tuning via vLLM + Triton."""

from ds_mezo.controller import DSMeZO_Controller, DSMeZOConfig
from ds_mezo.backend import VLLMBackend
from ds_mezo.model_config import LayerSpec, discover_layers

__all__ = [
    "DSMeZO_Controller",
    "DSMeZOConfig",
    "VLLMBackend",
    "LayerSpec",
    "discover_layers",
]
