"""Monkey patches for loading AutoRound NVFP4 checkpoints with vLLM INC."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from vllm.logger import init_logger

from .inc_nvfp4_scheme import INCNvfp4Scheme
from .inc_nvfp4_ue5m3_scheme import INCNvfp4UE5M3Scheme

logger = init_logger(__name__)
_PATCHED = False


def _layer_data_type(config: Any, layer_name: str, default: str) -> str:
    matches = [
        (key, value)
        for key, value in (config.extra_config or {}).items()
        if isinstance(key, str) and isinstance(value, dict) and (layer_name == key or layer_name.endswith(f".{key}"))
    ]
    if not matches:
        return default
    _, layer_config = max(matches, key=lambda item: len(item[0]))
    return layer_config.get("data_type", default)


def apply_patches() -> None:
    """Register NVFP4 metadata and scheme support in the current vLLM process."""
    global _PATCHED
    if _PATCHED:
        return

    from vllm.model_executor.layers.quantization.inc import INCConfig
    from vllm.model_executor.layers.quantization.inc import inc as inc_module
    from vllm.model_executor.layers.quantization.inc.config_parser import INCConfigParser
    from vllm.model_executor.layers.quantization.inc.schemes import factory

    INCConfig.SUPPORTED_DTYPES = set(INCConfig.SUPPORTED_DTYPES) | {"nv_fp", "nvfp4_v2"}
    INCConfig.SUPPORTED_FORMATS = set(INCConfig.SUPPORTED_FORMATS) | {
        "auto_round:llm_compressor",
        "auto_round:llm_compressor_nvfp4_e5m3",
    }

    original_resolve_scheme = factory.resolve_scheme
    original_resolve_config = INCConfigParser.resolve

    def resolve_config(self: Any, layer: Any, layer_name: str):
        layer_config = original_resolve_config(self, layer, layer_name)
        data_type = _layer_data_type(self._config, layer_name, layer_config.data_type)
        return replace(layer_config, data_type=data_type)

    def resolve_scheme(layer_config: Any):
        if INCNvfp4UE5M3Scheme.can_handle(layer_config):
            return INCNvfp4UE5M3Scheme()
        if INCNvfp4Scheme.can_handle(layer_config):
            return INCNvfp4Scheme()
        return original_resolve_scheme(layer_config)

    INCConfigParser.resolve = resolve_config
    factory.resolve_scheme = resolve_scheme
    inc_module.resolve_scheme = resolve_scheme
    _PATCHED = True
    logger.warning("vLLM NVFP4 patch applied: AutoRound nv_fp and nvfp4_v2 schemes registered")


def register() -> None:
    """Entry point used by vLLM's general plugin loader."""
    apply_patches()


__all__ = ["apply_patches", "register"]
