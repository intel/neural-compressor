"""Monkey patches for loading AutoRound NVFP4 checkpoints with vLLM INC."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from vllm.logger import init_logger

from .inc_nvfp4_scheme import INCNvfp4Scheme

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

    INCConfig.SUPPORTED_DTYPES = set(INCConfig.SUPPORTED_DTYPES) | {"nv_fp"}
    INCConfig.SUPPORTED_FORMATS = set(INCConfig.SUPPORTED_FORMATS) | {"auto_round:llm_compressor"}

    original_resolve_scheme = factory.resolve_scheme
    original_resolve_config = INCConfigParser.resolve
    original_validate_supported_quantization = INCConfig._validate_supported_quantization

    def validate_supported_quantization(self: Any) -> None:
        if self.data_type == "nv_fp" and self.weight_bits == 4:
            if self.packing_format not in INCConfig.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported packing_format: {self.packing_format}")
            return
        original_validate_supported_quantization(self)

    def resolve_config(self: Any, layer: Any, layer_name: str):
        layer_config = original_resolve_config(self, layer, layer_name)
        data_type = _layer_data_type(self._config, layer_name, layer_config.data_type)
        return replace(layer_config, data_type=data_type)

    def resolve_scheme(layer_config: Any):
        qdq_owns_nvfp4 = getattr(INCConfig, "_vllm_qdq_nvfp4_enabled", False)
        if INCNvfp4Scheme.can_handle(layer_config) and not qdq_owns_nvfp4:
            return INCNvfp4Scheme()
        return original_resolve_scheme(layer_config)

    INCConfigParser.resolve = resolve_config
    INCConfig._validate_supported_quantization = validate_supported_quantization
    factory.resolve_scheme = resolve_scheme
    inc_module.resolve_scheme = resolve_scheme

    from vllm_qdq_plugin.quantization.patch import apply_patches as apply_quantization_patches

    apply_quantization_patches(enable_nvfp4_qdq=False)
    _PATCHED = True
    logger.warning("vLLM NVFP4 hardware patch applied: AutoRound nv_fp scheme registered")


def register() -> None:
    """Entry point used by vLLM's general plugin loader."""
    from vllm_qdq_plugin import envs

    if envs.VLLM_QDQ:
        logger.info("Skipping NVFP4 hardware registration because VLLM_QDQ=1")
        return
    apply_patches()


__all__ = ["apply_patches", "register"]
