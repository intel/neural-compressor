"""Register AutoRound NVFP4 QDQ schemes with vLLM INC."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from vllm.logger import init_logger

from .inc_nvfp4_e5m3_scheme import INCNvfp4UE5M3Scheme
from .inc_nvfp4_scheme import INCNvfp4QDQScheme

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


def apply_patches(enable_nvfp4_qdq: bool = True) -> None:
    """Register AutoRound NVFP4 QDQ metadata and scheme routing."""
    global _PATCHED
    if _PATCHED:
        return

    from vllm.model_executor.layers.quantization.inc import INCConfig
    from vllm.model_executor.layers.quantization.inc import inc as inc_module
    from vllm.model_executor.layers.quantization.inc.config_parser import INCConfigParser
    from vllm.model_executor.layers.quantization.inc.schemes import factory

    supported_dtypes = {"nvfp4_v2"}
    if enable_nvfp4_qdq:
        supported_dtypes.add("nv_fp")
        INCConfig._vllm_qdq_nvfp4_enabled = True
    INCConfig.SUPPORTED_DTYPES = set(INCConfig.SUPPORTED_DTYPES) | supported_dtypes
    INCConfig.SUPPORTED_FORMATS = set(INCConfig.SUPPORTED_FORMATS) | {
        "auto_round:llm_compressor",
        "auto_round:llm_compressor_nvfp4_e5m3",
    }

    original_resolve_scheme = factory.resolve_scheme
    original_resolve_config = INCConfigParser.resolve
    original_validate_supported_quantization = INCConfig._validate_supported_quantization

    def validate_supported_quantization(self: Any) -> None:
        if enable_nvfp4_qdq and self.data_type == "nv_fp" and self.weight_bits == 4:
            if self.packing_format not in INCConfig.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported packing_format: {self.packing_format}")
            return
        original_validate_supported_quantization(self)

    def resolve_config(self: Any, layer: Any, layer_name: str):
        layer_config = original_resolve_config(self, layer, layer_name)
        data_type = _layer_data_type(self._config, layer_name, layer_config.data_type)
        return replace(layer_config, data_type=data_type)

    def resolve_scheme(layer_config: Any):
        if INCNvfp4UE5M3Scheme.can_handle(layer_config):
            return INCNvfp4UE5M3Scheme()
        if enable_nvfp4_qdq and INCNvfp4QDQScheme.can_handle(layer_config):
            return INCNvfp4QDQScheme()
        return original_resolve_scheme(layer_config)

    INCConfigParser.resolve = resolve_config
    INCConfig._validate_supported_quantization = validate_supported_quantization
    factory.resolve_scheme = resolve_scheme
    inc_module.resolve_scheme = resolve_scheme
    _PATCHED = True
    formats = "nv_fp and nvfp4_v2" if enable_nvfp4_qdq else "nvfp4_v2"
    logger.warning("vLLM QDQ patch applied: AutoRound %s schemes registered", formats)


__all__ = ["apply_patches"]
