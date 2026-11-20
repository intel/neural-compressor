"""Register AutoRound NVFP4 QDQ schemes with vLLM INC."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from nvfp4_hw.inc_nvfp4_scheme import INCNvfp4Scheme
from vllm.logger import init_logger

from .inc_nvfp4_e5m3_scheme import NVFP4_E5M3_DATA_TYPE, NVFP4_E5M3_GS_DATA_TYPE, INCNvfp4UE5M3Scheme
from .inc_nvfp4_scheme import INCNvfp4QDQScheme

logger = init_logger(__name__)
_PATCHED = False
_COMPRESSED_TENSORS_PATCHED = False


def _native_nvfp4_kernel_available() -> bool:
    """Return whether vLLM selects an accelerated W4A4 kernel for this device."""
    try:
        from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel

        kernel = init_nvfp4_linear_kernel()
    except Exception as error:
        logger.info_once("Native NVFP4 kernel unavailable; using QDQ: %s", error)
        return False

    kernel_name = type(kernel).__name__
    return kernel_name not in {"MarlinNvFp4LinearKernel", "EmulationNvFp4LinearKernel"} and "W4A16" not in kernel_name


def _matching_layer_config(config: Any, layer_name: str) -> dict[str, Any] | None:
    regex_special_chars = set(r"*+?^$()[]{}|\\")

    def normalized_parts(path: str) -> list[str]:
        parts = path.strip(".").split(".")
        return parts[1:] if parts and parts[0] == "model" else parts

    layer_parts = normalized_parts(layer_name)

    def matches_layer(key: str) -> bool:
        if any(char in regex_special_chars for char in key):
            try:
                normalized_name = ".".join(layer_parts)
                return re.search(key, layer_name) is not None or re.search(key, normalized_name) is not None
            except re.error:
                return False

        key_parts = normalized_parts(key)
        if not key_parts or len(key_parts) > len(layer_parts):
            return False
        return any(
            layer_parts[index : index + len(key_parts)] == key_parts
            for index in range(len(layer_parts) - len(key_parts) + 1)
        )

    matches = [
        (key, value)
        for key, value in (config.extra_config or {}).items()
        if isinstance(key, str) and isinstance(value, dict) and matches_layer(key)
    ]
    if not matches:
        return None
    exact_matches = [
        (key, value)
        for key, value in matches
        if not any(char in regex_special_chars for char in key) and normalized_parts(key) == layer_parts
    ]
    return max(exact_matches or matches, key=lambda item: len(item[0]))[1]


def _resolve_layer_config_override(config: Any, layer_name: str, layer_config: Any) -> Any:
    override = _matching_layer_config(config, layer_name)
    if override is None:
        return layer_config
    updates = {
        key: override[key]
        for key in ("bits", "group_size", "sym", "data_type")
        if key in override and override[key] is not None
    }
    if isinstance(updates.get("group_size"), list):
        updates["group_size"] = tuple(updates["group_size"])
    if "bits" in updates:
        updates["quantized"] = updates["bits"] < 16
    return replace(layer_config, **updates)


def _layer_data_type(config: Any, layer_name: str, default: str) -> str:
    override = _matching_layer_config(config, layer_name)
    if override is None:
        return default
    return override.get("data_type", default)


def _patch_compressed_tensors_nvfp4() -> None:
    """Use activation QDQ on compressed-tensors NVFP4 Marlin fallback paths."""
    global _COMPRESSED_TENSORS_PATCHED
    if _COMPRESSED_TENSORS_PATCHED:
        return

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        CompressedTensorsMoEMethod,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import (
        CompressedTensorsW4A4Fp4,
    )
    from vllm.model_executor.kernels.linear.nvfp4.marlin import MarlinNvFp4LinearKernel

    original_linear_apply = CompressedTensorsW4A4Fp4.apply_weights
    original_moe_method = CompressedTensorsMoEMethod.get_moe_method

    def apply_nvfp4_qdq(self: Any, layer: Any, x: Any, bias: Any = None):
        enabled = getattr(CompressedTensorsConfig, "_vllm_qdq_nvfp4_enabled", False)
        if enabled and isinstance(self.kernel, MarlinNvFp4LinearKernel):
            from vllm_qdq_plugin.qdq.nvfp4 import nvfp4_qdq

            # vLLM keeps the checkpoint quantization scale here; input_global_scale holds its reciprocal.
            input_scale = getattr(layer, "input_global_scale_inv", None)
            if input_scale is not None:
                original_shape = x.shape
                x = nvfp4_qdq(
                    x.reshape(-1, x.shape[-1]).contiguous(),
                    input_scale,
                    group_size=16,
                    trace_op_name="compressed_tensors_nvfp4_linear",
                ).reshape(original_shape)
        return original_linear_apply(self, layer, x, bias)

    def get_nvfp4_qdq_moe_method(quant_config: Any, layer: Any, layer_name: str):
        enabled = getattr(CompressedTensorsConfig, "_vllm_qdq_nvfp4_enabled", False)
        if enabled:
            quant_config._add_fused_moe_to_target_scheme_map()
            projection_names = [
                layer_name + suffix
                for suffix in (".0.gate_proj", ".0.up_proj", ".0.down_proj")
            ]
            scheme_dicts = [quant_config.get_scheme_dict(layer, name) for name in projection_names]
            scheme_dict = scheme_dicts[-1]
            if scheme_dict is not None and all(item == scheme_dict for item in scheme_dicts[:-1]):
                weight_quant = scheme_dict.get("weights")
                input_quant = scheme_dict.get("input_activations")
                if quant_config._is_nvfp4_format(weight_quant) and quant_config._is_nvfp4_format(input_quant):
                    from .inc_nvfp4_moe import INCNvfp4QDQMoEMethod

                    return INCNvfp4QDQMoEMethod(layer.moe_config, weight_quant.group_size)
        return original_moe_method(quant_config, layer, layer_name)

    CompressedTensorsW4A4Fp4.apply_weights = apply_nvfp4_qdq
    CompressedTensorsMoEMethod.get_moe_method = staticmethod(get_nvfp4_qdq_moe_method)
    _COMPRESSED_TENSORS_PATCHED = True


def apply_patches(enable_nvfp4_qdq: bool = True) -> None:
    """Register AutoRound NVFP4 QDQ metadata and scheme routing."""
    global _PATCHED
    from vllm.model_executor.layers.quantization.inc import INCConfig

    INCConfig._vllm_qdq_nvfp4_enabled = enable_nvfp4_qdq
    _patch_compressed_tensors_nvfp4()
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    CompressedTensorsConfig._vllm_qdq_nvfp4_enabled = enable_nvfp4_qdq
    if _PATCHED:
        return

    from vllm.model_executor.layers.quantization.inc import inc as inc_module
    from vllm.model_executor.layers.quantization.inc.config_parser import INCConfigParser
    from vllm.model_executor.layers.quantization.inc.schemes import factory

    supported_dtypes = {NVFP4_E5M3_DATA_TYPE, NVFP4_E5M3_GS_DATA_TYPE, "nv_fp"}
    INCConfig.SUPPORTED_DTYPES = set(INCConfig.SUPPORTED_DTYPES) | supported_dtypes
    INCConfig.SUPPORTED_FORMATS = set(INCConfig.SUPPORTED_FORMATS) | {
        "auto_round:llm_compressor",
        "auto_round:llm_compressor_nvfp4_e5m3",
        "auto_round:llm_compressor_nvfp4_e5m3_gs",
    }

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
        return _resolve_layer_config_override(self._config, layer_name, layer_config)

    def resolve_scheme(layer_config: Any):
        if INCNvfp4UE5M3Scheme.can_handle(layer_config):
            return INCNvfp4UE5M3Scheme()
        if INCNvfp4QDQScheme.can_handle(layer_config):
            force_qdq = getattr(INCConfig, "_vllm_qdq_nvfp4_enabled", False)
            if not force_qdq and _native_nvfp4_kernel_available():
                return INCNvfp4Scheme()
            return INCNvfp4QDQScheme()
        return original_resolve_scheme(layer_config)

    INCConfigParser.resolve = resolve_config
    INCConfig._validate_supported_quantization = validate_supported_quantization
    factory.resolve_scheme = resolve_scheme
    inc_module.resolve_scheme = resolve_scheme
    _PATCHED = True
    formats = "nv_fp, nvfp4_v2, and nvfp4_v2_gs"
    logger.warning("vLLM QDQ patch applied: AutoRound %s schemes registered", formats)


def register() -> None:
    """Entry point used by vLLM's general plugin loader."""
    from vllm_qdq_plugin import envs

    apply_patches(enable_nvfp4_qdq=envs.VLLM_QDQ)


__all__ = ["apply_patches", "register"]
