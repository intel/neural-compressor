"""Monkey patches for loading AutoRound NVFP4 checkpoints with vLLM INC."""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from .inc_nvfp4_scheme import INCNvfp4Scheme

logger = init_logger(__name__)
_PATCHED = False


def apply_patches() -> None:
    """Register NVFP4 metadata and scheme support in the current vLLM process."""
    global _PATCHED
    if _PATCHED:
        return

    from vllm.model_executor.layers.quantization.inc import INCConfig
    from vllm.model_executor.layers.quantization.inc.schemes import factory

    INCConfig.SUPPORTED_DTYPES = set(INCConfig.SUPPORTED_DTYPES) | {"nv_fp"}
    INCConfig.SUPPORTED_FORMATS = set(INCConfig.SUPPORTED_FORMATS) | {
        "auto_round:llm_compressor"
    }

    original_resolve_scheme = factory.resolve_scheme

    def resolve_scheme(layer_config: Any):
        if INCNvfp4Scheme.can_handle(layer_config):
            return INCNvfp4Scheme()
        return original_resolve_scheme(layer_config)

    factory.resolve_scheme = resolve_scheme
    _PATCHED = True
    logger.warning("vLLM NVFP4 patch applied: AutoRound nv_fp scheme registered")


def register() -> None:
    """Entry point used by vLLM's general plugin loader."""
    apply_patches()


__all__ = ["apply_patches", "register"]
