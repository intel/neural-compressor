"""
sage3 — Composable SageAttention3 Triton implementation.

Refactored from the monolithic sageattention3_standalone.py into focused modules
with a composable QuantConfig + zero-dispatch kernel architecture.

Public API:
    sageattn3_torch_triton_standalone()  — backward-compatible with monolith
    sageattn3_standalone()               — new config-based API
    scaled_dot_product_attention()       — SDPA-compatible wrapper

Configuration:
    ATTENTION_CONFIGS                    — registry of all attention configs
    AttentionConfig, QuantConfig         — config dataclasses
"""

from .api import (
    sageattn3_torch_triton_standalone,
    sageattn3_standalone,
    scaled_dot_product_attention,
)
from .quant_config import (
    ATTENTION_CONFIGS,
    AttentionConfig,
    QuantConfig,
    DataFormat,
    ScaleFormat,
    RoundingMode,
)
from .p_quant_registry import P_QUANT_REGISTRY
from .layer_adaptive import LayerAdaptiveAttention, StepLayerAdaptiveAttention

__all__ = [
    # Entry points
    "sageattn3_torch_triton_standalone",
    "sageattn3_standalone",
    "scaled_dot_product_attention",
    # Configuration
    "ATTENTION_CONFIGS",
    "AttentionConfig",
    "QuantConfig",
    "DataFormat",
    "ScaleFormat",
    "RoundingMode",
    "P_QUANT_REGISTRY",
    "LayerAdaptiveAttention",
    "StepLayerAdaptiveAttention",
]
