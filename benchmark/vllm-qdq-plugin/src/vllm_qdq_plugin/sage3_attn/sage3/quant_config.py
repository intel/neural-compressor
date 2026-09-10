"""
Quantization configuration: enums, QuantConfig, AttentionConfig, and the ATTENTION_CONFIGS registry.

This is the central data model for the composable architecture. Each quantization
scheme is fully described by an AttentionConfig that composes:
- QK quantization config (host-side Q/K preprocessing)
- PV quantization config (host-side V preprocessing + P-quant format)
- P-quant function (in-kernel, passed as constexpr)
- Pre-transforms pipeline (smoothing, rotation, etc.)
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import triton

if TYPE_CHECKING:
    from .transforms import TransformFn

from .quant_primitives import (
    apply_e2m1_quantization_torch,
    apply_e4m3_quantization_torch,
    round_to_e4m3_torch,
    round_to_e8m0_torch,
)

# ============================================================================
# Enums
# ============================================================================


class DataFormat(IntEnum):
    """Data quantization format."""

    E2M1 = 0  # FP4 (±{0, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6})
    E4M3 = 1  # FP8 ([-448, 448])
    E5M2 = 2  # FP8 alternate
    INT8 = 3  # 8-bit integer
    INT4 = 4  # 4-bit integer


class ScaleFormat(IntEnum):
    """Scale rounding format."""

    E4M3 = 0  # NVFP4 style (3-bit mantissa)
    E8M0 = 1  # MXFP4/MXFP8 style (power-of-2 shared exponent)
    NONE = 2  # No scale rounding (INT schemes, standalone FP8)


class RoundingMode(IntEnum):
    """Rounding mode for scale and/or data quantization."""

    STOCHASTIC = 0
    RNE = 1  # Round-to-nearest-even
    CEIL = 2
    FLOOR = 3


# ============================================================================
# Enum → Torch callable dispatch tables
# ============================================================================

_SCALE_FN_BY_FORMAT = {
    ScaleFormat.E4M3: round_to_e4m3_torch,
    ScaleFormat.E8M0: round_to_e8m0_torch,
    ScaleFormat.NONE: lambda x: x,
}

_DATA_FN_BY_FORMAT = {
    DataFormat.E2M1: apply_e2m1_quantization_torch,
    DataFormat.E4M3: apply_e4m3_quantization_torch,
}


# ============================================================================
# QuantConfig — per-stage quantization configuration
# ============================================================================


@dataclass(frozen=True)
class QuantConfig:
    """Configuration for one quantization stage (QK or PV).

    Carries both the abstract format description (enums) and the concrete
    callable implementations needed by host-side quantization (quantize.py).
    """

    name: str
    data_format: DataFormat
    scale_format: ScaleFormat
    rounding_mode: RoundingMode = RoundingMode.STOCHASTIC
    block_size: int = 16
    scale_levels: int = 2  # 1 = single-level, 2 = two-level
    fp_max: float = 6.0  # Max representable value for the data format

    # Torch callables for host-side quantization (used by quantize.py)
    round_scale_torch: Optional[Callable] = None  # e.g., round_to_e4m3_torch
    quant_data_torch: Optional[Callable] = None  # e.g., apply_e2m1_quantization_torch; None → E2M1 default


# ============================================================================
# AttentionConfig — full attention configuration
# ============================================================================


@dataclass(frozen=True)
class AttentionConfig:
    """Composes QK quant, PV quant, P-quant kernel, and pre-transforms.

    This is the single object that fully describes an attention quantization scheme.
    Frozen to prevent accidental mutation of shared configs in the global registry.
    """

    name: str
    qk_quant: QuantConfig  # Host-side Q/K quantization config
    pv_quant: QuantConfig  # Host-side V quantization config
    p_quant_fn: triton.JITFunction  # @triton.jit function for in-kernel P quantization
    pre_transforms: Tuple["TransformFn", ...] = field(default_factory=tuple)

    def validate(self):
        """Validate config consistency at registration time."""
        # Required fields
        assert self.qk_quant.round_scale_torch is not None, f"{self.name}: qk_quant missing round_scale_torch"
        assert self.pv_quant.round_scale_torch is not None, f"{self.name}: pv_quant missing round_scale_torch"
        assert self.p_quant_fn is not None, f"{self.name}: missing p_quant_fn"

        # Validate enum/callable consistency for QK
        expected_scale_fn = _SCALE_FN_BY_FORMAT.get(self.qk_quant.scale_format)
        if expected_scale_fn is not None:
            assert self.qk_quant.round_scale_torch is expected_scale_fn, (
                f"{self.name}: qk_quant.scale_format={self.qk_quant.scale_format.name} "
                f"but round_scale_torch is {self.qk_quant.round_scale_torch.__name__}, "
                f"expected {expected_scale_fn.__name__}"
            )

        # Validate enum/callable consistency for PV
        expected_scale_fn = _SCALE_FN_BY_FORMAT.get(self.pv_quant.scale_format)
        if expected_scale_fn is not None:
            assert self.pv_quant.round_scale_torch is expected_scale_fn, (
                f"{self.name}: pv_quant.scale_format={self.pv_quant.scale_format.name} "
                f"but round_scale_torch is {self.pv_quant.round_scale_torch.__name__}, "
                f"expected {expected_scale_fn.__name__}"
            )


# ============================================================================
# ATTENTION_CONFIGS registry
# ============================================================================
# Populated after imports to avoid circular dependencies.
# p_quant_kernels.py registers its functions into P_QUANT_REGISTRY on import.

ATTENTION_CONFIGS: dict[str, AttentionConfig] = {}


def _register_builtin_configs():
    """Register all built-in attention configurations."""
    # Ensure P-quant kernels are registered (import triggers @register_p_quant)
    from . import p_quant_kernels as _  # noqa: F401
    from .p_quant_registry import P_QUANT_REGISTRY
    from .transforms import qk_smoothing, v_smoothing

    # ── NVFP4: Two-level, E4M3 scales, block_size=16 ──
    nvfp4_quant = QuantConfig(
        name="nvfp4",
        data_format=DataFormat.E2M1,
        scale_format=ScaleFormat.E4M3,
        block_size=16,
        scale_levels=2,
        fp_max=6.0,
        round_scale_torch=round_to_e4m3_torch,
        quant_data_torch=None,  # Default E2M1
    )
    ATTENTION_CONFIGS["nvfp4"] = AttentionConfig(
        name="nvfp4",
        qk_quant=nvfp4_quant,
        pv_quant=nvfp4_quant,
        p_quant_fn=P_QUANT_REGISTRY["nvfp4"],
        pre_transforms=(qk_smoothing,),
    )

    # ── MXFP4: Two-level, E8M0 scales, block_size=32 ──
    mxfp4_quant = QuantConfig(
        name="mxfp4",
        data_format=DataFormat.E2M1,
        scale_format=ScaleFormat.E8M0,
        block_size=32,
        scale_levels=2,
        fp_max=6.0,
        round_scale_torch=round_to_e8m0_torch,
        quant_data_torch=None,  # Default E2M1
    )
    ATTENTION_CONFIGS["mxfp4"] = AttentionConfig(
        name="mxfp4",
        qk_quant=mxfp4_quant,
        pv_quant=mxfp4_quant,
        p_quant_fn=P_QUANT_REGISTRY["mxfp4"],
        pre_transforms=(qk_smoothing,),
    )

    # ── MXFP4_S1: Single-level, E8M0 scales, block_size=32 ──
    mxfp4_s1_quant = QuantConfig(
        name="mxfp4_s1",
        data_format=DataFormat.E2M1,
        scale_format=ScaleFormat.E8M0,
        block_size=32,
        scale_levels=1,
        fp_max=6.0,
        round_scale_torch=round_to_e8m0_torch,
        quant_data_torch=None,  # Default E2M1
    )
    ATTENTION_CONFIGS["mxfp4_s1"] = AttentionConfig(
        name="mxfp4_s1",
        qk_quant=mxfp4_s1_quant,
        pv_quant=mxfp4_s1_quant,
        p_quant_fn=P_QUANT_REGISTRY["mxfp4_s1"],
        pre_transforms=(qk_smoothing,),
    )

    # ── MXFP8_S1: Single-level, E8M0 scales, FP8 data, block_size=32 ──
    mxfp8_s1_quant = QuantConfig(
        name="mxfp8_s1",
        data_format=DataFormat.E4M3,
        scale_format=ScaleFormat.E8M0,
        block_size=32,
        scale_levels=1,
        fp_max=448.0,
        round_scale_torch=round_to_e8m0_torch,
        quant_data_torch=apply_e4m3_quantization_torch,
    )
    ATTENTION_CONFIGS["mxfp8_s1"] = AttentionConfig(
        name="mxfp8_s1",
        qk_quant=mxfp8_s1_quant,
        pv_quant=mxfp8_s1_quant,
        p_quant_fn=P_QUANT_REGISTRY["mxfp8_s1"],
        pre_transforms=(qk_smoothing,),
    )

    # ── MXFP4_S1 + Two-level P-quant: E8M0 QK/V, but two-level MXFP4 P-quant ──
    ATTENTION_CONFIGS["mxfp4_s1_2level"] = AttentionConfig(
        name="mxfp4_s1_2level",
        qk_quant=mxfp4_s1_quant,
        pv_quant=mxfp4_s1_quant,
        p_quant_fn=P_QUANT_REGISTRY["mxfp4"],  # Two-level MXFP4 P-quant
        pre_transforms=(qk_smoothing,),
    )

    # ── MXFP4_S1 with E4M3 scales: finer scale granularity at block_size=32 ──
    mxfp4_s1_e4m3_quant = QuantConfig(
        name="mxfp4_s1_e4m3",
        data_format=DataFormat.E2M1,
        scale_format=ScaleFormat.E4M3,
        block_size=32,
        scale_levels=1,
        fp_max=6.0,
        round_scale_torch=round_to_e4m3_torch,
        quant_data_torch=None,  # Default E2M1
    )
    ATTENTION_CONFIGS["mxfp4_s1_e4m3"] = AttentionConfig(
        name="mxfp4_s1_e4m3",
        qk_quant=mxfp4_s1_e4m3_quant,
        pv_quant=mxfp4_s1_e4m3_quant,
        p_quant_fn=P_QUANT_REGISTRY["mxfp4_s1_e4m3"],  # E4M3 scale P-quant
        pre_transforms=(qk_smoothing,),
    )

    # ── Mixed precision: FP8-QK / FP4-PV ──
    ATTENTION_CONFIGS["mixed_fp8qk_fp4pv"] = AttentionConfig(
        name="mixed_fp8qk_fp4pv",
        qk_quant=mxfp8_s1_quant,  # FP8 for Q, K
        pv_quant=mxfp4_s1_quant,  # FP4 for V
        p_quant_fn=P_QUANT_REGISTRY["mxfp4_s1"],  # FP4 P-quant
        pre_transforms=(qk_smoothing,),
    )

    # ── Mixed precision: FP4-QK / FP8-PV ──
    ATTENTION_CONFIGS["mixed_fp4qk_fp8pv"] = AttentionConfig(
        name="mixed_fp4qk_fp8pv",
        qk_quant=mxfp4_s1_quant,  # FP4 for Q, K
        pv_quant=mxfp8_s1_quant,  # FP8 for V
        p_quant_fn=P_QUANT_REGISTRY["mxfp8_s1"],  # FP8 P-quant
        pre_transforms=(qk_smoothing,),
    )

    # ── V-smoothed variants: add V mean subtraction before quantization ──

    # MXFP4_S1 + V smoothing: should especially help FP4 V path
    ATTENTION_CONFIGS["mxfp4_s1_vsmooth"] = AttentionConfig(
        name="mxfp4_s1_vsmooth",
        qk_quant=mxfp4_s1_quant,
        pv_quant=mxfp4_s1_quant,
        p_quant_fn=P_QUANT_REGISTRY["mxfp4_s1"],
        pre_transforms=(qk_smoothing, v_smoothing),
    )

    # FP8-QK / FP4-PV + V smoothing: V is FP4, so smoothing helps most here
    ATTENTION_CONFIGS["mixed_fp8qk_fp4pv_vs"] = AttentionConfig(
        name="mixed_fp8qk_fp4pv_vs",
        qk_quant=mxfp8_s1_quant,
        pv_quant=mxfp4_s1_quant,
        p_quant_fn=P_QUANT_REGISTRY["mxfp4_s1"],
        pre_transforms=(qk_smoothing, v_smoothing),
    )

    # FP4-QK / FP8-PV + V smoothing: V is already FP8 but smoothing still helps
    ATTENTION_CONFIGS["mixed_fp4qk_fp8pv_vs"] = AttentionConfig(
        name="mixed_fp4qk_fp8pv_vs",
        qk_quant=mxfp4_s1_quant,
        pv_quant=mxfp8_s1_quant,
        p_quant_fn=P_QUANT_REGISTRY["mxfp8_s1"],
        pre_transforms=(qk_smoothing, v_smoothing),
    )

    # Hardware MXFP8 — uses tl.dot_scaled directly, no host-side quantize path
    # Skips validate() since qk_quant/pv_quant/p_quant_fn are None (handled in kernel)
    ATTENTION_CONFIGS["mxfp8_hw"] = AttentionConfig(
        name="mxfp8_hw",
        qk_quant=None,
        pv_quant=None,
        p_quant_fn=None,
        pre_transforms=(),
    )

    # Hardware MXFP4 — uses tl.dot_scaled e2m1 directly, no host-side quantize path
    ATTENTION_CONFIGS["mxfp4_hw"] = AttentionConfig(
        name="mxfp4_hw",
        qk_quant=None,
        pv_quant=None,
        p_quant_fn=None,
        pre_transforms=(),
    )

    # Mixed MXFP8 QK + MXFP4 PV — uses tl.dot_scaled with e4m3 for QK and e2m1 for PV
    ATTENTION_CONFIGS["mixed_mxfp8qk_mxfp4pv_hw"] = AttentionConfig(
        name="mixed_mxfp8qk_mxfp4pv_hw",
        qk_quant=None,
        pv_quant=None,
        p_quant_fn=None,
        pre_transforms=(),
    )

    # Mixed MXFP4 QK + MXFP8 PV — uses tl.dot_scaled with e2m1 for QK and e4m3 for PV
    ATTENTION_CONFIGS["mixed_mxfp4qk_mxfp8pv_hw"] = AttentionConfig(
        name="mixed_mxfp4qk_mxfp8pv_hw",
        qk_quant=None,
        pv_quant=None,
        p_quant_fn=None,
        pre_transforms=(),
    )

    # Validate all configs at registration time (skip hardware MX configs)
    for config in ATTENTION_CONFIGS.values():
        if config.name in ("mxfp8_hw", "mxfp4_hw", "mixed_mxfp8qk_mxfp4pv_hw", "mixed_mxfp4qk_mxfp8pv_hw"):
            continue
        config.validate()


# Auto-register on module import
_register_builtin_configs()
