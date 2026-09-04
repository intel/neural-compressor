# SPDX-License-Identifier: Apache-2.0
"""CuTe QDQ backend selection and capability checks.

The reference QDQ functions remain the correctness oracle. A fused CuTe kernel
must be bitwise validated against them before it is enabled for inference.
"""

import importlib.util
import warnings

import torch

_FALLBACK_WARNING_EMITTED = False


@torch.library.custom_op("vllm_qdq_plugin::mxfp4_qdq_cute", mutates_args=())
def _mxfp4_qdq_cute_op(x: torch.Tensor) -> torch.Tensor:
    from .cute_kernels import run_cute_qdq

    return run_cute_qdq(x, "MXFP4")


@_mxfp4_qdq_cute_op.register_fake
def _mxfp4_qdq_cute_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("vllm_qdq_plugin::mxfp8_qdq_cute", mutates_args=())
def _mxfp8_qdq_cute_op(x: torch.Tensor) -> torch.Tensor:
    from .cute_kernels import run_cute_qdq

    return run_cute_qdq(x, "MXFP8")


@_mxfp8_qdq_cute_op.register_fake
def _mxfp8_qdq_cute_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def cute_qdq_status(x: torch.Tensor) -> tuple[bool, str]:
    """Return whether this tensor can execute a CuTe QDQ kernel."""
    if not x.is_cuda:
        return False, "input is not a CUDA tensor"
    if torch.cuda.get_device_capability(x.device) < (8, 0):
        return False, "CuTe QDQ requires SM80 or newer"
    if importlib.util.find_spec("cutlass") is None:
        return False, "NVIDIA CUTLASS DSL is not installed"
    return True, "CuTe DSL is available"


def _reference_fallback(x: torch.Tensor, group_size: int, format_name: str, reason: str | None = None) -> torch.Tensor:
    global _FALLBACK_WARNING_EMITTED
    available, capability_reason = cute_qdq_status(x)
    if not _FALLBACK_WARNING_EMITTED:
        status = reason or ("unsupported input" if available else f"unavailable: {capability_reason}")
        warnings.warn(
            f"VLLM_QDQ_CUTE=1 requested for {format_name}; {status}. Falling back to the reference QDQ.",
            RuntimeWarning,
            stacklevel=2,
        )
        _FALLBACK_WARNING_EMITTED = True

    if format_name == "MXFP4":
        from .mxfp4 import _mxfp4_qdq_reference

        return _mxfp4_qdq_reference(x, group_size)

    from .mxfp8 import _mxfp8_qdq_reference

    return _mxfp8_qdq_reference(x, group_size)


def _run_cute_or_fallback(x: torch.Tensor, group_size: int, format_name: str) -> torch.Tensor:
    op = _mxfp4_qdq_cute_op if format_name == "MXFP4" else _mxfp8_qdq_cute_op
    if torch.compiler.is_compiling():
        if group_size != 32:
            raise ValueError(f"CuTe QDQ requires group_size=32, got {group_size}")
        return op(x)

    available, capability_reason = cute_qdq_status(x)
    if available and group_size == 32 and x.is_contiguous() and x.shape[-1] % group_size == 0:
        return op(x)
    if not available:
        reason = capability_reason
    elif group_size != 32:
        reason = f"group_size={group_size} is unsupported"
    elif not x.is_contiguous():
        reason = "input is not contiguous"
    else:
        reason = f"K={x.shape[-1]} is not divisible by 32"
    return _reference_fallback(x, group_size, format_name, reason)


def mxfp4_qdq_cute(x: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Run the MXFP4 CuTe backend, or the validated reference fallback."""
    return _run_cute_or_fallback(x, group_size, "MXFP4")


def mxfp8_qdq_cute(x: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Run the MXFP8 CuTe backend, or the validated reference fallback."""
    return _run_cute_or_fallback(x, group_size, "MXFP8")
