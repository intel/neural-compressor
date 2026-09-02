# SPDX-License-Identifier: Apache-2.0
"""Fused CuTe DSL kernels for MXFP4 and MXFP8 QDQ."""

import threading

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T

GROUP_SIZE = 32
WARPS_PER_BLOCK = 4
THREADS_PER_BLOCK = GROUP_SIZE * WARPS_PER_BLOCK

_COMPILED_KERNELS = {}
_COMPILE_LOCK = threading.Lock()


@cutlass.dsl_user_op
def _abs_float(value, *, loc=None, ip=None):
    return cutlass.max(value, -value, loc=loc, ip=ip)


@cutlass.dsl_user_op
def _clamp(value, lower, upper, *, loc=None, ip=None):
    return cutlass.min(cutlass.max(value, lower, loc=loc, ip=ip), upper, loc=loc, ip=ip)


@cutlass.dsl_user_op
def _round_mxfp4(value, *, loc=None, ip=None):
    """Round a non-negative value to the E2M1 grid with ties-to-even."""
    return (
        (value > 0.25).to(cutlass.Float32) * 0.5
        + (value >= 0.75).to(cutlass.Float32) * 0.5
        + (value > 1.25).to(cutlass.Float32) * 0.5
        + (value >= 1.75).to(cutlass.Float32) * 0.5
        + (value > 2.5).to(cutlass.Float32)
        + (value >= 3.5).to(cutlass.Float32)
        + (value > 5.0).to(cutlass.Float32) * 2.0
    )


@cutlass.dsl_user_op
def _floor_log2(value, *, loc=None, ip=None):
    bits = cutlass.Int32(llvm.bitcast(T.i32(), value.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
    return ((bits >> 23) & 0xFF) - 127


@cutlass.dsl_user_op
def _round_half_to_even_positive(value, *, loc=None, ip=None):
    floor_value = value.to(cutlass.Int32)
    fraction = value - floor_value.to(cutlass.Float32)
    round_up = (fraction > 0.5).to(cutlass.Int32)
    tie_and_odd = ((fraction == 0.5) & ((floor_value & 1) != 0)).to(cutlass.Int32)
    return (floor_value + round_up + tie_and_odd).to(cutlass.Float32)


@cutlass.dsl_user_op
def _round_mxfp8(value, *, loc=None, ip=None):
    magnitude = _abs_float(value)
    magnitude = _clamp(magnitude, 0.0, 448.0)

    subnormal = _round_half_to_even_positive(magnitude * 512.0) / 512.0
    subnormal = _clamp(subnormal, 0.0, 7.0 / 512.0)

    normal_input = cutlass.max(magnitude, 2.0**-6)
    exponent = _clamp(_floor_log2(normal_input), -6, 8)
    step = cute.math.exp2(exponent.to(cutlass.Float32) - 3.0)
    normal = _round_half_to_even_positive(magnitude / step) * step
    normal = _clamp(normal, 2.0**-6, 448.0)

    is_normal = (magnitude >= 2.0**-6).to(cutlass.Float32)
    quantized = subnormal * (1.0 - is_normal) + normal * is_normal
    sign = 1.0 - 2.0 * (value < 0.0).to(cutlass.Float32)
    return quantized * sign


@cute.kernel
def mxfp4_qdq_kernel(x: cute.Tensor, output: cute.Tensor, group_count: cutlass.Int32):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    warp = thread // GROUP_SIZE
    lane = thread % GROUP_SIZE
    group = block * WARPS_PER_BLOCK + warp

    if group < group_count:
        index = group * GROUP_SIZE + lane
        value = x[index].to(cutlass.Float32)
        magnitude = _abs_float(value)
        block_max = cute.arch.warp_reduction_max(magnitude, threads_in_group=GROUP_SIZE)

        # Match the reference E8M0 exponent offset for E2M1 (maximum exponent 2).
        safe_max = cutlass.max(block_max, 2.0**-126)
        max_exp = _floor_log2(safe_max)
        max_power = cute.math.exp2(max_exp.to(cutlass.Float32))
        max_exp = max_exp + (safe_max >= max_power * 1.75).to(cutlass.Int32)
        scale_exp = _clamp(max_exp - 2, -127, 127)
        scale = cute.math.exp2(scale_exp.to(cutlass.Float32))

        scaled = value / scale
        quantized = _round_mxfp4(_abs_float(scaled))
        sign = 1.0 - 2.0 * (scaled < 0.0).to(cutlass.Float32)
        quantized = quantized * sign
        output[index] = (quantized * scale).to(output.element_type)


@cute.kernel
def mxfp8_qdq_kernel(x: cute.Tensor, output: cute.Tensor, group_count: cutlass.Int32):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    warp = thread // GROUP_SIZE
    lane = thread % GROUP_SIZE
    group = block * WARPS_PER_BLOCK + warp

    if group < group_count:
        index = group * GROUP_SIZE + lane
        value = x[index].to(cutlass.Float32)
        magnitude = _abs_float(value)
        block_max = cute.arch.warp_reduction_max(magnitude, threads_in_group=GROUP_SIZE)

        safe_max = cutlass.max(block_max, 2.0**-126)
        scale_exp = _clamp(_floor_log2(safe_max) - 8, -127, 127)
        scale = cute.math.exp2(scale_exp.to(cutlass.Float32))

        scaled = value / scale
        quantized = _round_mxfp8(scaled)
        output[index] = (quantized * scale).to(output.element_type)


@cute.jit
def launch_mxfp4_qdq(
    x: cute.Tensor,
    output: cute.Tensor,
    group_count: cutlass.Int32,
    stream: cuda.CUstream,
):
    mxfp4_qdq_kernel(x, output, group_count).launch(
        grid=[(group_count + WARPS_PER_BLOCK - 1) // WARPS_PER_BLOCK, 1, 1],
        block=[THREADS_PER_BLOCK, 1, 1],
        stream=stream,
    )


@cute.jit
def launch_mxfp8_qdq(
    x: cute.Tensor,
    output: cute.Tensor,
    group_count: cutlass.Int32,
    stream: cuda.CUstream,
):
    mxfp8_qdq_kernel(x, output, group_count).launch(
        grid=[(group_count + WARPS_PER_BLOCK - 1) // WARPS_PER_BLOCK, 1, 1],
        block=[THREADS_PER_BLOCK, 1, 1],
        stream=stream,
    )


def _get_compiled_kernel(format_name: str, dtype: torch.dtype, element_count: int, device: torch.device):
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    key = (format_name, dtype, device_index)
    compiled = _COMPILED_KERNELS.get(key)
    if compiled is not None:
        return compiled

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "CuTe QDQ kernel is not prepared for this format, dtype, and device. "
            "Run one eager QDQ warmup before vLLM starts CUDA Graph capture."
        )

    with _COMPILE_LOCK:
        compiled = _COMPILED_KERNELS.get(key)
        if compiled is not None:
            return compiled

        with torch.cuda.device(device_index):
            example = torch.empty(element_count, device=device, dtype=dtype)
            example_output = torch.empty_like(example)
            cute_example = from_dlpack(example)
            cute_output = from_dlpack(example_output)
            group_count = element_count // GROUP_SIZE
            launcher = launch_mxfp4_qdq if format_name == "MXFP4" else launch_mxfp8_qdq
            compiled = cute.compile(
                launcher,
                cute_example,
                cute_output,
                group_count,
                cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
            )
        _COMPILED_KERNELS[key] = compiled
        return compiled


def prepare_cute_qdq(x: torch.Tensor, format_name: str) -> None:
    """Compile a QDQ specialization before CUDA Graph capture."""
    _get_compiled_kernel(format_name, x.dtype, x.numel(), x.device)


def run_cute_qdq(x: torch.Tensor, format_name: str) -> torch.Tensor:
    """Run a compiled fused QDQ kernel on a contiguous 2D tensor."""
    flat_x = x.reshape(-1)
    compiled = _get_compiled_kernel(format_name, x.dtype, flat_x.numel(), x.device)
    output = torch.empty_like(flat_x)
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    compiled(from_dlpack(flat_x), from_dlpack(output), flat_x.numel() // GROUP_SIZE, stream)
    return output.reshape_as(x)
