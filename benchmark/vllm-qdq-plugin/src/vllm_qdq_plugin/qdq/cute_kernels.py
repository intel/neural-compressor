# SPDX-License-Identifier: Apache-2.0
"""Fused CuTe DSL kernels for MXFP4, MXFP8, NVFP4, and NVFP4_E5M3(_GS) QDQ."""

import importlib.util
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

NVFP4_VEC = 8
NVFP4_THREADS_PER_BLOCK = 256

_COMPILED_KERNELS = {}
_COMPILED_NVFP4_KERNELS = {}
_COMPILE_LOCK = threading.Lock()
# TVM-FFI accepts torch tensors directly, avoiding per-call DLPack conversion overhead.
_USE_TVM_FFI = importlib.util.find_spec("tvm_ffi") is not None


@cutlass.dsl_user_op
def _abs_float(value, *, loc=None, ip=None):
    return cutlass.max(value, -value, loc=loc, ip=ip)


@cutlass.dsl_user_op
def _clamp(value, lower, upper, *, loc=None, ip=None):
    return cutlass.min(cutlass.max(value, lower, loc=loc, ip=ip), upper, loc=loc, ip=ip)


@cutlass.dsl_user_op
def _round_mxfp4(value, *, loc=None, ip=None):
    """Round a non-negative value to the E2M1 grid with ties-to-even."""
    # Adding 1.5 * 2^23 * step makes the FP32 adder round to a multiple of step (ties-to-even).
    magic = cutlass.Float32(
        cutlass.select_(value < 2.0, 6291456.0, cutlass.select_(value < 4.0, 12582912.0, 25165824.0))
    )
    return cutlass.min((value + magic) - magic, 6.0, loc=loc, ip=ip)


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
def _quantize_ue5m3_scale(value, *, loc=None, ip=None):
    bits = cutlass.Int32(llvm.bitcast(T.i32(), value.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
    fp32_exponent = (bits >> 23) & 0xFF
    fp32_mantissa = bits & 0x7FFFFF
    ue5m3_exponent = fp32_exponent - 112
    ue5m3_mantissa = fp32_mantissa >> 20
    remainder = fp32_mantissa & 0xFFFFF
    increment = ((remainder > 0x80000) | ((remainder == 0x80000) & ((ue5m3_mantissa & 1) != 0))).to(cutlass.Int32)
    ue5m3_mantissa = ue5m3_mantissa + increment
    carry = (ue5m3_mantissa == 8).to(cutlass.Int32)
    ue5m3_exponent = ue5m3_exponent + carry
    ue5m3_mantissa = ue5m3_mantissa * (1 - carry)
    normal = cute.math.exp2(ue5m3_exponent.to(cutlass.Float32) - 15.0) * (
        1.0 + ue5m3_mantissa.to(cutlass.Float32) * 0.125
    )
    subnormal = _round_half_to_even_positive(value * 131072.0) / 131072.0
    is_subnormal = (value < 2.0**-14).to(cutlass.Float32)
    quantized = normal * (1.0 - is_subnormal) + subnormal * is_subnormal
    return cutlass.min(quantized, 114688.0, loc=loc, ip=ip)


@cutlass.dsl_user_op
def _round_e4m3_positive(value, *, loc=None, ip=None):
    """Round a value in [0, 448] to FP8 E4M3 with ties-to-even."""
    subnormal = _round_half_to_even_positive(value * 512.0) / 512.0
    exponent = _clamp(_floor_log2(cutlass.max(value, 2.0**-6)), -6, 8)
    step = cute.math.exp2(exponent.to(cutlass.Float32) - 3.0)
    normal = _round_half_to_even_positive(value / step) * step
    is_normal = (value >= 2.0**-6).to(cutlass.Float32)
    return subnormal * (1.0 - is_normal) + normal * is_normal


@cutlass.dsl_user_op
def _safe_reciprocal(value, *, loc=None, ip=None):
    return cutlass.Float32(cutlass.select_(value == 0.0, 0.0, 1.0 / value))


@cutlass.dsl_user_op
def _decode_e2m1(nibble, *, loc=None, ip=None):
    value = nibble.to(cutlass.Int32)
    sign = (value >> 3) & 1
    exponent = (value >> 1) & 0x3
    mantissa = value & 1
    subnormal = mantissa.to(cutlass.Float32) * 0.5
    normal = cute.math.exp2(exponent.to(cutlass.Float32) - 1.0) * (1.0 + mantissa.to(cutlass.Float32) * 0.5)
    is_subnormal = (exponent == 0).to(cutlass.Float32)
    magnitude = normal * (1.0 - is_subnormal) + subnormal * is_subnormal
    return magnitude * (1.0 - 2.0 * sign.to(cutlass.Float32))


@cutlass.dsl_user_op
def _decode_ue5m3(scale_bits, *, loc=None, ip=None):
    value = scale_bits.to(cutlass.Int32)
    exponent = value >> 3
    mantissa = value & 0x7
    subnormal = mantissa.to(cutlass.Float32) * 2.0**-17
    normal = cute.math.exp2(exponent.to(cutlass.Float32) - 15.0) * (1.0 + mantissa.to(cutlass.Float32) * 0.125)
    is_subnormal = (exponent == 0).to(cutlass.Float32)
    return normal * (1.0 - is_subnormal) + subnormal * is_subnormal


@cutlass.dsl_user_op
def _round_mxfp8(value, *, loc=None, ip=None):
    quantized = _round_e4m3_positive(_clamp(_abs_float(value), 0.0, 448.0))
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


def _make_nvfp4_qdq_launcher(group_size: int, scale_format: str, has_global_scale: bool, silu_mul: bool = False):
    """Build a QDQ launcher where each thread owns one 16-byte vector of a 16/32-element group.

    ``scale_format`` is ``"E4M3"`` (standard NVFP4, requires a global scale) or ``"UE5M3"``.
    Output row ``r`` reads input row ``r // topk`` so MoE routes need no expanded copy.
    With ``silu_mul`` the input row is ``[gate | up]`` and QDQ applies to vLLM's ``silu_and_mul`` result.
    """
    use_e4m3 = scale_format == "E4M3"
    wide_group = group_size == 32

    @cute.kernel
    def kernel(
        x: cute.Tensor,
        global_scales: cute.Tensor,
        output: cute.Tensor,
        vec_count: cutlass.Int32,
        vecs_per_row: cutlass.Int32,
        scale_row_stride: cutlass.Int32,
        topk: cutlass.Int32,
    ):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        vec = block * NVFP4_THREADS_PER_BLOCK + thread
        is_valid = vec < vec_count
        # Out-of-range lanes recompute the last vector so all lanes join the shuffle.
        vec = cutlass.min(vec, vec_count - 1)
        row = vec // vecs_per_row
        x_vecs = cute.zipped_divide(x, (NVFP4_VEC,))

        values = cute.make_rmem_tensor(NVFP4_VEC, output.element_type)
        if cutlass.const_expr(silu_mul):
            gate_vec = row * 2 * vecs_per_row + vec % vecs_per_row
            gates = cute.make_rmem_tensor(NVFP4_VEC, x.element_type)
            ups = cute.make_rmem_tensor(NVFP4_VEC, x.element_type)
            gates.store(x_vecs[(None, gate_vec)].load())
            ups.store(x_vecs[(None, gate_vec + vecs_per_row)].load())
            for i in cutlass.range_constexpr(NVFP4_VEC):
                gate = gates[i].to(cutlass.Float32)
                # Same rounding points as vLLM's silu_and_mul: SiLU and the product each round to the input dtype.
                activated = (gate / (1.0 + cute.math.exp(-gate))).to(output.element_type).to(cutlass.Float32)
                values[i] = (activated * ups[i].to(cutlass.Float32)).to(output.element_type)
        else:
            source_vec = (row // topk) * vecs_per_row + vec % vecs_per_row
            values.store(x_vecs[(None, source_vec)].load())
        block_max = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(NVFP4_VEC):
            block_max = cutlass.max(block_max, _abs_float(values[i].to(cutlass.Float32)))
        block_max = cutlass.max(block_max, cute.arch.shuffle_sync_bfly(block_max, offset=1))
        if cutlass.const_expr(wide_group):
            block_max = cutlass.max(block_max, cute.arch.shuffle_sync_bfly(block_max, offset=2))

        global_scale = cutlass.Float32(1.0)
        inverse_global_scale = cutlass.Float32(1.0)
        if cutlass.const_expr(has_global_scale):
            global_scale = global_scales[row * scale_row_stride]
            inverse_global_scale = _safe_reciprocal(global_scale)
        if cutlass.const_expr(use_e4m3):
            scale = _round_e4m3_positive(cutlass.min(global_scale * (block_max * (1.0 / 6.0)), 448.0))
            normalize_scale = _safe_reciprocal(scale * inverse_global_scale)
            # Matches the reference, which dequantizes with the reciprocal of the reciprocal.
            dequant_scale = _safe_reciprocal(normalize_scale)
        elif cutlass.const_expr(has_global_scale):
            dequant_scale = _quantize_ue5m3_scale(block_max / 6.0 * global_scale) * inverse_global_scale
        else:
            dequant_scale = _quantize_ue5m3_scale(block_max / 6.0)

        result = cute.make_rmem_tensor(NVFP4_VEC, output.element_type)
        for i in cutlass.range_constexpr(NVFP4_VEC):
            value = values[i].to(cutlass.Float32)
            if cutlass.const_expr(use_e4m3):
                normalized = value * normalize_scale
            else:
                normalized = cutlass.Float32(cutlass.select_(dequant_scale == 0.0, 0.0, value / dequant_scale))
            quantized = _round_mxfp4(_abs_float(normalized))
            sign = 1.0 - 2.0 * (normalized < 0.0).to(cutlass.Float32)
            result[i] = (quantized * sign * dequant_scale).to(output.element_type)
        if is_valid:
            cute.zipped_divide(output, (NVFP4_VEC,))[(None, vec)].store(result.load())

    @cute.jit
    def launch(
        x: cute.Tensor,
        global_scales: cute.Tensor,
        output: cute.Tensor,
        vec_count: cutlass.Int32,
        vecs_per_row: cutlass.Int32,
        scale_row_stride: cutlass.Int32,
        topk: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        kernel(x, global_scales, output, vec_count, vecs_per_row, scale_row_stride, topk).launch(
            grid=[(vec_count + NVFP4_THREADS_PER_BLOCK - 1) // NVFP4_THREADS_PER_BLOCK, 1, 1],
            block=[NVFP4_THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )

    return launch


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


def _ensure_not_capturing() -> None:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "CuTe QDQ kernel is not prepared for this format, dtype, and device. "
            "Run one eager QDQ warmup before vLLM starts CUDA Graph capture."
        )


def _get_compiled_kernel(
    format_name: str,
    dtype: torch.dtype,
    element_count: int,
    device: torch.device,
):
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    key = (format_name, dtype, device_index)
    compiled = _COMPILED_KERNELS.get(key)
    if compiled is not None:
        return compiled

    _ensure_not_capturing()

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


def _dynamic_vector(tensor: torch.Tensor):
    return from_dlpack(tensor, assumed_align=16).mark_compact_shape_dynamic(mode=0, divisibility=NVFP4_VEC)


def _dynamic_scales(tensor: torch.Tensor):
    return from_dlpack(tensor).mark_layout_dynamic()


def _get_compiled_nvfp4_kernel(
    group_size: int,
    scale_format: str,
    has_global_scale: bool,
    dtype: torch.dtype,
    device: torch.device,
    silu_mul: bool = False,
):
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    key = (group_size, scale_format, has_global_scale, dtype, device_index, silu_mul)
    entry = _COMPILED_NVFP4_KERNELS.get(key)
    if entry is not None:
        return entry

    _ensure_not_capturing()

    with _COMPILE_LOCK:
        entry = _COMPILED_NVFP4_KERNELS.get(key)
        if entry is not None:
            return entry
        with torch.cuda.device(device_index):
            example = torch.empty(group_size, device=device, dtype=dtype)
            # Placeholder bound to the scale argument when the format has no global scale.
            unit_scale = torch.ones(1, device=device, dtype=torch.float32)
            compiled = cute.compile(
                _make_nvfp4_qdq_launcher(group_size, scale_format, has_global_scale, silu_mul),
                _dynamic_vector(torch.empty(group_size * (2 if silu_mul else 1), device=device, dtype=dtype)),
                _dynamic_scales(unit_scale),
                _dynamic_vector(torch.empty_like(example)),
                group_size // NVFP4_VEC,
                group_size // NVFP4_VEC,
                0,
                1,
                cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
                options="--enable-tvm-ffi" if _USE_TVM_FFI else None,
            )
        entry = (compiled, unit_scale)
        _COMPILED_NVFP4_KERNELS[key] = entry
        return entry


def run_nvfp4_qdq(
    x: torch.Tensor,
    group_size: int,
    scale_format: str,
    global_scale: torch.Tensor | None = None,
    topk: int = 1,
    output: torch.Tensor | None = None,
    silu_mul: bool = False,
) -> torch.Tensor:
    """QDQ a contiguous ``[rows, K]`` tensor into ``[rows * topk, K]`` (row ``r`` reads input row ``r // topk``).

    ``global_scale`` holds one float32 value or one per output row. ``output`` may alias ``x`` when
    ``topk == 1``. ``silu_mul`` treats ``x`` as ``[rows, 2 * K]`` gate/up halves and QDQs
    ``silu_and_mul(x)`` (requires ``topk == 1``).
    """
    rows, width = x.shape
    if silu_mul:
        if topk != 1 or width % (2 * NVFP4_VEC):
            raise ValueError("Fused SiLU-and-mul QDQ requires topk=1 and an even, vector-aligned width")
        width //= 2
    if output is None:
        output = torch.empty((rows * topk, width), dtype=x.dtype, device=x.device)
    if output.numel() == 0:
        return output
    if x.numel() >= 2**31 or output.numel() >= 2**31:
        raise ValueError(f"NVFP4 CuTe QDQ supports fewer than 2**31 elements, got {x.numel()}")
    compiled, unit_scale = _get_compiled_nvfp4_kernel(
        group_size, scale_format, global_scale is not None, x.dtype, x.device, silu_mul
    )
    scales = unit_scale if global_scale is None else global_scale.reshape(-1)
    source = x if x.data_ptr() % 16 == 0 else x.clone()
    target = output if output.data_ptr() % 16 == 0 else torch.empty_like(output)
    args = (
        output.numel() // NVFP4_VEC,
        width // NVFP4_VEC,
        0 if scales.numel() == 1 else 1,
        topk,
    )
    stream = torch.cuda.current_stream(x.device).cuda_stream
    if _USE_TVM_FFI:
        compiled(source.reshape(-1), scales, target.reshape(-1), *args, stream)
    else:
        compiled(
            _dynamic_vector(source.reshape(-1)),
            _dynamic_scales(scales),
            _dynamic_vector(target.reshape(-1)),
            *args,
            cuda.CUstream(stream),
        )
    if target is not output:
        output.copy_(target)
    return output
