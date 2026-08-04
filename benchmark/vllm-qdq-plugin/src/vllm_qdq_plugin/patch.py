# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch vllm._custom_ops to inject QDQ on activations.

Wraps marlin_gemm and moe_wna16_marlin_gemm at the ops level so that
every call site (dense, MoE gate+up, MoE down) gets QDQ automatically.
"""

import sys

import torch
from vllm.logger import init_logger

from . import envs

logger = init_logger(__name__)


def _patch_marlin_gemm(ops, scalar_types, mxfp4_qdq, mxfp8_qdq, trace_qdq):
    """Patch ops.marlin_gemm with QDQ wrapper.

    Returns:
        Tuple of (attr_name, original_fn, patched_fn) for sys.modules fixup.
    """
    from vllm.scalar_type import ScalarType

    _orig = ops.marlin_gemm

    def _patched(
        a: torch.Tensor,
        c: torch.Tensor | None,
        b_q_weight: torch.Tensor,
        b_bias: torch.Tensor | None,
        b_scales: torch.Tensor,
        a_scales: torch.Tensor | None,
        global_scale: torch.Tensor | None,
        b_zeros: torch.Tensor | None,
        g_idx: torch.Tensor | None,
        perm: torch.Tensor | None,
        workspace: torch.Tensor,
        b_q_type: ScalarType,
        size_m: int,
        size_n: int,
        size_k: int,
        is_k_full: bool = True,
        use_atomic_add: bool = False,
        use_fp32_reduce: bool = False,
        is_zp_float: bool = False,
    ) -> torch.Tensor:
        # MXFP4 QDQ — extend with elif for other dtypes
        if b_q_type == scalar_types.float4_e2m1f and a.dim() == 2 and a.dtype in (torch.float16, torch.bfloat16):
            trace_qdq("marlin_gemm", a.shape, a.dtype)
            a = mxfp4_qdq(a, group_size=32)
            # logger.debug(
            #     "Applied MXFP4 QDQ to marlin_gemm as b_q_type is float4_e2m1f"
            # )
        elif b_q_type == scalar_types.float8_e4m3fn and a.dim() == 2 and a.dtype in (torch.float16, torch.bfloat16):
            trace_qdq("marlin_gemm", a.shape, a.dtype)
            a = mxfp8_qdq(a, group_size=32)
            # logger.debug(
            #     "Applied MXFP8 QDQ to marlin_gemm as b_q_type is float8_e4m3fn"
            # )

        return _orig(
            a,
            c,
            b_q_weight,
            b_bias,
            b_scales,
            a_scales,
            global_scale,
            b_zeros,
            g_idx,
            perm,
            workspace,
            b_q_type,
            size_m,
            size_n,
            size_k,
            is_k_full,
            use_atomic_add,
            use_fp32_reduce,
            is_zp_float,
        )

    ops.marlin_gemm = _patched
    return ("marlin_gemm", _orig, _patched)


def _patch_moe_marlin_gemm(ops, scalar_types, mxfp4_qdq, mxfp8_qdq, trace_qdq):
    """Patch ops.moe_wna16_marlin_gemm with QDQ wrapper.

    Returns:
        Tuple of (attr_name, original_fn, patched_fn) for sys.modules fixup.
    """
    from vllm.scalar_type import ScalarType

    _orig = ops.moe_wna16_marlin_gemm

    def _patched(
        input: torch.Tensor,
        output: torch.Tensor | None,
        b_qweight: torch.Tensor,
        b_bias: torch.Tensor | None,
        b_scales: torch.Tensor,
        a_scales: torch.Tensor | None,
        global_scale: torch.Tensor | None,
        b_qzeros: torch.Tensor | None,
        g_idx: torch.Tensor | None,
        perm: torch.Tensor | None,
        workspace: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_past_padded: torch.Tensor,
        topk_weights: torch.Tensor,
        moe_block_size: int,
        top_k: int,
        mul_topk_weights: bool,
        b_q_type: ScalarType,
        size_m: int,
        size_n: int,
        size_k: int,
        is_k_full: bool,
        use_atomic_add: bool,
        use_fp32_reduce: bool,
        is_zp_float: bool,
        thread_k: int = -1,
        thread_n: int = -1,
        blocks_per_sm: int = -1,
    ) -> torch.Tensor:
        # For the DSV4 W4A4 emulation, we do Q-DQ on the activation regardless of b_q_type, since the kernel will internally treat it as MXFP4.
        if input.dim() == 2 and envs.VLLM_MARLIN_MOE_QDQ_MODE == "FORCE_MXFP4":
            trace_qdq("moe_wna16_marlin_gemm", input.shape, input.dtype)
            input = mxfp4_qdq(input, group_size=32)
            # logger.debug(
            #     "Applied MXFP4 QDQ to moe_wna16_marlin_gemm due to FORCE_MXFP4 mode"
            # )
        # MXFP4 QDQ — extend with elif for other dtypes
        elif b_q_type == scalar_types.float4_e2m1f and input.dim() == 2:
            trace_qdq("moe_wna16_marlin_gemm", input.shape, input.dtype)
            input = mxfp4_qdq(input, group_size=32)
        elif b_q_type == scalar_types.float8_e4m3fn and input.dim() == 2:
            trace_qdq("moe_wna16_marlin_gemm", input.shape, input.dtype)
            input = mxfp8_qdq(input, group_size=32)

        return _orig(
            input,
            output,
            b_qweight,
            b_bias,
            b_scales,
            a_scales,
            global_scale,
            b_qzeros,
            g_idx,
            perm,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_past_padded,
            topk_weights,
            moe_block_size,
            top_k,
            mul_topk_weights,
            b_q_type,
            size_m,
            size_n,
            size_k,
            is_k_full,
            use_atomic_add,
            use_fp32_reduce,
            is_zp_float,
            thread_k,
            thread_n,
            blocks_per_sm,
        )

    ops.moe_wna16_marlin_gemm = _patched
    return ("moe_wna16_marlin_gemm", _orig, _patched)


def _patch_mla_kv_b_proj_dtype():
    """Fix MLACommonImpl._compute_prefill_context dtype detection for packed
    quantized weights (e.g. MXFP4 Marlin stores weights as torch.int32).

    vllm reads ``kv_b_proj.weight.dtype`` to decide what dtype to cast the
    activation ``kv_c_normed`` to.  For packed/non-float weight formats this
    produces a wrong cast (int32, float8, etc.) which then crashes the kernel.

    Fix: wrap ``kv_b_proj.forward`` to cast any non-float activation back to
    ``params_dtype`` before the actual computation.  This corrects the dtype
    regardless of what vllm's dtype-detection code computed.
    """
    import functools

    try:
        from vllm.model_executor.layers.attention.mla_attention import MLACommonImpl
    except ImportError:
        return

    _orig = MLACommonImpl._compute_prefill_context

    @functools.wraps(_orig)
    def _patched(self, *args, **kwargs):
        kv_b_proj = self.kv_b_proj
        real_weight = getattr(kv_b_proj, "weight", None)
        if real_weight is not None and not real_weight.dtype.is_floating_point and hasattr(kv_b_proj, "params_dtype"):
            compute_dtype = kv_b_proj.params_dtype
            orig_forward = kv_b_proj.forward

            @functools.wraps(orig_forward)
            def _fixed_forward(input_, *fwd_args, **fwd_kwargs):
                if not input_.dtype.is_floating_point:
                    input_ = input_.to(compute_dtype)
                return orig_forward(input_, *fwd_args, **fwd_kwargs)

            kv_b_proj.forward = _fixed_forward
            try:
                return _orig(self, *args, **kwargs)
            finally:
                kv_b_proj.forward = orig_forward
        return _orig(self, *args, **kwargs)

    MLACommonImpl._compute_prefill_context = _patched
    logger.warning(
        "QDQ patch applied: MLACommonImpl._compute_prefill_context " "(packed-weight dtype fix for MXFP4/NVFP4)"
    )


def apply_patches():
    """Patch ops.marlin_gemm and ops.moe_wna16_marlin_gemm with QDQ wrappers."""
    import vllm._custom_ops as ops
    from vllm.scalar_type import scalar_types

    from .qdq.mxfp4 import mxfp4_qdq
    from .qdq.mxfp8 import mxfp8_qdq
    from .trace import trace_qdq

    _patch_mla_kv_b_proj_dtype()

    patches = [
        _patch_marlin_gemm(ops, scalar_types, mxfp4_qdq, mxfp8_qdq, trace_qdq),
        _patch_moe_marlin_gemm(ops, scalar_types, mxfp4_qdq, mxfp8_qdq, trace_qdq),
    ]

    # Fix up any modules that imported these via
    # `from vllm._custom_ops import marlin_gemm` (direct ref).
    for mod in list(sys.modules.values()):
        try:
            for attr_name, orig, patched in patches:
                if getattr(mod, attr_name, None) is orig:
                    setattr(mod, attr_name, patched)
        except Exception:
            pass

    logger.warning(
        "QDQ patches applied: %s (MXFP4, MXFP8)",
        ", ".join(name for name, _, _ in patches),
    )
