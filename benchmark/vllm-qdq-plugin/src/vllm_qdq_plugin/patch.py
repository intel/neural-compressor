# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch vllm._custom_ops to inject QDQ on activations.

Wraps marlin_gemm and moe_wna16_marlin_gemm at the ops level so that
every call site (dense, MoE gate+up, MoE down) gets QDQ automatically.
"""

import inspect
import sys

import torch
from vllm.logger import init_logger

from . import envs

logger = init_logger(__name__)

# Pre-0.31 vLLM signatures; used when the wrapped fn has no introspectable params.
_LEGACY_MARLIN_GEMM_PARAMS = (
    "a", "c", "b_q_weight", "b_bias", "b_scales", "a_scales", "global_scale", "b_zeros",
    "g_idx", "perm", "workspace", "b_q_type", "size_m", "size_n", "size_k",
)  # fmt: skip
_LEGACY_MOE_MARLIN_GEMM_PARAMS = (
    "input", "output", "b_qweight", "b_bias", "b_scales", "a_scales", "global_scale", "b_qzeros",
    "g_idx", "perm", "workspace", "sorted_token_ids", "expert_ids", "num_tokens_past_padded",
    "topk_weights", "moe_block_size", "top_k", "mul_topk_weights", "b_q_type",
)  # fmt: skip


def _positional_param_names(fn, legacy: tuple[str, ...]) -> tuple[str, ...]:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return legacy
    names = tuple(
        p.name for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    return names or legacy


def _make_qdq_wrapper(_orig, param_names, input_name, type_name, type_matches, mxfp4_qdq, mxfp8_qdq, op_name):
    """Wrap a marlin gemm so QDQ is applied to its activation regardless of signature."""
    idx = {name: i for i, name in enumerate(param_names)}

    def _get(args, kwargs, name):
        if name in kwargs:
            return kwargs[name]
        i = idx.get(name)
        return args[i] if i is not None and i < len(args) else None

    def _patched(*args, **kwargs):
        x = _get(args, kwargs, input_name)
        if isinstance(x, torch.Tensor) and x.dim() == 2:
            b_q_type = _get(args, kwargs, type_name)
            global_scale = _get(args, kwargs, "global_scale")
            qdq = None
            if op_name == "moe_wna16_marlin_gemm" and envs.VLLM_MARLIN_MOE_QDQ_MODE == "FORCE_MXFP4":
                qdq = mxfp4_qdq
            elif type_matches(b_q_type, "float4_e2m1f") and global_scale is None:
                qdq = mxfp4_qdq
            elif type_matches(b_q_type, "float8_e4m3fn"):
                qdq = mxfp8_qdq
            if qdq is not None and (op_name == "moe_wna16_marlin_gemm" or x.dtype in (torch.float16, torch.bfloat16)):
                x = qdq(x, group_size=32, trace_op_name=op_name)
                if input_name in kwargs:
                    kwargs[input_name] = x
                else:
                    args = (x, *args[1:])
        return _orig(*args, **kwargs)

    return _patched


def _patch_marlin_gemm(ops, scalar_types, mxfp4_qdq, mxfp8_qdq):
    """Patch ops.marlin_gemm with QDQ wrapper.

    Returns:
        Tuple of (attr_name, original_fn, patched_fn) for sys.modules fixup.
    """
    _orig = ops.marlin_gemm
    _patched = _make_qdq_wrapper(
        _orig,
        _positional_param_names(_orig, _LEGACY_MARLIN_GEMM_PARAMS),
        "a",
        "b_q_type",
        lambda t, name: t == getattr(scalar_types, name),
        mxfp4_qdq,
        mxfp8_qdq,
        "marlin_gemm",
    )
    ops.marlin_gemm = _patched
    return ("marlin_gemm", _orig, _patched)


def _patch_marlin_gemm_custom_op(scalar_types, mxfp4_qdq, mxfp8_qdq) -> bool:
    """vLLM>=0.31 dense path: torch.ops.vllm.marlin_gemm -> torch.ops._C.marlin_gemm, bypassing ops.marlin_gemm.

    Patches the ``_C`` op looked up at runtime inside the opaque custom-op impl.
    """
    try:
        from vllm.model_executor.layers.quantization.utils import marlin_utils
    except ImportError:
        return False
    if not hasattr(marlin_utils, "_marlin_gemm"):
        return False

    _orig = torch.ops._C.marlin_gemm
    param_names = tuple(a.name for a in _orig.default._schema.arguments)
    torch.ops._C.marlin_gemm = _make_qdq_wrapper(
        _orig,
        param_names,
        "a",
        "b_type_id",
        lambda t, name: t == getattr(scalar_types, name).id,
        mxfp4_qdq,
        mxfp8_qdq,
        "marlin_gemm",
    )
    return True


def _patch_moe_marlin_gemm(ops, scalar_types, mxfp4_qdq, mxfp8_qdq):
    """Patch ops.moe_wna16_marlin_gemm with QDQ wrapper.

    Returns:
        Tuple of (attr_name, original_fn, patched_fn) for sys.modules fixup.
    """
    _orig = ops.moe_wna16_marlin_gemm
    _patched = _make_qdq_wrapper(
        _orig,
        _positional_param_names(_orig, _LEGACY_MOE_MARLIN_GEMM_PARAMS),
        "input",
        "b_q_type",
        lambda t, name: t == getattr(scalar_types, name),
        mxfp4_qdq,
        mxfp8_qdq,
        "moe_wna16_marlin_gemm",
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

    _patch_mla_kv_b_proj_dtype()

    # ops.marlin_gemm also forwards to torch.ops._C.marlin_gemm, so patch only one layer for dense.
    dense_custom_op = _patch_marlin_gemm_custom_op(scalar_types, mxfp4_qdq, mxfp8_qdq)
    patches = [_patch_moe_marlin_gemm(ops, scalar_types, mxfp4_qdq, mxfp8_qdq)]
    if not dense_custom_op:
        patches.insert(0, _patch_marlin_gemm(ops, scalar_types, mxfp4_qdq, mxfp8_qdq))

    # Fix up any modules that imported these via
    # `from vllm._custom_ops import marlin_gemm` (direct ref).
    for mod in list(sys.modules.values()):
        try:
            for attr_name, orig, patched in patches:
                if getattr(mod, attr_name, None) is orig:
                    setattr(mod, attr_name, patched)
        except Exception:
            pass

    names = [name for name, _, _ in patches]
    if dense_custom_op:
        names.append("torch.ops._C.marlin_gemm")
    logger.warning("QDQ patches applied: %s (MXFP4, MXFP8)", ", ".join(names))
    try:
        if not envs.is_set("VLLM_QDQ_CUTE"):
            logger.warning("CUTE QDQ backend defaults to automatic capability detection")
        elif envs.VLLM_QDQ_CUTE:
            logger.warning("CUTE QDQ backend requested: VLLM_QDQ_CUTE=1 (will probe availability at runtime)")
        else:
            logger.info("CUTE QDQ backend disabled: VLLM_QDQ_CUTE=0 (using reference QDQ)")
    except Exception:
        # Be defensive: envs may raise in exotic contexts; avoid crashing EngineCore startup.
        logger.debug("Failed to read VLLM_QDQ_CUTE environment flag for CUTE logging")
