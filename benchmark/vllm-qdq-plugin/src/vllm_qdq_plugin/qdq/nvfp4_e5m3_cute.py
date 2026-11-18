"""CuTe DSL custom operators for NVFP4-family activation QDQ and NVFP4_E5M3 weight conversion."""

import torch
from vllm.utils.torch_utils import direct_register_custom_op


def _nvfp4_family_qdq_cute_impl(
    x: torch.Tensor,
    global_scale: torch.Tensor | None,
    scale_format: str,
    group_size: int,
    topk: int,
) -> torch.Tensor:
    from .cute_kernels import run_nvfp4_qdq

    return run_nvfp4_qdq(x, group_size, scale_format, global_scale, topk)


def _nvfp4_family_qdq_cute_fake(
    x: torch.Tensor,
    global_scale: torch.Tensor | None,
    scale_format: str,
    group_size: int,
    topk: int,
) -> torch.Tensor:
    del global_scale, scale_format, group_size
    return x.new_empty((x.shape[0] * topk, x.shape[1]))


direct_register_custom_op(
    op_name="qdq_plugin_nvfp4_family_qdq",
    op_func=_nvfp4_family_qdq_cute_impl,
    mutates_args=[],
    fake_impl=_nvfp4_family_qdq_cute_fake,
)


def _nvfp4_family_qdq_cute_inplace_impl(
    x: torch.Tensor,
    global_scale: torch.Tensor | None,
    scale_format: str,
    group_size: int,
) -> None:
    from .cute_kernels import run_nvfp4_qdq

    run_nvfp4_qdq(x, group_size, scale_format, global_scale, output=x)


def _nvfp4_family_qdq_cute_inplace_fake(
    x: torch.Tensor,
    global_scale: torch.Tensor | None,
    scale_format: str,
    group_size: int,
) -> None:
    del x, global_scale, scale_format, group_size


direct_register_custom_op(
    op_name="qdq_plugin_nvfp4_family_qdq_",
    op_func=_nvfp4_family_qdq_cute_inplace_impl,
    mutates_args=["x"],
    fake_impl=_nvfp4_family_qdq_cute_inplace_fake,
)


def nvfp4_family_qdq_cute(
    x: torch.Tensor,
    group_size: int,
    scale_format: str,
    global_scale: torch.Tensor | None = None,
    *,
    topk: int = 1,
    inplace: bool = False,
) -> torch.Tensor:
    """Run fused NVFP4 (``"E4M3"``) or NVFP4_E5M3 (``"UE5M3"``) activation QDQ.

    ``topk > 1`` returns each input row repeated ``topk`` times; ``inplace`` writes back into ``x``.
    """
    if global_scale is not None:
        global_scale = global_scale.reshape(-1).contiguous()
    if inplace and topk == 1 and x.is_contiguous():
        torch.ops.vllm.qdq_plugin_nvfp4_family_qdq_(x, global_scale, scale_format, group_size)
        return x
    result = torch.ops.vllm.qdq_plugin_nvfp4_family_qdq(x.contiguous(), global_scale, scale_format, group_size, topk)
    if inplace:
        return x.copy_(result)
    return result


def _silu_mul_nvfp4_family_qdq_cute_impl(
    output: torch.Tensor,
    x: torch.Tensor,
    global_scale: torch.Tensor | None,
    scale_format: str,
    group_size: int,
) -> None:
    from .cute_kernels import run_nvfp4_qdq

    run_nvfp4_qdq(x, group_size, scale_format, global_scale, output=output, silu_mul=True)


def _silu_mul_nvfp4_family_qdq_cute_fake(
    output: torch.Tensor,
    x: torch.Tensor,
    global_scale: torch.Tensor | None,
    scale_format: str,
    group_size: int,
) -> None:
    del output, x, global_scale, scale_format, group_size


direct_register_custom_op(
    op_name="qdq_plugin_silu_mul_nvfp4_family_qdq",
    op_func=_silu_mul_nvfp4_family_qdq_cute_impl,
    mutates_args=["output"],
    fake_impl=_silu_mul_nvfp4_family_qdq_cute_fake,
)


def try_fused_silu_and_mul_qdq(
    activation,
    output: torch.Tensor,
    activation_input: torch.Tensor,
    activation_config,
    group_size: int,
    scale_format: str,
    global_scale: torch.Tensor | None,
    *,
    format_name: str,
    trace_op_name: str,
) -> bool:
    """Write ``QDQ(silu_and_mul(activation_input))`` into ``output`` in one pass when supported.

    Returns ``False`` (without touching ``output``) for activations other than plain SiLU,
    so callers fall back to ``apply_moe_activation`` followed by a separate QDQ.
    """
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm_qdq_plugin.trace import trace_qdq

    from .cute import use_nvfp4_cute

    if activation != MoEActivation.SILU or (
        activation_config is not None and activation_config.clamp_limit is not None
    ):
        return False
    if (
        output.dim() != 2
        or output.dtype not in (torch.bfloat16, torch.float16)
        or activation_input.dtype != output.dtype
        or activation_input.shape != (output.shape[0], 2 * output.shape[1])
        or not output.is_contiguous()
        or not activation_input.is_contiguous()
        or not use_nvfp4_cute(output, group_size, format_name)
    ):
        return False
    if global_scale is not None:
        global_scale = global_scale.reshape(-1).contiguous()
        if global_scale.dtype != torch.float32 or global_scale.numel() not in (1, output.shape[0]):
            raise ValueError(f"{format_name} QDQ requires one float32 scale or one scale per row")
    trace_qdq(trace_op_name, output.shape, output.dtype, backend="CuTe", format_name=format_name)
    torch.ops.vllm.qdq_plugin_silu_mul_nvfp4_family_qdq(
        output, activation_input, global_scale, scale_format, group_size
    )
    return True
