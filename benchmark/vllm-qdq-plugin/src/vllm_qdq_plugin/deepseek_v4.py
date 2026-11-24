# SPDX-License-Identifier: Apache-2.0
"""Opt-in compressed-tensors compatibility for DeepSeek V4 on NVIDIA."""

import os
from functools import wraps


def _dequantize_mxfp8(weight, scale, chunk_rows=256):
    """Decode checkpoint-layout E4M3/E8M0 group32 weights with bounded scratch."""
    import torch

    if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
        raise ValueError("Expected a 2D E4M3 checkpoint weight")
    rows, width = weight.shape
    if width % 32 or scale.dtype != torch.uint8 or scale.shape != (rows, width // 32):
        raise ValueError("Expected uint8 E8M0 group32 scales")
    if chunk_rows <= 0 or weight.device != scale.device:
        raise ValueError("Invalid chunk size or mismatched devices")
    result = torch.empty_like(weight, dtype=torch.bfloat16)
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        # Native decoding preserves E8M0 byte 0 (2**-127), unlike bit shifting.
        factors = scale[start:stop].contiguous().view(torch.float8_e8m0fnu).float()
        block = weight[start:stop].float().reshape(stop - start, width // 32, 32)
        decoded = (block * factors.unsqueeze(-1)).reshape(stop - start, width).bfloat16()
        if not torch.isfinite(decoded).all():
            raise ValueError("Nonfinite or overflowing MXFP8 wo_a dequantization")
        result[start:stop].copy_(decoded)
    return result


def _prepare_wo_a(layer):
    import torch

    groups = getattr(layer, "bmm_batch_size", 0)
    if groups <= 0 or layer.weight.shape[0] % groups:
        raise ValueError("Invalid grouped wo_a shape")
    weight = _dequantize_mxfp8(layer.weight.detach(), layer.weight_scale.detach())
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    del layer.weight_scale
    # Native DeepSeek V4 o_proj selects inverse RoPE without FP8 quantization
    # and grouped torch.bmm from weight.dtype. No forward/kernel patch needed.


def register():
    """Install only when explicitly requested; independent of VLLM_QDQ."""
    if os.getenv("VLLM_DEEPSEEK_V4_WO_A_BF16", "0") != "1":
        return

    import vllm
    from vllm.platforms import current_platform

    # Only the NVIDIA o_proj path has been verified to dispatch by weight
    # dtype. XPU, for example, needs its original FP8 BMM post-processing.
    if not current_platform.is_cuda():
        return

    # This hook depends on checkpoint-layout weights and the native BF16
    # o_proj branch. Do not silently apply it to unverified vLLM releases.
    if vllm.__version__ != "0.33.0":
        raise RuntimeError("VLLM_DEEPSEEK_V4_WO_A_BF16 requires vLLM 0.33.0")

    from vllm.config import get_current_vllm_config
    from vllm.logger import init_logger
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        CompressedTensorsMoEMethod,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_mxfp8 import (
        CompressedTensorsW8A8Mxfp8,
    )
    from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import DeepseekV4FlashInferMLAAttention

    cls = CompressedTensorsW8A8Mxfp8
    original = cls.process_weights_after_loading
    if getattr(original, "_inc_deepseek_v4_wo_a_bf16", False):
        return
    logger = init_logger(__name__)
    get_moe_method = CompressedTensorsMoEMethod.get_moe_method

    @wraps(get_moe_method)
    def get_e5m3_moe_method(quant_config, layer, layer_name):
        if (
            layer_name.endswith(".ffn.experts")
            and get_current_vllm_config().model_config.hf_config.model_type == "deepseek_v4"
        ):
            quant_config._add_fused_moe_to_target_scheme_map()
            projection_names = (layer.ckpt_gate_proj_name, layer.ckpt_up_proj_name, layer.ckpt_down_proj_name)
            schemes = [quant_config.get_scheme_dict(layer, f"{layer_name}.0.{name}") for name in projection_names]
            scheme = schemes[0]
            if (
                scheme is not None
                and scheme.get("format") == "nvfp4-e5m3-pack-quantized"
                and all(item == scheme for item in schemes)
            ):
                from .quantization.inc_nvfp4_e5m3_moe import INCNvfp4UE5M3MoEMethod

                logger.info("DeepSeek V4 compressed-tensors NVFP4 E5M3 routed experts: %s", layer_name)
                return INCNvfp4UE5M3MoEMethod(layer.moe_config, scheme["weights"].group_size)
        return get_moe_method(quant_config, layer, layer_name)

    # FlashInfer caches the fused inverse-RoPE/FP8-output choice in __init__,
    # before weights are loaded. Its FP8 output cannot feed BF16 wo_a BMM.
    # Disable only for the exact scheme/layer that this plugin will convert.
    attention_init = DeepseekV4FlashInferMLAAttention.__init__

    @wraps(attention_init)
    def init_attention(self, vllm_config, *args, **kwargs):
        attention_init(self, vllm_config, *args, **kwargs)
        layer = self.wo_a
        if (
            vllm_config.model_config.hf_config.model_type == "deepseek_v4"
            and isinstance(getattr(layer, "scheme", None), cls)
            and getattr(layer, "is_bmm", False)
            and getattr(layer, "prefix", "").endswith(".wo_a")
        ):
            self._fuse_rope_quant = False

    DeepseekV4FlashInferMLAAttention.__init__ = init_attention

    @wraps(original)
    def process_weights_after_loading(self, layer):
        # Test cheap layer-local guards first; ordinary linears do not even
        # need to query the active model configuration.
        if getattr(layer, "is_bmm", False) and getattr(layer, "prefix", "").endswith(".wo_a"):
            config = get_current_vllm_config()
            if config.model_config.hf_config.model_type == "deepseek_v4":
                _prepare_wo_a(layer)
                logger.info("DeepSeek V4 MXFP8 wo_a loaded for BF16 compute: %s", layer.prefix)
                return None
        return original(self, layer)

    process_weights_after_loading._inc_deepseek_v4_wo_a_bf16 = True
    cls.process_weights_after_loading = process_weights_after_loading
    CompressedTensorsMoEMethod.get_moe_method = staticmethod(get_e5m3_moe_method)
