# SPDX-License-Identifier: Apache-2.0
"""SpargeAttn attention implementation for vllm-omni diffusion models.

Wraps the prebuilt ``spas_sage_attn`` block-sparse CUDA kernels with:
- NHD↔HND layout transpose (vllm-omni uses NHD, SpargeAttn uses HND)
- A guard chain that falls back to torch SDPA whenever SpargeAttn's hard
  requirements are not met (fp32, cross-attention, seq_len < 128, head_dim
  not in {64,128}, or an attention mask — which SpargeAttn silently ignores).
"""

import torch
import torch.nn.functional as F

# Imported lazily (this module is only imported when SpargeAttn is selected), so a
# missing/unbuilt spas_sage_attn surfaces a clear ImportError only at that point.
from spas_sage_attn import (
    spas_sage2_attn_meansim_cuda,
    spas_sage2_attn_meansim_topk_cuda,
)
from vllm.logger import init_logger
from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionImpl,
    AttentionMetadata,
)

from .. import envs

logger = init_logger(__name__)

# SpargeAttn hard constraints (see SpargeAttn/spas_sage_attn/core.py asserts and
# inference_examples/modify_model/modify_wan.py guard).
_MIN_SEQ_LEN = 128
_SUPPORTED_HEAD_DIMS = (64, 128)


class SpargeAttnImpl(AttentionImpl):
    """Attention implementation using the SpargeAttn block-sparse kernel."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale

        # Configuration from env vars.
        self._mode = envs.SPARGE_MODE
        self._topk = float(envs.SPARGE_TOPK)
        self._cdfthreshd = float(envs.SPARGE_CDFTHRESHD)

        # Override from backend_kwargs if provided.
        if backend_kwargs:
            self._mode = backend_kwargs.pop("sparge_mode", self._mode)
            if "sparge_topk" in backend_kwargs:
                self._topk = float(backend_kwargs.pop("sparge_topk"))
            if "sparge_cdfthreshd" in backend_kwargs:
                self._cdfthreshd = float(backend_kwargs.pop("sparge_cdfthreshd"))
            if backend_kwargs:
                logger.warning(
                    "SpargeAttnImpl ignoring backend_kwargs: %s",
                    list(backend_kwargs.keys()),
                )

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
        *,
        config_override: str | None = None,
    ) -> torch.Tensor:
        # config_override is accepted for signature-compat with the sage3 routing
        # patch (which guards on isinstance Sage3TritonImpl and never targets us);
        # SpargeAttn has no per-(layer,step) config, so it is ignored.
        #
        # Input layout: NHD = [B, N, H, D]. Fall back to SDPA whenever SpargeAttn's
        # requirements are not met.
        has_mask = attn_metadata is not None and attn_metadata.attn_mask is not None
        if (
            query.dtype == torch.float32
            or query.shape[1] != key.shape[1]  # cross-attention (N_q != N_k)
            or query.shape[1] < _MIN_SEQ_LEN  # seq_len too short
            or query.shape[-1] not in _SUPPORTED_HEAD_DIMS  # head_dim unsupported
            or has_mask  # SpargeAttn silently ignores masks
        ):
            return self._forward_sdpa(query, key, value)
        return self._forward_sparge(query, key, value)

    def _forward_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """SDPA fallback when SpargeAttn's constraints are not met."""
        # Input is NHD [B, N, H, D]; SDPA expects [B, H, N, D].
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.softmax_scale, is_causal=False)
        return out.transpose(1, 2)  # back to NHD

    @torch.compiler.disable()
    def _forward_sparge(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Forward using the SpargeAttn kernel."""
        logger.warning_once(
            "SpargeAttnImpl: SpargeAttn kernel active (mode=%s, topk=%s, " "cdfthreshd=%s) — q shape %s",
            self._mode,
            self._topk,
            self._cdfthreshd,
            tuple(query.shape),
        )
        # SpargeAttn expects HND = [B, H, N, D], input is NHD = [B, N, H, D].
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()

        if self._mode == "cdfthreshd":
            out = spas_sage2_attn_meansim_cuda(
                q,
                k,
                v,
                is_causal=self.causal,
                scale=self.softmax_scale,
                cdfthreshd=self._cdfthreshd,
                tensor_layout="HND",
            )
        else:
            out = spas_sage2_attn_meansim_topk_cuda(
                q,
                k,
                v,
                is_causal=self.causal,
                scale=self.softmax_scale,
                topk=self._topk,
                tensor_layout="HND",
            )
        return out.transpose(1, 2)  # back to NHD
