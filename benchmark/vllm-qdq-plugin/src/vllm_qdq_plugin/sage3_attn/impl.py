# SPDX-License-Identifier: Apache-2.0
"""Sage3 Triton attention implementation for vllm-omni diffusion models.

Imports the sage3 standalone kernel and wraps it with:
- NHD↔HND layout transpose (vllm-omni uses NHD, sage3 uses HND)
- Cross-attention fallback to torch SDPA (different Q/K seq lengths)
"""

import os

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionImpl,
    AttentionMetadata,
)

from .. import envs
from .sage3 import sageattn3_standalone as _sage3_fn

logger = init_logger(__name__)


class Sage3TritonImpl(AttentionImpl):
    """Attention implementation using sage3 Triton kernel."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale

        # Configuration from env vars
        self._config = envs.SAGE3_QUANT_FORMAT
        self._acc_dtype = envs.SAGE3_ACC_DTYPE

        # Override from backend_kwargs if provided
        if backend_kwargs:
            self._config = backend_kwargs.pop("sage3_config", self._config)
            self._acc_dtype = backend_kwargs.pop("sage3_acc_dtype", self._acc_dtype)
            if backend_kwargs:
                logger.warning(
                    "Sage3TritonImpl ignoring backend_kwargs: %s",
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
        # Input layout: NHD = [B, N, H, D]
        # Fall back to SDPA when Q and K/V have different sequence lengths
        # (e.g. cross-attention).  sage3's qk_smoothing and kernel both
        # assume N_q == N_k and produce out-of-bounds accesses otherwise.
        if query.shape[1] != key.shape[1]:
            return self._forward_sdpa(query, key, value)
        cfg = config_override or self._config
        return self._forward_sage3(query, key, value, config=cfg)

    def _forward_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """SDPA fallback for cross-attention (N_q != N_k)."""
        # Input is NHD [B, N, H, D]; SDPA expects [B, H, N, D]
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.softmax_scale, is_causal=False)
        return out.transpose(1, 2)  # back to NHD

    @torch.compiler.disable()
    def _forward_sage3(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        config: str | None = None,
    ) -> torch.Tensor:
        """Forward using sage3 Triton kernel."""
        cfg = config or self._config
        # sage3 expects HND = [B, H, N, D], input is NHD = [B, N, H, D]
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()

        # Dump self-attention inputs for offline analysis
        if os.environ.get("SAGE3_DUMP_INPUTS", "0") == "1":
            dump_dir = os.environ.get("SAGE3_DUMP_DIR", "/tmp/sage3_inputs")
            os.makedirs(dump_dir, exist_ok=True)
            if not hasattr(self, "_dump_count"):
                self._dump_count = 0
            max_dumps = int(os.environ.get("SAGE3_DUMP_MAX", "1"))
            if self._dump_count < max_dumps:
                path = os.path.join(dump_dir, f"self_attn_input_{self._dump_count}.pt")
                torch.save(
                    {
                        "q": q.cpu(),
                        "k": k.cpu(),
                        "v": v.cpu(),
                        "sm_scale": self.softmax_scale,
                        "is_causal": self.causal,
                        "config": cfg,
                        "acc_dtype": self._acc_dtype,
                        "q_shape": list(q.shape),  # [B, H, N, D]
                    },
                    path,
                )
                logger.info("Dumped self-attention inputs to %s", path)
                self._dump_count += 1

        out = _sage3_fn(
            q,
            k,
            v,
            config=cfg,
            is_causal=self.causal,
            sm_scale=self.softmax_scale,
            acc_dtype=self._acc_dtype,
        )
        return out.transpose(1, 2)  # back to NHD
