# SPDX-License-Identifier: Apache-2.0
"""SpargeAttn attention backend registration for vllm-omni."""

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
)


class SpargeAttnBackend(AttentionBackend):
    """Out-of-tree SpargeAttn block-sparse attention backend.

    Registered via vllm_omni.general_plugins entry_point when VLLM_SPARGE_ATTN=1.
    Overrides the in-tree SAGE_ATTN backend (mutually exclusive with sage3).
    """

    accept_output_buffer: bool = False

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # SpargeAttn kernels only support head_dim 64 or 128; other sizes hit the
        # SDPA fallback in the impl's guard chain.
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN"

    @staticmethod
    def get_impl_cls() -> type:
        from .impl import SpargeAttnImpl

        return SpargeAttnImpl
