# SPDX-License-Identifier: Apache-2.0
"""Sage3 Triton attention backend registration for vllm-omni."""

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
)


class Sage3TritonBackend(AttentionBackend):
    """Out-of-tree sage3 Triton attention backend.

    Registered via vllm_omni.general_plugins entry_point when VLLM_SAGE3_TRITON=1.
    Overrides the in-tree SAGE_ATTN backend.
    """

    accept_output_buffer: bool = False

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN"

    @staticmethod
    def get_impl_cls() -> type:
        from .impl import Sage3TritonImpl

        return Sage3TritonImpl


class Sage3CuteBackend(Sage3TritonBackend):
    @staticmethod
    def get_impl_cls() -> type:
        from .impl_cute import Sage3CuteImpl

        return Sage3CuteImpl
