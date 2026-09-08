# SPDX-License-Identifier: Apache-2.0
"""SpargeAttn block-sparse attention backend for vllm-omni diffusion models.

Wraps the prebuilt ``spas_sage_attn`` CUDA kernels as a vllm-omni attention
backend. Registered (overriding the SAGE_ATTN slot) when VLLM_SPARGE_ATTN=1.
SpargeAttn's source is never modified — this package is pure integration glue.
"""
