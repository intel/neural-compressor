"""
P-quantization function registry.

P-quant functions self-register via the @register_p_quant decorator,
which wraps the function in @triton.jit and stores it in P_QUANT_REGISTRY.
"""

import triton

P_QUANT_REGISTRY: dict[str, triton.JITFunction] = {}


def register_p_quant(name: str):
    """
    Decorator that JIT-compiles a function and registers it as a P-quant variant.

    Usage:
        @register_p_quant("mxfp4_s1")
        def p_quant_mxfp4_s1(p_tile, BLOCK_N: tl.constexpr):
            ...

    The function becomes available as P_QUANT_REGISTRY["mxfp4_s1"].
    """
    def wrapper(fn):
        jit_fn = triton.jit(fn)
        P_QUANT_REGISTRY[name] = jit_fn
        return jit_fn
    return wrapper
