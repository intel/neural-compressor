# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch ``Attention._run_local_attention`` for per-(layer, step) routing.

The sage3 ``Sage3TritonImpl`` is already the live ``SAGE_ATTN`` backend. Its
quant config is normally fixed at construction. To route the config per layer
and per denoising step we wrap the *module-level* ``_run_local_attention`` (an
unbound method, so ``self`` is the ``Attention`` module) because the layer index
lives on the module, not the impl.

``install()`` is idempotent and only takes effect when a route file is supplied;
on any unexpected condition the wrapper falls through to the original method, so
behavior is unchanged when routing does not apply.
"""

import torch
from vllm.logger import init_logger
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    is_forward_context_available,
)

from ..impl import Sage3TritonImpl
from .route_table import SDPA_TOKEN, RouteTable

logger = init_logger(__name__)

_installed = False
_ORIG = None
_TABLE: RouteTable | None = None
_DEBUG = False
_seen: set = set()


def _debug_log(layer_idx, step, route) -> None:
    """Log each unique (layer, step, route) once, when SAGE3_ROUTE_DEBUG is set.

    Emits at WARNING because vllm's worker logger runs at WARNING level, so
    INFO lines are filtered out before they reach the log. This line is gated
    behind the explicit SAGE3_ROUTE_DEBUG opt-in and deduped per unique key, so
    it cannot spam a normal run.
    """
    key = (layer_idx, step, route)
    if key in _seen:
        return
    _seen.add(key)
    logger.warning("sage3 route: layer=%s step=%s -> %s", layer_idx, step, route)


def _routed_run_local_attention(self, query, key, value, attn_metadata):
    # Run the dispatch as control flow, never traced: it only *chooses* an
    # impl (the compute kernels run below). Without this, branching on
    # self.layer_idx / step makes torch.compile recompile per (layer, step),
    # and Python-side debug logging here would be swallowed during tracing.
    return _dispatch(self, query, key, value, attn_metadata)


@torch.compiler.disable()
def _dispatch(self, query, key, value, attn_metadata):
    # 1. Preserve the float32 → SDPA guard exactly as upstream handles it.
    if query.dtype == torch.float32:
        return _ORIG(self, query, key, value, attn_metadata)

    # 2. Only route sage3 self-attention; everything else uses the original path.
    impl = self.attention
    if self.role != "self" or not isinstance(impl, Sage3TritonImpl):
        return _ORIG(self, query, key, value, attn_metadata)

    # 3. Per-request denoising step (None outside a Wan-style denoise loop).
    step = get_forward_context().denoise_step_idx if is_forward_context_available() else None

    # 4. Resolve a route; None means "leave the impl's own config alone".
    route = _TABLE.resolve(self.layer_idx, step)
    if route is None:
        return _ORIG(self, query, key, value, attn_metadata)

    if _DEBUG:
        _debug_log(self.layer_idx, step, route)

    # 5. Full-precision token.
    if route == SDPA_TOKEN:
        return impl._forward_sdpa(query, key, value)

    # 6. Route to a specific sage3 quant config for this call only.
    return impl.forward_cuda(query, key, value, attn_metadata, config_override=route)


def install(route_file: str) -> None:
    """Install the routing wrapper from a JSON route file.

    Idempotent.
    """
    global _installed, _ORIG, _TABLE, _DEBUG

    if _installed:
        return

    from ... import envs as _plugin_envs

    _TABLE = RouteTable.from_json_file(route_file)
    _DEBUG = bool(_plugin_envs.SAGE3_ROUTE_DEBUG)
    _ORIG = Attention._run_local_attention
    Attention._run_local_attention = _routed_run_local_attention
    _installed = True

    logger.warning_once(
        "sage3 per-(layer, step) attention routing installed from %s " "(default=%s, %d rule(s), debug=%s)",
        route_file,
        _TABLE.default,
        len(_TABLE.rules),
        _DEBUG,
    )
