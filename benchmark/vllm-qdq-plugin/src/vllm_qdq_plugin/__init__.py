# SPDX-License-Identifier: Apache-2.0
"""VLLM QDQ Plugin — out-of-tree activation quant-dequant simulation.

Registered as a vllm.general_plugins entry point. Activated by VLLM_QDQ=1.
Also provides sage3 Triton attention backend for vllm-omni (VLLM_SAGE3_TRITON=1).
"""

from vllm.logger import init_logger

from . import envs

logger = init_logger(__name__)


def register():
    """Called by vLLM plugin loader in every process (main + workers)."""

    if not envs.VLLM_QDQ:
        return

    from .patch import apply_patches

    apply_patches()
    logger.info("vllm-qdq-plugin: patches applied (VLLM_QDQ enabled)")


def register_omni_sage3_triton():
    try:
        from vllm_omni.diffusion.attention.backends.registry import (
            DiffusionAttentionBackendEnum,
            register_diffusion_backend,
        )

        register_diffusion_backend(
            DiffusionAttentionBackendEnum.SAGE_ATTN,
            "vllm_qdq_plugin.sage3_attn.backend.Sage3TritonBackend",
        )
        logger.warning("vllm-qdq-plugin: registered sage3 Triton backend as SAGE_ATTN " "(VLLM_SAGE3_TRITON=1)")
    except ImportError as e:
        logger.warning(
            "vllm-qdq-plugin: cannot register sage3 backend — " "vllm_omni not available (%s)",
            e,
        )


def register_omni_sage3_cute():
    try:
        from vllm_omni.diffusion.attention.backends.registry import (
            DiffusionAttentionBackendEnum,
            register_diffusion_backend,
        )

        register_diffusion_backend(
            DiffusionAttentionBackendEnum.SAGE_ATTN,
            "vllm_qdq_plugin.sage3_attn.backend.Sage3CuteBackend",
        )
        logger.warning("vllm-qdq-plugin: registered sage3 cute backend as SAGE_ATTN " "(VLLM_SAGE3_CUTE=1)")
    except ImportError as e:
        logger.warning(
            "vllm-qdq-plugin: cannot register sage3 cute backend — " "vllm_omni not available (%s)",
            e,
        )


def register_omni_sparge_attn():
    import importlib.util
    import sys

    # Put the SpargeAttn repo on sys.path if its package is not already importable.
    # register_omni runs at worker startup, before any Attention module is built and
    # thus before backend.get_impl_cls() triggers the spas_sage_attn import — so the
    # path is in place by the time the impl is loaded.
    if importlib.util.find_spec("spas_sage_attn") is None and envs.SPARGE_ATTN_REPO:
        sys.path.insert(0, envs.SPARGE_ATTN_REPO)
        logger.warning(
            "vllm-qdq-plugin: added SPARGE_ATTN_REPO to sys.path (%s)",
            envs.SPARGE_ATTN_REPO,
        )

    try:
        from vllm_omni.diffusion.attention.backends.registry import (
            DiffusionAttentionBackendEnum,
            register_diffusion_backend,
        )

        register_diffusion_backend(
            DiffusionAttentionBackendEnum.SAGE_ATTN,
            "vllm_qdq_plugin.sparge_attn.backend.SpargeAttnBackend",
        )
        logger.warning("vllm-qdq-plugin: registered SpargeAttn backend as SAGE_ATTN " "(VLLM_SPARGE_ATTN=1)")
    except ImportError as e:
        logger.warning(
            "vllm-qdq-plugin: cannot register SpargeAttn backend — " "vllm_omni not available (%s)",
            e,
        )


def _maybe_install_route(route_file: str):
    """Install per-(layer, step) attention routing if a route file is set.

    A bad route file logs and falls back to un-routed behavior rather than
    killing worker startup.
    """
    if not route_file:
        return
    try:
        from .sage3_attn.routing import patch_attention

        patch_attention.install(route_file)
    except Exception as e:
        logger.warning(
            "vllm-qdq-plugin: failed to install sage3 attention routing " "from %s (%s) — continuing without routing",
            route_file,
            e,
        )


def register_omni():
    """Called by vllm-omni plugin loader in diffusion workers.

    Conditionally overrides SAGE_ATTN backend with sage3 Triton implementation.
    When VLLM_SAGE3_TRITON=0 (default), does nothing — original in-tree backend used.
    """
    # sage3 and SpargeAttn both override the SAGE_ATTN slot, so they are mutually
    # exclusive. If both are requested, SpargeAttn wins deterministically.
    sage3_requested = envs.VLLM_SAGE3_TRITON or envs.VLLM_SAGE3_CUTE
    if envs.VLLM_SPARGE_ATTN and sage3_requested:
        logger.warning(
            "vllm-qdq-plugin: both VLLM_SPARGE_ATTN and a sage3 flag are set; "
            "they share the SAGE_ATTN backend slot. Using SpargeAttn."
        )

    if envs.VLLM_SPARGE_ATTN:
        register_omni_sparge_attn()
        logger.warning_once("vllm-qdq-plugin: registered SpargeAttn backend for vllm-omni")
    elif envs.VLLM_SAGE3_TRITON:
        register_omni_sage3_triton()
        logger.warning_once("vllm-qdq-plugin: registered sage3 Triton backend for vllm-omni")
        _maybe_install_route(envs.SAGE3_ROUTE_FILE)
    elif envs.VLLM_SAGE3_CUTE:
        register_omni_sage3_cute()
        logger.warning_once("vllm-qdq-plugin: registered sage3 cute backend for vllm-omni")
    else:
        logger.warning_once(
            "vllm-qdq-plugin: no sage3 backend registered for vllm-omni — set VLLM_SAGE3_TRITON=1 or VLLM_SAGE3_CUTE=1 to enable"
        )
