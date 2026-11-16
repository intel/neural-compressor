# SPDX-License-Identifier: Apache-2.0
"""Optional Omni INC compatibility, isolated from vLLM's plugin entry points."""

from functools import wraps


_E5M3_DTYPES = frozenset({"nvfp4_v2", "nvfp4_v2_gs"})
_E5M3_FORMATS = frozenset(
    {"auto_round:llm_compressor_nvfp4_e5m3", "auto_round:llm_compressor_nvfp4_e5m3_gs"}
)


def register() -> None:
    """Called by Omni's general plugin loader, before quantized model creation.

    Keep the regular vLLM entry points unchanged and do not require Omni for
    users who only install vLLM. E5M3 always needs its own activation QDQ;
    registration must therefore also work when VLLM_QDQ=0.
    """
    try:
        from vllm_omni.quantization.inc_config import OmniINCConfig
    except ModuleNotFoundError as error:
        if error.name == "vllm_omni":
            return
        # Do not silently hide missing dependencies in an installed Omni.
        raise

    from vllm.model_executor.layers.quantization.inc import INCConfig

    from .patch import register as register_inc

    register_inc()
    if OmniINCConfig.__dict__.get("_vllm_qdq_e5m3_registered", False):
        return

    original = OmniINCConfig.get_quant_method

    @wraps(original)
    def get_quant_method(self, layer, prefix):
        if self.data_type in _E5M3_DTYPES:
            return INCConfig.get_quant_method(self, layer, prefix)
        return original(self, layer, prefix)

    # Assign new sets: do not mutate a set inherited from the base class.
    OmniINCConfig.SUPPORTED_DTYPES = set(OmniINCConfig.SUPPORTED_DTYPES) | _E5M3_DTYPES
    OmniINCConfig.SUPPORTED_FORMATS = set(OmniINCConfig.SUPPORTED_FORMATS) | _E5M3_FORMATS
    OmniINCConfig.get_quant_method = get_quant_method
    OmniINCConfig._vllm_qdq_e5m3_registered = True