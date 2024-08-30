"""Base class and helper functions for registering scaling methods."""

from typing import Dict, Optional, Any
from abc import abstractmethod, ABC
import torch
from neural_compressor.torch.algorithms.fp8_quant.model_configs import (
    ModuleConfig,
    SCALING_METHODS_TABLE,
    import_external_scaling_methods,
)

__all__ = [
    "ScalingMethodBase",
    "register_scaling_methods",
]


class ScalingMethodBase(ABC):
    """Base class for scaling methods.

    This class defines the interface for generating scaling methods for patched modules.
    """

    @staticmethod
    @abstractmethod
    def generate_op_scale_method(
        mod: torch.nn.Module,
        measurement: Optional[ModuleConfig] = None,
        params: Optional[Dict[str, Any]] = None,
        device: Optional[str] = "hpu",
        **kwargs: Dict[str, Any],
    ) -> ModuleConfig:
        raise NotImplementedError("`generate_op_scale_method` is not implemented")

    @staticmethod
    @abstractmethod
    def op_scales_to_mod_config(
        mod: torch.nn.Module,
        scales: ModuleConfig,
        params: Dict[str, Any],
        device: Optional[str] = "hpu",
        **kwargs: Dict[str, Any],
    ) -> ModuleConfig:
        raise NotImplementedError("`op_scales_to_mod_config` is not implemented")


def register_scaling_methods(patch_module_type: str, scaling_method_name: str):
    """Register a scaling method for a patched module.

    Args:
        patch_module_type (str): The type of the patched module.
        scaling_method_name (str): The name of the scaling method.

    Examples:
        @register_scaling_methods("linear", "unit_scale")
        class ScaleByMax(ScalingMethodBase):
            ...
    """

    def decorate(cls: ScalingMethodBase):
        SCALING_METHODS_TABLE[scaling_method_name] = (patch_module_type, cls)
        import_external_scaling_methods()
        return cls

    return decorate
