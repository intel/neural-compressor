import torch
from collections import OrderedDict
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union
from neural_compressor.common.base_config import (
    BaseConfig,
    config_registry,
    register_config,
    register_supported_configs_for_fwk,
)
from neural_compressor.common.utils import (
    FP8_QUANT,
    DEFAULT_WHITE_LIST,
    OP_NAME_OR_MODULE_TYPE,
)
from neural_compressor.torch.utils import is_hpex_available, logger

FRAMEWORK_NAME = "torch"

class OperatorConfig(NamedTuple):
    config: BaseConfig
    operators: List[Union[str, Callable]]
    valid_func_list: List[Callable] = []

@register_config(framework_name=FRAMEWORK_NAME, algo_name=FP8_QUANT)
class Fp8cfg(BaseConfig):
    """Config class for FP8 quantization."""

    name = FP8_QUANT
    supported_configs: List[OperatorConfig] = []
    params_list = [
        "w_dtype",
        "w_observer",
        "act_dtype",
        "act_observer",
        "stochastic",
        "calibration_result",
        "scale_method",
        "approach",
        "device",
    ]

    def __init__(
        self,
        w_dtype: str = "fp8_e4m3",
        w_observer: Union[str, List[str]] = "minmax_per_channel",
        act_dtype: str = "fp8_e4m3",
        act_observer: Union[str, List[str]] = "minmax",
        stochastic: bool = True,
        calibration_result: str = None,
        scale_method: str = "without_scale",
        approach: Union[str, List[str]] = "static",
        device: Union[str, List[str]] = "hpu",
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
    ):
        """Init FP8 config.

        Args:
        """
        super().__init__(white_list=white_list)
        self.w_dtype = w_dtype
        self.w_observer = w_observer
        self.act_dtype = act_dtype
        self.act_observer = act_observer
        self.stochastic = stochastic
        self.calibration_result = calibration_result
        self.scale_method = scale_method
        self.approach = approach
        self.device = device
        self._post_init()

    @classmethod
    def serialize(cls):
        # json file -> dict
        pass

    @classmethod
    def deserialize(cls):
        # dict -> json file
        pass

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        supported_configs = []
        fp8_config = Fp8cfg(
            w_dtype=["fp8_e5m2", "fp8_e4m3"],
            w_observer=["minmax", "minmax_per_channel"],
            act_dtype=["fp8_e5m2", "fp8_e4m3"],
            act_observer=["minmax", "kl"],
            stochastic=[True, False],
            approach=["static", "dynamic"],
            device=["hpu"],
        )
        if is_hpex_available():
            from neural_compressor.torch.algorithms.habana_fp8 import white_list

            operators = white_list
        else:
            operators = ()
        supported_configs.append(OperatorConfig(config=fp8_config, operators=operators))
        cls.supported_configs = supported_configs

    @staticmethod
    def get_model_info(model: torch.nn.Module) -> List[Tuple[str, Callable]]:
        from neural_compressor.torch.algorithms.habana_fp8 import white_list

        filter_result = []
        for op_name, module in model.named_modules():
            if isinstance(module, white_list):
                pair = (op_name, type(module).__name__)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "Fp8cfg", List["Fp8cfg"]]:
        # TODO fwk owner needs to update it.
        return Fp8cfg(act_observer=["minmax", "kl"])


def get_default_fp8_config() -> Fp8cfg:
    """Generate the default fp8 config.

    Returns:
        the default fp8 config.
    """
    return Fp8cfg()


def get_default_fp8_config_set() -> Fp8cfg:
    """Generate the default fp8 config set.

    Returns:
        the default fp8 config.
    """
    return Fp8cfg.get_config_set_for_tuning()
