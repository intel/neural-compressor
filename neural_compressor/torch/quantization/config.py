# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, List, NamedTuple, Optional, Union

from neural_compressor.common.base_config import BaseConfig, register_config
from neural_compressor.common.utils import DEFAULT_WHITE_LIST, FP8_QUANT, OP_NAME_OR_MODULE_TYPE

FRAMEWORK_NAME = "torch"


class OperatorConfig(NamedTuple):
    config: BaseConfig
    operators: List[Union[str, Callable]]
    valid_func_list: List[Callable] = []


@register_config(framework_name=FRAMEWORK_NAME, algo_name=FP8_QUANT)
class FP8QuantConfig(BaseConfig):
    """Config class for FP8 quantization."""

    name = FP8_QUANT
    supported_configs: List[OperatorConfig] = []
    params_list = [
        "w_dtype",
        "w_observer",
        "act_dtype",
        "act_observer",
        "stochastic",
        "calibration_file",
        "scale_method",
        "approach",
    ]

    def __init__(
        self,
        w_dtype: str = "fp8_e4m3",
        w_observer: Union[str, List[str]] = "minmax_per_channel",
        act_dtype: str = "fp8_e4m3",
        act_observer: Union[str, List[str]] = "minmax",
        stochastic: bool = True,
        calibration_file: str = "calibration_file",
        scale_method: str = "without_scale",
        approach: Union[str, List[str]] = "static",
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
        self.calibration_file = calibration_file
        self.scale_method = scale_method
        self.approach = approach
        self._post_init()

    @classmethod
    def serialize(cls):
        # dict -> json file
        pass

    @classmethod
    def deserialize(cls):
        # json file -> dict
        pass

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "FP8QuantConfig", List["FP8QuantConfig"]]:
        # TODO: for auto-tune
        return FP8QuantConfig(act_observer=["minmax", "kl"])


def get_default_fp8_config() -> FP8QuantConfig:
    """Generate the default fp8 config.

    Returns:
        the default fp8 config.
    """
    return FP8QuantConfig()


def get_default_fp8_config_set() -> FP8QuantConfig:
    """Generate the default fp8 config set.

    Returns:
        the default fp8 config.
    """
    return FP8QuantConfig.get_config_set_for_tuning()
