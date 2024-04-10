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

import json
import torch
from typing import Callable, List, NamedTuple, Optional, Union, Tuple

from neural_compressor.common.base_config import BaseConfig, register_config
from neural_compressor.common.utils import DEFAULT_WHITE_LIST, FP8_QUANT, OP_NAME_OR_MODULE_TYPE
from neural_compressor.common import logger

FRAMEWORK_NAME = "torch"


class OperatorConfig(NamedTuple):
    config: BaseConfig
    operators: List[Union[str, Callable]]
    valid_func_list: List[Callable] = []


@register_config(framework_name=FRAMEWORK_NAME, algo_name=FP8_QUANT)
class FP8QuantConfig(BaseConfig):
    """Config class for FP8 quantization."""

    name = FP8_QUANT
    params_list = [
        "method",
        "mode",
        "observer",
        "dump_stats_path",
        "scale_method",
    ]

    def __init__(
        self,
        method: str = "HOOKS",
        mode: str = "AUTO",
        observer: str = "maxabs",
        dump_stats_path: str = "./hqt_output/measure2",
        scale_method: str = "maxabs_hw",
        json_file: str = None,
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
        **kwargs,
    ):
        """Init FP8 config.

        Args:
        """
        super().__init__(white_list=white_list)
        self.method = method
        self.mode = mode
        self.observer = observer
        self.dump_stats_path = dump_stats_path
        self.json_file = json_file
        self._post_init()

    @property
    def calibrate(self):
        assert self.mode is not None, "Please set 'mode' to 'MEASURE' or 'QUANTIZE' in your config file"
        return self.mode == "MEASURE"

    @property
    def quantize(self):
        assert self.mode is not None, "Please set 'mode' to 'MEASURE' or 'QUANTIZE' in your config file"
        return self.mode == "QUANTIZE"

    @classmethod
    def from_json_file(cls, filename):
        with open(filename, "r", encoding="utf-8") as file:
            config_dict = json.load(file)
        config_dict['json_file'] = filename
        config = cls.from_dict(config_dict)
        return config

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "FP8QuantConfig", List["FP8QuantConfig"]]:
        # TODO: for auto-tune
        return FP8QuantConfig(act_observer=["minmax", "kl"])

    @classmethod
    def register_supported_configs(cls):
        """Add all supported configs."""
        # TODO: add supported configs
        pass

    @staticmethod
    def get_model_info(model: torch.nn.Module) -> List[Tuple[str, Callable]]:
        filter_result = [("*", "*")]
        logger.debug(f"Get model info: {filter_result}")
        return filter_result


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
