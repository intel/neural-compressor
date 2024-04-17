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
from collections import OrderedDict
from typing import Callable, List, NamedTuple, Optional, Tuple, Union

import torch

from neural_compressor.common import logger
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
    params_list = [
        "dump_stats_path",
        "fp8_config",
        "hp_dtype",
        "blocklist",
        "allowlist",
        "mode"
        "scale_method",
        "scale_params",
        "observer",
        "mod_dict",
        "measure_exclude",
    ]

    def __init__(
        self,
        dump_stats_path: str = "./hqt_output/measure",
        fp8_config: str = "E4M3",
        hp_dtype: torch.dtype = torch.bfloat16,
        blocklist: dict = {'names': [], 'types': ()},
        allowlist: dict = {'names': [], 'types': ('torch.nn.Linear', 'torch.nn.Conv2d', 'BMM')},
        mode: str = "AUTO",
        scale_method: str = "maxabs_hw",
        scale_params: dict = {},
        observer: str = "maxabs",
        mod_dict: dict = {},
        measure_exclude: str = "OUTPUT",
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
        **kwargs,
    ):
        """Init FP8 config.

        Args:
        """
        super().__init__(white_list=white_list)
        self.dump_stats_path =dump_stats_path
        self.fp8_config = fp8_config
        self.hp_dtype = hp_dtype
        self.blocklist = blocklist
        self.allowlist = allowlist
        self.mode = mode
        self.scale_method = scale_method
        self.scale_params = scale_params
        self.observer = observer
        self.mod_dict = mod_dict
        self._json_file = None
        self._post_init()

    @property
    def calibrate(self):
        return self.mode == "MEASURE"

    @property
    def quantize(self):
        return self.mode == "QUANTIZE"

    @property
    def json_file(self):
        if self._json_file is None:
            import tempfile
            from pathlib import Path

            json_file_tmp = tempfile.NamedTemporaryFile(suffix=".json")
            self.to_json_file(json_file_tmp.name)
            self.json_file(json_file_tmp.name)
        return self._json_file

    @json_file.setter
    def json_file(self, json_file):
        self._json_file = json_file

    @classmethod
    def from_json_file(cls, filename):
        with open(filename, "r", encoding="utf-8") as file:
            config_dict = json.load(file)
        config = cls.from_dict(config_dict)
        config.json_file = filename
        return config

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "FP8QuantConfig", List["FP8QuantConfig"]]:
        # TODO: for auto-tune
        return FP8QuantConfig(fp8_config=["E4M3", "E5M2"])

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

    def to_config_mapping(
        self, config_list: List[BaseConfig] = None, model_info: List[Tuple[str, str]] = None
    ):
        config_mapping = OrderedDict()
        if config_list is None:
            config_list = [self]
        for config in config_list:
            for op_name, op_type in model_info:
                config_mapping[(op_name, op_type)] = self
        return config_mapping


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
