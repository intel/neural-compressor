#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from neural_compressor.common.logger import Logger
from neural_compressor.common.utility import (
    BASE_CONFIG,
    COMPOSABLE_CONFIG,
    DEFAULT_WHITE_LIST,
    EMPTY_WHITE_LIST,
    GLOBAL,
    LOCAL,
    OP_NAME_OR_MODULE_TYPE,
)

logger = Logger().get_logger()


# Dictionary to store registered configurations
registered_configs = {}


def register_config(framework_name="None", algo_name=None):
    """Register config decorator.

    The register the configuration classes for different algorithms within specific frameworks.

    Usage example:
        @register_config(framework_name="PyTorch", algo_name="ExampleAlgorithm")
        class ExampleAlgorithmConfig:
            # Configuration details for the ExampleAlgorithm

    Args:
        framework_name: the framework name. Defaults to "None".
        algo_name: the algorithm name. Defaults to None.
    """

    def decorator(config_cls):
        registered_configs.setdefault(framework_name, {})
        registered_configs[framework_name][algo_name] = config_cls
        return config_cls

    return decorator


class BaseConfig(ABC):
    """The base config for all algorithm configs."""

    name = BASE_CONFIG
    params_list = []

    def __init__(self, white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST) -> None:
        self._global_config: Optional[BaseConfig] = None
        # For PyTorch, operator_type is the collective name for module type and functional operation type,
        # for example, `torch.nn.Linear`, and `torch.nn.functional.linear`.
        # local config is the collections of operator_type configs and operator configs
        self._local_config: Dict[str, Optional[BaseConfig]] = {}
        self._white_list = white_list

    def _post_init(self):
        if self.white_list == DEFAULT_WHITE_LIST:
            global_config = self.get_params_dict()
            self._global_config = self.__class__(**global_config, white_list=None)
        elif isinstance(self.white_list, list) and len(self.white_list) > 0:
            for op_name_or_type in self.white_list:
                global_config = self.get_params_dict()
                tmp_config = self.__class__(**global_config, white_list=None)
                self.set_local(op_name_or_type, tmp_config)
        elif self.white_list == EMPTY_WHITE_LIST:
            return
        else:
            raise NotImplementedError(
                f"The white list should be one of {DEFAULT_WHITE_LIST}, {EMPTY_WHITE_LIST},"
                " a not empty list, but got {self.white_list}"
            )

    @property
    def white_list(self):
        return self._white_list

    @white_list.setter
    def white_list(self, op_name_or_type_list: Optional[List[OP_NAME_OR_MODULE_TYPE]]):
        self._white_list = op_name_or_type_list

    @property
    def global_config(self):
        return self._global_config

    @global_config.setter
    def global_config(self, config):
        self._global_config = config

    @property
    def local_config(self):
        return self._local_config

    @local_config.setter
    def local_config(self, config):
        self._local_config = config

    def set_local(self, operator_name: str, config: BaseConfig) -> BaseConfig:
        if operator_name in self.local_config:
            logger.warning("The configuration for %s has already been set, update it.", operator_name)
        self.local_config[operator_name] = config
        return self

    def to_dict(self, params_list=[], operator2str=None):
        result = {}
        global_config = self.get_params_dict()
        if bool(self.local_config):
            result[LOCAL] = {}
            for op_name, config in self.local_config.items():
                result[LOCAL][op_name] = config.to_dict()
            if self.global_config:
                result[GLOBAL] = global_config
        else:
            result = global_config
        return result

    def get_params_dict(self):
        result = dict()
        for param in self.params_list:
            result[param] = getattr(self, param)
        return result

    @classmethod
    def from_dict(cls, config_dict, str2operator=None):
        """Construct config from a dict.

        Args:
            config_dict: _description_
            str2operator: _description_. Defaults to None.

        Returns:
            The constructed config.
        """
        if GLOBAL not in config_dict and LOCAL not in config_dict:
            config = cls(**config_dict)
            return config
        else:
            config = cls(**config_dict.get(GLOBAL, {}))
            operator_config = config_dict.get(LOCAL, {})
            if operator_config:
                for op_name, op_config in operator_config.items():
                    config.set_local(op_name, cls(**op_config))
            return config

    @classmethod
    def to_diff_dict(cls, instance) -> Dict[str, Any]:
        # TODO (Yi) to implement it
        return {}

    @classmethod
    def from_json_file(cls, filename):
        with open(filename, "r", encoding="utf-8") as file:
            config_dict = json.load(file)
        return cls.from_dict(**config_dict)

    def to_json_file(self, filename):
        config_dict = self.to_dict()
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(config_dict, file, indent=4)
        logger.info("Dump the config into %s.", filename)

    def to_json_string(self, use_diff: bool = False) -> str:
        """Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `BaseConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict(self)
        else:
            config_dict = self.to_dict()
        try:
            return json.dumps(config_dict, indent=2) + "\n"
        except:
            return config_dict

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} {self.to_json_string()}"

    @classmethod
    @abstractmethod
    def register_supported_configs(cls):
        """Add all supported configs."""
        raise NotImplementedError

    @classmethod
    def validate(self, user_config: BaseConfig):
        # TODO(Yi) validate the user config
        pass

    def __add__(self, other: BaseConfig) -> BaseConfig:
        if isinstance(other, type(self)):
            for op_name, config in other.local_config.items():
                self.set_local(op_name, config)
            return self
        else:
            return ComposableConfig(configs=[self, other])

    def expand(self) -> List[BaseConfig]:
        """Expand the config.

        case 1
            {
                "global": { "weight_bits": [4, 6]}
            }
            expand to :
            1st trial config:
            {
                "global": { "weight_bits": 4}
            }
            2nd trial config:
            {
                "global": { "weight_bits": 6}
            }
        case 2
        # TODO (Yi) to support the expansion of config with `local`
        {
            "global": {
                "weight_bits": [4, 6]
            },
            "local":
            {
                "fc1":{
                    "weight_bits": [6, 8]
                },
                "fc2":{
                    "weight_bits": [4]
                }
            }

        } -> ?
        """
        config_list: List[BaseConfig] = []
        params_list = self.params_list
        params_dict = OrderedDict()
        config = self
        for param in params_list:
            param_val = getattr(config, param)
            # TODO (Yi) to handle param_val itself is a list
            if isinstance(param_val, list):
                params_dict[param] = param_val
            else:
                params_dict[param] = [param_val]
        for params_values in product(*params_dict.values()):
            new_config = self.__class__(**dict(zip(params_list, params_values)))
            config_list.append(new_config)
        logger.info(f"Expanded the {self.__class__.name} and got {len(config_list)} configs.")
        return config_list

    def _get_op_name_op_type_config(self):
        op_type_config_dict = dict()
        op_name_config_dict = dict()
        for name, config in self.local_config.items():
            if self._is_op_type(name):
                op_type_config_dict[name] = config
            else:
                op_name_config_dict[name] = config
        return op_type_config_dict, op_name_config_dict

    def to_config_mapping(
        self, config_list: List[BaseConfig] = None, model_info: List[Tuple[str, str]] = None
    ) -> OrderedDict[Union[str, Callable], OrderedDict[str, BaseConfig]]:
        config_mapping = OrderedDict()
        if config_list is None:
            config_list = [self]
        for config in config_list:
            global_config = config.global_config
            op_type_config_dict, op_name_config_dict = config._get_op_name_op_type_config()
            for op_name, op_type in model_info:
                if self.global_config is not None:
                    config_mapping[(op_name, op_type)] = global_config
                if op_type in op_type_config_dict:
                    config_mapping[(op_name, op_type)] = op_name_config_dict[op_type]
                for op_name_pattern in op_name_config_dict:
                    if re.match(op_name_pattern, op_name):
                        config_mapping[(op_name, op_type)] = op_name_config_dict[op_name_pattern]
        return config_mapping

    @staticmethod
    def _is_op_type(name: str) -> bool:
        # TODO (Yi), ort and tf need override it
        return not isinstance(name, str)


class ComposableConfig(BaseConfig):
    name = COMPOSABLE_CONFIG

    def __init__(self, configs: List[BaseConfig]) -> None:
        self.config_list = configs

    def __add__(self, other: BaseConfig) -> BaseConfig:
        if isinstance(other, type(self)):
            self.config_list.extend(other.config_list)
        else:
            self.config_list.append(other)
        return self

    def to_dict(self, params_list=[], operator2str=None):
        result = {}
        for config in self.config_list:
            result[config.name] = config.to_dict()
        return result

    @classmethod
    def from_dict(cls, config_dict: OrderedDict[str, Dict], config_registry: Dict[str, BaseConfig]):
        assert len(config_dict) >= 1, "The config dict must include at least one configuration."
        num_configs = len(config_dict)
        name, value = next(iter(config_dict.items()))
        config = config_registry[name].from_dict(value)
        for _ in range(num_configs - 1):
            name, value = next(iter(config_dict.items()))
            config += config_registry[name].from_dict(value)
        return config

    def to_json_string(self, use_diff: bool = False) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_config_mapping(
        self, config_list: List[BaseConfig] = None, model_info: List[Tuple[str, str]] = None
    ) -> OrderedDict[str, BaseConfig]:
        return super().to_config_mapping(self.config_list, model_info)

    @classmethod
    def register_supported_configs(cls):
        """Add all supported configs."""
        raise NotImplementedError
