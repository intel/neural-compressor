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
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Union

from neural_compressor.common.utility import BASE_CONFIG, GLOBAL, OPERATOR_NAME
from neural_compressor.utils import logger

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

    def __init__(self) -> None:
        self.global_config: Optional[BaseConfig] = None
        # For PyTorch, operator_type is the collective name for module type and functional operation type,
        # for example, `torch.nn.Linear`, and `torch.nn.functional.linear`.
        self.operator_type_config: Dict[Union[str, Callable], Optional[BaseConfig]] = {}
        self.operator_name_config: Dict[str, Optional[BaseConfig]] = {}

    def set_operator_name(self, operator_name: str, config: BaseConfig) -> BaseConfig:
        self.operator_name_config[operator_name] = config
        return self

    def _set_operator_type(self, operator_type: Union[str, Callable], config: BaseConfig) -> BaseConfig:
        # TODO (Yi), clean the usage
        # hide it from user, as we can use set_operator_name with regular expression to convert its functionality
        self.operator_type_config[operator_type] = config
        return self

    def to_dict(self, params_list=[], operator2str=None):
        result = {}
        global_config = {}
        for param in params_list:
            global_config[param] = getattr(self, param)
        if bool(self.operator_name_config):
            result[OPERATOR_NAME] = {}
            for op_name, config in self.operator_name_config.items():
                result[OPERATOR_NAME][op_name] = config.to_dict()
            result[GLOBAL] = global_config
        else:
            result = global_config
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
        config = cls(**config_dict.get(GLOBAL, {}))
        operator_config = config_dict.get(OPERATOR_NAME, {})
        if operator_config:
            for op_name, op_config in operator_config.items():
                config.set_operator_name(op_name, cls(**op_config))
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
        logger.info(f"Dump the config into {filename}")

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
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

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
        # TODO(Yi) implement config add, like RTNWeightOnlyQuantConfig() + GPTQWeightOnlyQuantConfig()
        pass

    @staticmethod
    def _is_op_type(name: str) -> bool:
        # TODO (Yi), ort and tf need override it
        return not isinstance(name, str)
