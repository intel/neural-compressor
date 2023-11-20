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
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from neural_compressor.common.logger import Logger
from neural_compressor.common.utility import BASE_CONFIG, COMPOSABLE_CONFIG, GLOBAL, LOCAL

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

    def __init__(self) -> None:
        self._global_config: Optional[BaseConfig] = None
        # For PyTorch, operator_type is the collective name for module type and functional operation type,
        # for example, `torch.nn.Linear`, and `torch.nn.functional.linear`.
        # local config is the collections of operator_type configs and operator configs
        self._local_config: Dict[str, Optional[BaseConfig]] = {}

    @property
    def global_config(self):
        if self._global_config is None:
            self._global_config = self.__class__(**self.to_dict())
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
        if self.global_config is None:
            self.global_config = self.__class__(**self.to_dict())
        self.local_config[operator_name] = config
        return self

    def to_dict(self, params_list=[], operator2str=None):
        result = {}
        global_config = {}
        for param in params_list:
            global_config[param] = getattr(self, param)
        if bool(self.local_config):
            result[LOCAL] = {}
            for op_name, config in self.local_config.items():
                result[LOCAL][op_name] = config.to_dict()
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
        return json.dumps(config_dict, indent=2) + "\n"

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

    def to_config_mapping(
        self, config_list: List[BaseConfig] = None, model_info: List[Tuple[str, str]] = None
    ) -> OrderedDict[str, BaseConfig]:
        # TODO (Yi) add UTs
        config_mapping = OrderedDict()
        if config_list is None:
            config_list = [self]
        for config in config_list:
            global_config = config.global_config
            op_type_config_dict = config.operator_type_config
            op_name_config_dict = config.local_config
            for op_name, op_type in model_info:
                config_mapping[op_type][op_name] = global_config
                if op_type in op_type_config_dict:
                    config_mapping[op_type][op_name] = op_name_config_dict[op_type]
                    if op_name in op_name_config_dict:
                        config_mapping[op_type][op_name] = op_name_config_dict[op_name]

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

    def to_dict(self, params_list=[], operator2str=None):
        result = {}
        for config in self.config_list:
            result[config.name] = config.to_dict()
        return result

    @classmethod
    def from_dict(cls, config_dict, str2operator=None):
        # TODO(Yi)
        pass

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
