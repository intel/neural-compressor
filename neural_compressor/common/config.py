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

"""
#TODO(Yi)
- [ ] support configs add, like RTNWeightOnlyQuantConfig() + GPTQWeightOnlyQuantConfig()
- [ ] validate the config
"""

import json
from abc import ABC
from enum import Enum, auto
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

from neural_compressor.common.utility import not_empty_dict

registered_configs = {}


def register_config(framework_name="None", algo_name=None):
    def decorator(config_cls):
        registered_configs.setdefault(framework_name, {})
        registered_configs[framework_name][algo_name] = config_cls
        return config_cls

    return decorator


GLOBAL = "global"
OPERATOR_TYPE = "operator_type"
OPERATOR_NAME = "operator_name"


class BaseConfig(ABC):
    def __init__(self) -> None:
        self.global_config: Optional[BaseConfig] = None
        # For PyTorch, operator_type is the collective name for module type and functional operation type,
        # for example, `torch.nn.Linear`, and `torch.nn.functional.linear`.
        self.operator_type_config: Dict[Union[str, Callable], Optional[BaseConfig]] = {}
        self.operator_name_config: Dict[str, Optional[BaseConfig]] = {}

    def set_global(self, config: BaseConfig):
        self.global_config = config

    def set_operator_type(self, operator_type: Union[str, Callable], config: BaseConfig) -> BaseConfig:
        self.operator_type_config[operator_type] = config
        return self

    def set_operator_name(self, operator_name: str, config: BaseConfig) -> BaseConfig:
        self.operator_name_config[operator_name] = config
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_dict(self, params_list=[], operator2str=None, depth=0):
        result = {}
        if not_empty_dict(self.operator_name_config):
            result[OPERATOR_NAME] = {}
            for op_name, config in self.operator_name_config.items():
                result[OPERATOR_NAME][op_name] = config.to_dict(depth=depth + 1)
        if not_empty_dict(self.operator_type_config) > 0:
            result[OPERATOR_TYPE] = {}
            for op_type, config in self.operator_type_config.items():
                _op_type = operator2str[op_type] if operator2str else op_type
                result[OPERATOR_TYPE][_op_type] = config.to_dict(depth=depth + 1)
        if self.global_config is not None:
            result[GLOBAL] = self.global_config.to_dict(depth=depth + 1)
        else:
            global_config = {}
            for param in params_list:
                global_config[param] = getattr(self, param)
            if depth == 0:
                result[GLOBAL] = global_config
            else:
                return global_config
        return result

    @classmethod
    def from_dict(cls, config_dict, str2operator=None):
        q_config = cls()
        if GLOBAL in config_dict:
            global_config = cls(**config_dict[GLOBAL])
            q_config.set_global(global_config)
        for type_name, config in config_dict.get(OPERATOR_TYPE, {}).items():
            _op_type = str2operator[type_name] if str2operator else type_name
            q_config.set_operator_type(_op_type, cls(**config))
        for op_name, config in config_dict.get(OPERATOR_NAME, {}).items():
            q_config.set_operator_name(op_name, cls(**config))
        return q_config

    @classmethod
    def to_diff_dict(cls, instance) -> Dict[str, Any]:
        # TODO (Yi) to implement it
        return {}

    @classmethod
    def from_json_file(cls, filename):
        with open(filename, "r") as file:
            config_dict = json.load(file)
        return cls.from_dict(**config_dict)

    def to_json_file(self, filename):
        config_dict = self.to_dict()
        with open(filename, "w") as file:
            json.dump(config_dict, file, indent=4)

    def to_json_string(self, use_diff: bool = False) -> str:
        """Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict(self)
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def validate(self, config: BaseConfig):
        # TODO(Yi) verify the config
        pass
