#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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
"""Tuning utility."""
import enum
from collections import OrderedDict
from copy import deepcopy
from typing import Dict


class QuantType(enum.IntEnum):
    """Quantization type."""

    DYNAMIC = 0
    STATIC = 1
    QAT = 2
    WEIGHT_ONLY = 3
    AUTO = 4


class QuantOptions:
    """Option Class for Quantization.

    This class is used for configuring global variable related to quantization.
    The global variable quant_options is created with this class.

    Args:
        quant_type(int): Quantization type. Default value is 1.
    """

    def __init__(self, quant_type=1):
        """Init an QuantOptions object."""
        self._quant_type = quant_type

    @property
    def quant_type(self):
        """Get quant type."""
        return self._quant_type

    @quant_type.setter
    def quant_type(self, quant_type):
        """Set quant type.

        Args:
            quant_type(int): Quantization type. Default value is 1.
        """
        self._quant_type = quant_type


quant_options = QuantOptions()


def preprocess_user_cfg(op_user_cfg: Dict):
    """Preprocess the op user config for weight only.

    Args:
        op_user_cfg: The original user config.

    Example:
        op_user_cfg = {'activation': {'bits': [4]}}
        op_user_cfg_modified = {'activation': {'bits': [4], 'group_size': [32]}}

    Returns:
        The modified config.
    """
    op_user_cfg_modified = deepcopy(op_user_cfg)
    if quant_options.quant_type == QuantType.WEIGHT_ONLY:
        for att, att_cfg in op_user_cfg.items():
            if "bits" not in att_cfg:
                op_user_cfg_modified[att]["bits"] = [4]
            if "group_size" not in att_cfg:
                op_user_cfg_modified[att]["group_size"] = [32]
    return op_user_cfg_modified


class OrderedDefaultDict(OrderedDict):
    """Ordered default dict."""

    def __missing__(self, key):
        """Initialize value for the missing key."""
        self[key] = value = OrderedDefaultDict()
        return value


def extract_data_type(data_type: str) -> str:
    """Extract data type and signed from data type.

    Args:
        data_type: The original data type such as uint8, int8.

    Returns:
        (signed or unsigned, data type without signed)
    """
    return ("signed", data_type) if data_type[0] != "u" else ("unsigned", data_type[1:])


def reverted_data_type(signed_flag: str, data_type: str) -> str:
    """Revert the data type."""
    return data_type if signed_flag == "signed" else "u" + data_type


def get_adaptor_name(adaptor):
    """Get adaptor name.

    Args:
        adaptor: adaptor instance.
    """
    adaptor_name = type(adaptor).__name__.lower()
    adaptor_name_lst = ["onnx", "tensorflow", "pytorch"]
    for name in adaptor_name_lst:
        if adaptor_name.startswith(name):
            return name
    return ""


def build_slave_faker_model():
    """Slave does not have a model, so construct a fake model.

    Returns:
        object:  a class object where all properties and methods are virtual.
    """
    from ...utils import logger

    class FakerModel:
        def __call__(self, *args, **kwargs):
            logger.warning("Slave node has no quantized model, please handle it yourself.")

        def __getitem__(self, key):
            return self.__getattr__(str(key))

        def __getattr__(self, name):
            logger.warning("Slave node has no quantized model, please handle it yourself.")
            return self

    return FakerModel()


class ClassRegister:
    """Class register."""

    def __init__(self):
        """Init class register."""
        self.register = {}

    def __call__(self, name):
        """Call the class register."""

        def decorator(func):
            self.register[name] = func
            return func

        return decorator

    def get_class(self, name):
        """Get class by name."""
        if name in self.register:
            return self.register[name]
        else:
            raise ValueError(f"Class with name '{name}' is not registered.")
