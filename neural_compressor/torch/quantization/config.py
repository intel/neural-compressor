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

from enum import Enum
from typing import Callable, Dict, List, NamedTuple, Union

import torch

from neural_compressor.common.base_config import BaseConfig, register_config, registered_configs
from neural_compressor.common.utility import DUMMY_CONFIG, GPTQ, RTN_WEIGHT_ONLY_QUANT

FRAMEWORK_NAME = "torch"


class Backend(Enum):
    DEFAULT = "stock_pytorch"
    IPEX = "ipex"


class OperatorConfig(NamedTuple):
    config: BaseConfig
    operators: List[Union[str, Callable]]
    backend: List[Backend]
    valid_func_list: List[Callable] = []


# mapping the torch module type and functional operation type to string representations
operator2str = {torch.nn.Linear: "Linear", torch.nn.functional.linear: "linear", torch.nn.Conv2d: "Conv2d"}

# Mapping from string representations to their corresponding torch operation/module type
str2operator = {"Linear": torch.nn.Linear, "linear": torch.nn.functional.linear, "Conv2d": torch.nn.Conv2d}


######################## RNT Config ###############################


@register_config(framework_name=FRAMEWORK_NAME, algo_name=RTN_WEIGHT_ONLY_QUANT)
class RTNWeightQuantConfig(BaseConfig):
    """Config class for round-to-nearest weight-only quantization."""

    supported_configs: List[OperatorConfig] = []
    params_list = [
        "weight_dtype",
        "weight_bits",
        "weight_group_size",
        "weight_sym",
        "act_dtype",
        "enable_full_range",
        "enable_mse_search",
        "group_dim",
        "return_int",
        "double_quant_dtype",
        "double_quant_bits",
        "double_quant_sym",
        "double_quant_group_size",
    ]
    name = RTN_WEIGHT_ONLY_QUANT

    def __init__(
        self,
        weight_dtype: str = "int",
        weight_bits: int = 4,
        weight_group_size: int = 32,
        weight_sym: bool = True,
        act_dtype: str = "fp32",
        enable_full_range: bool = False,
        enable_mse_search: bool = False,
        group_dim: int = 1,
        return_int: bool = False,
        double_quant_dtype: str = "fp32",
        double_quant_bits: int = 8,
        double_quant_sym: bool = True,
        double_quant_group_size: int = 256,
    ):
        """Init RTN weight-only quantization config.

        Args:
            weight_dtype (str): Data type for weights, default is "int".
            weight_bits (int): Number of bits used to represent weights, default is 4.
            weight_group_size (int): Size of weight groups, default is 32.
            weight_sym (bool): Indicates whether weights are symmetric, default is True.
            act_dtype (str): Data type for activations, default is "fp32".
            enable_full_range (bool): Enables full range for activations, default is False.
            enable_mse_search (bool): Enables mean squared error (MSE) search, default is False.
            group_dim (int): Dimension for grouping, default is 1.
            return_int (bool): Enables return model in int8/uint8 format or not. Defaults to False.
        """
        super().__init__()
        self.weight_bits = weight_bits
        self.weight_dtype = weight_dtype
        self.weight_group_size = weight_group_size
        self.weight_sym = weight_sym
        self.act_dtype = act_dtype
        self.enable_full_range = enable_full_range
        self.enable_mse_search = enable_mse_search
        self.group_dim = group_dim
        self.return_int = return_int
        self.double_quant_bits = double_quant_bits
        self.double_quant_dtype = double_quant_dtype
        self.double_quant_sym = double_quant_sym
        self.double_quant_group_size = double_quant_group_size

    def to_dict(self):
        return super().to_dict(params_list=self.params_list, operator2str=operator2str)

    @classmethod
    def from_dict(cls, config_dict):
        return super(RTNWeightQuantConfig, cls).from_dict(config_dict=config_dict, str2operator=str2operator)

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        supported_configs = []
        linear_rtn_config = RTNWeightQuantConfig(
            weight_dtype=["int", "int8", "int4", "nf4", "fp4", "fp4_e2m1_bnb", "fp4_e2m1"],
            weight_bits=[4, 1, 2, 3, 5, 6, 7, 8],
            weight_group_size=[32, -1, 1, 4, 8, 16, 64, 128, 256, 512, 1024],
            weight_sym=[True, False],
            act_dtype=["fp32"],
            enable_full_range=[False, True],
            enable_mse_search=[False, True],
            group_dim=[1, 0],
        )
        operators = [torch.nn.Linear, torch.nn.functional.linear]
        supported_configs.append(OperatorConfig(config=linear_rtn_config, operators=operators, backend=Backend.DEFAULT))
        cls.supported_configs = supported_configs


# TODO(Yi) run `register_supported_configs` for all registered config.
RTNWeightQuantConfig.register_supported_configs()


def get_default_rtn_config() -> RTNWeightQuantConfig:
    """Generate the default rtn config.

    Returns:
        the default rtn config.
    """
    return RTNWeightQuantConfig()


######################## GPTQ Config ###############################
@register_config(framework_name=FRAMEWORK_NAME, algo_name=GPTQ)
class GPTQConfig(BaseConfig):
    """Config class for GPTQ.

    GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.
    https://arxiv.org/abs/2210.17323
    """

    name = GPTQ
    supported_configs: List[OperatorConfig] = []
    params_list = [
        "weight_dtype",
        "weight_bits",
        "weight_group_size",
        "weight_sym",
        "block_size",
        "act_dtype",
        "group_dim",
        "nsamples",
        "dataloader_len",
        "percdamp",
        "act_order",
        "use_max_length",
        "pad_max_length",
        "enable_mse_search",
        "device",
        "layer_wise",
        "return_int",
    ]

    def __init__(
        self,
        weight_dtype: str = "int",
        weight_bits: int = 4,
        weight_group_size: int = 32,
        weight_sym: bool = True,
        block_size: int = 128,
        act_dtype: str = "fp32",
        group_dim: int = 1,
        nsamples: int = 128,
        dataloader_len: int = 10,
        percdamp: float = 0.01,
        act_order: bool = False,
        use_max_length: bool = True,
        pad_max_length: int = 2048,
        enable_mse_search: bool = False,
        device=None,
        layer_wise: bool = False,
        return_int: bool = False,
    ):
        """Init GPTQ config.

        Args:
        """
        super().__init__()
        self.weight_dtype = weight_dtype
        self.weight_bits = weight_bits
        self.weight_group_size = weight_group_size
        self.weight_sym = weight_sym
        self.act_dtype = act_dtype
        self.block_size = block_size
        self.enable_mse_search = enable_mse_search
        self.group_dim = group_dim
        self.nsamples = nsamples
        # TODO(Yi) detect it auto
        self.dataloader_len = dataloader_len
        self.percdamp = percdamp
        self.act_order = act_order
        self.use_max_length = use_max_length
        self.pad_max_length = pad_max_length
        self.layer_wise = layer_wise
        self.device = device
        self.return_int = return_int

    def to_dict(self):
        return super().to_dict(params_list=self.params_list, operator2str=operator2str)

    @classmethod
    def from_dict(cls, config_dict):
        return super(GPTQConfig, cls).from_dict(config_dict=config_dict, str2operator=str2operator)

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        supported_configs = []
        # TODO(Yi)
        linear_gptq_config = GPTQConfig()
        operators = [torch.nn.Linear, torch.nn.functional.linear]
        supported_configs.append(
            OperatorConfig(config=linear_gptq_config, operators=operators, backend=Backend.DEFAULT)
        )
        cls.supported_configs = supported_configs


# TODO(Yi) run `register_supported_configs` for all registered config.
GPTQConfig.register_supported_configs()


def get_default_gptq_config() -> GPTQConfig:
    """Generate the default gptq config.

    Returns:
        the default gptq config.
    """
    return GPTQConfig()


######################## Dummy Config ###############################
# TODO (Yi) remove it after finishing the GPTQ config
@register_config(framework_name=FRAMEWORK_NAME, algo_name=DUMMY_CONFIG)
class DummyConfig(BaseConfig):
    """Config class for round-to-nearest weight-only quantization."""

    supported_configs: List[OperatorConfig] = []
    params_list = ["act_dtype", "weight_dtype", "dummy_attr"]
    name = DUMMY_CONFIG

    def __init__(
        self,
        weight_dtype: str = "int",
        act_dtype: str = "fp32",
        dummy_attr: int = 0,
    ):
        """Init RTN weight-only quantization config.

        Args:
            act_dtype (str): Data type for activations, default is "fp32".
            weight_dtype (str): Data type for weights, default is "int".
            dummy_attr (int): Dummy attribute, default is 0.
        """
        super().__init__()
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.dummy_attr = dummy_attr

    def to_dict(self):
        return super().to_dict(params_list=self.params_list, operator2str=operator2str)

    @classmethod
    def from_dict(cls, config_dict):
        return super(DummyConfig, cls).from_dict(config_dict=config_dict, str2operator=str2operator)

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        supported_configs = []
        linear_dummy_config = DummyConfig(
            act_dtype=["fp32"],
            weight_dtype=["int4", "int8"],
            dummy_attr=[1, 2, 3],
        )
        operators = [torch.nn.Linear, torch.nn.functional.linear]
        supported_configs.append(
            OperatorConfig(config=linear_dummy_config, operators=operators, backend=Backend.DEFAULT)
        )
        cls.supported_configs = supported_configs


def get_default_dummy_config() -> DummyConfig:
    """Generate the default dummy config.

    Returns:
        the default dummy config.
    """
    return DummyConfig()


##################### Algo Configs End ###################################


def get_all_registered_configs() -> Dict[str, BaseConfig]:
    return registered_configs.get(FRAMEWORK_NAME, {})
