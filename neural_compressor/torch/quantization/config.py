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
# pylint:disable=import-error

from __future__ import annotations

from enum import Enum
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import torch

from neural_compressor.common.base_config import BaseConfig, config_registry, register_config
from neural_compressor.common.utils import DEFAULT_WHITE_LIST, FP8_QUANT, GPTQ, OP_NAME_OR_MODULE_TYPE, RTN
from neural_compressor.torch.utils.constants import PRIORITY_GPTQ, PRIORITY_RTN
from neural_compressor.torch.utils.utility import is_hpex_avaliable, logger

FRAMEWORK_NAME = "torch"
DTYPE_RANGE = Union[torch.dtype, List[torch.dtype]]


class OperatorConfig(NamedTuple):
    config: BaseConfig
    operators: List[Union[str, Callable]]
    valid_func_list: List[Callable] = []


######################## RNT Config ###############################


@register_config(framework_name=FRAMEWORK_NAME, algo_name=RTN, priority=PRIORITY_RTN)
class RTNConfig(BaseConfig):
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
    name = RTN

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
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
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
            double_quant_dtype (str): Data type for double_quant scale, default is "int".
            double_quant_bits (int): Number of bits used to represent double_quant scale, default is 4.
            double_quant_sym (bool): Indicates whether double_quant scale are symmetric, default is True.
            double_quant_group_size (int): Size of double_quant groups, default is 32.
        """
        super().__init__(white_list=white_list)
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
        self._post_init()

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        supported_configs = []
        linear_rtn_config = RTNConfig(
            weight_dtype=["int", "int8", "int4", "nf4", "fp4", "fp4_e2m1_bnb", "fp4_e2m1"],
            weight_bits=[4, 1, 2, 3, 5, 6, 7, 8],
            weight_group_size=[32, -1, 1, 4, 8, 16, 64, 128, 256, 512, 1024],
            weight_sym=[True, False],
            act_dtype=["fp32"],
            enable_full_range=[False, True],
            enable_mse_search=[False, True],
            group_dim=[1, 0],
            double_quant_bits=[4, 1, 2, 3, 5, 6, 7, 8],
            double_quant_dtype=["int", "int8", "int4", "nf4", "fp4", "fp4_e2m1_bnb", "fp4_e2m1"],
            double_quant_sym=[True, False],
            double_quant_group_size=[32, -1, 1, 4, 8, 16, 64, 128, 256, 512, 1024],
        )
        operators = [torch.nn.Linear, torch.nn.functional.linear]
        supported_configs.append(OperatorConfig(config=linear_rtn_config, operators=operators))
        cls.supported_configs = supported_configs

    @staticmethod
    def get_model_info(model: torch.nn.Module) -> List[Tuple[str, Callable]]:
        white_list = (torch.nn.Linear,)
        filter_result = []
        for op_name, module in model.named_modules():
            if isinstance(module, white_list):
                pair = (op_name, type(module).__name__)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result


# TODO(Yi) run `register_supported_configs` for all registered config.
RTNConfig.register_supported_configs()


def get_default_rtn_config() -> RTNConfig:
    """Generate the default rtn config.

    Returns:
        the default rtn config.
    """
    return RTNConfig()


######################## GPTQ Config ###############################
@register_config(framework_name=FRAMEWORK_NAME, algo_name=GPTQ, priority=PRIORITY_GPTQ)
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
        "double_quant_dtype",
        "double_quant_bits",
        "double_quant_sym",
        "double_quant_group_size",
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
        double_quant_dtype: str = "fp32",
        double_quant_bits: int = 8,
        double_quant_sym: bool = True,
        double_quant_group_size: int = 256,
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
    ):
        """Init GPTQ config.

        Args:
        """
        super().__init__(white_list=white_list)
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
        self.double_quant_bits = double_quant_bits
        self.double_quant_dtype = double_quant_dtype
        self.double_quant_sym = double_quant_sym
        self.double_quant_group_size = double_quant_group_size
        self._post_init()

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        supported_configs = []
        # TODO(Yi)
        linear_gptq_config = GPTQConfig()
        operators = [torch.nn.Linear, torch.nn.functional.linear]
        supported_configs.append(OperatorConfig(config=linear_gptq_config, operators=operators))
        cls.supported_configs = supported_configs

    @staticmethod
    def get_model_info(model: torch.nn.Module) -> List[Tuple[str, Callable]]:
        white_list = (torch.nn.Linear,)
        filter_result = []
        for op_name, module in model.named_modules():
            if isinstance(module, white_list):
                pair = (op_name, type(module).__name__)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result


# TODO(Yi) run `register_supported_configs` for all registered config.
GPTQConfig.register_supported_configs()


def get_default_gptq_config() -> GPTQConfig:
    """Generate the default gptq config.

    Returns:
        the default gptq config.
    """
    return GPTQConfig()


######################## FP8 Config ###############################
if is_hpex_avaliable():

    @register_config(framework_name=FRAMEWORK_NAME, algo_name=FP8_QUANT)
    class FP8QConfig(BaseConfig):
        """Config class for FP8 quantization."""

        name = FP8_QUANT
        supported_configs: List[OperatorConfig] = []
        params_list = [
            "weight_dtype",
            "act_dtype",
            "act_algo",
            "approach",
            "device",
        ]

        def __init__(
            self,
            weight_dtype: DTYPE_RANGE = torch.float8_e4m3fn,
            act_dtype: DTYPE_RANGE = torch.float8_e4m3fn,
            act_algo: Union[str, List[str]] = "minmax",
            approach: Union[str, List[str]] = "static",
            device: Union[str, List[str]] = "hpu",
            white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
        ):
            """Init FP8 config.

            Args:
            """
            super().__init__(white_list=white_list)
            self.weight_dtype = weight_dtype
            self.act_dtype = act_dtype
            self.act_algo = act_algo
            self.approach = approach
            self.device = device
            self._post_init()

        @classmethod
        def register_supported_configs(cls) -> List[OperatorConfig]:
            supported_configs = []
            fp8_config = FP8QConfig(
                weight_dtype=[torch.float8_e5m2, torch.float8_e4m3fn],
                act_dtype=[torch.float8_e5m2, torch.float8_e4m3fn],
                act_algo=["minmax", "kl"],
                approach=["static", "dynamic"],
                device=["hpu"],
            )
            from .fp8.quantization_impl import white_list

            operators = white_list
            supported_configs.append(OperatorConfig(config=fp8_config, operators=operators))
            cls.supported_configs = supported_configs

        @staticmethod
        def get_model_info(model: torch.nn.Module) -> List[Tuple[str, Callable]]:
            from .fp8.quantization_impl import white_list

            filter_result = []
            for op_name, module in model.named_modules():
                if isinstance(module, white_list):
                    pair = (op_name, type(module).__name__)
                    filter_result.append(pair)
            logger.debug(f"Get model info: {filter_result}")
            return filter_result

    # TODO(Yi) run `register_supported_configs` for all registered config.
    FP8QConfig.register_supported_configs()

    def get_default_fp8_qconfig() -> FP8QConfig:
        """Generate the default gptq config.

        Returns:
            the default gptq config.
        """
        return FP8QConfig()

    ##################### Algo Configs End ###################################


def get_all_registered_configs() -> Dict[str, BaseConfig]:
    registered_configs = config_registry.get_all_configs()
    return registered_configs.get(FRAMEWORK_NAME, {})