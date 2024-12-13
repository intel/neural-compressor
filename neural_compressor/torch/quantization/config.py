#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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
"""Intel Neural Compressor Pytorch quantization config API."""


import importlib
import json
from collections import OrderedDict
from typing import Callable, Dict, List, NamedTuple, Optional
from typing import OrderedDict as OrderedDictType
from typing import Tuple, Union

import torch

import neural_compressor.torch.utils as torch_utils
from neural_compressor.common.base_config import (
    BaseConfig,
    config_registry,
    register_config,
    register_supported_configs_for_fwk,
)
from neural_compressor.common.utils import (
    AUTOROUND,
    AWQ,
    DEFAULT_WHITE_LIST,
    FP8_QUANT,
    GPTQ,
    HQQ,
    MIXED_PRECISION,
    MX_QUANT,
    OP_NAME_OR_MODULE_TYPE,
    RTN,
    SMOOTH_QUANT,
    STATIC_QUANT,
    TEQ,
)
from neural_compressor.torch.utils import is_hpex_available, is_ipex_imported, is_transformers_imported, logger
from neural_compressor.torch.utils.constants import (
    LM_HEAD_NAMES,
    PRIORITY_AUTOROUND,
    PRIORITY_AWQ,
    PRIORITY_GPTQ,
    PRIORITY_HQQ,
    PRIORITY_RTN,
    PRIORITY_TEQ,
    PT2E_DYNAMIC_QUANT,
)

FRAMEWORK_NAME = "torch"
if is_transformers_imported():
    import transformers

    WOQ_WHITE_LIST = (torch.nn.Linear, transformers.Conv1D)
else:
    WOQ_WHITE_LIST = (torch.nn.Linear,)


class OperatorConfig(NamedTuple):
    """OperatorConfig."""

    config: BaseConfig
    operators: List[Union[str, Callable]]
    valid_func_list: List[Callable] = []


class TorchBaseConfig(BaseConfig):
    """Base config class for torch backend."""

    # re-write func _get_op_name_op_type_config to fallback op_type with string
    # because there are some special op_types for IPEX backend: `Linear&Relu`, `Linear&add`, ...
    def _get_op_name_op_type_config(self):
        op_type_config_dict = dict()
        op_name_config_dict = dict()
        for name, config in self.local_config.items():
            if self._is_op_type(name):
                # Convert the Callable to String.
                new_name = self._op_type_to_str(name)
                op_type_config_dict[new_name] = config
            else:
                op_name_config_dict[name] = config
                if is_ipex_imported():
                    op_type_config_dict[name] = config
        return op_type_config_dict, op_name_config_dict


######################## RNT Config ###############################
@register_config(framework_name=FRAMEWORK_NAME, algo_name=RTN, priority=PRIORITY_RTN)
class RTNConfig(TorchBaseConfig):
    """Config class for round-to-nearest weight-only quantization."""

    name = RTN
    params_list = [
        "dtype",
        "bits",
        "use_sym",
        "group_size",
        "group_dim",
        "use_full_range",
        "use_mse_search",
        # layer wise params
        "use_layer_wise",
        "model_path",
        # double quant
        "use_double_quant",
        "double_quant_dtype",
        "double_quant_bits",
        "double_quant_use_sym",
        "double_quant_group_size",
        # quant_lm_head
        "quant_lm_head",
    ]
    supported_configs: List[OperatorConfig] = []

    def __init__(
        self,
        dtype: str = "int",
        bits: int = 4,
        use_sym: bool = True,
        group_size: int = 32,
        group_dim: int = 1,
        use_full_range: bool = False,
        use_mse_search: bool = False,
        # layer wise
        use_layer_wise: bool = False,
        model_path: str = "",
        # double quant
        use_double_quant: bool = False,
        double_quant_dtype: str = "int",
        double_quant_bits: int = 8,  # not available when double_quant_dtype is not 'int'
        double_quant_use_sym: bool = False,
        double_quant_group_size: int = 256,
        # quant lm_head
        quant_lm_head: bool = False,
        # Tuning space
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
        **kwargs,
    ):
        """Init RTN weight-only quantization config.

        Args:
            dtype (str): Data type for weights. Default is "int".
            bits (int): Number of bits used to represent weights. Default is 4.
            use_sym (bool): Indicates whether weights are symmetric. Default is True.
            group_size (int): Size of weight groups. Default is 32.
            group_dim (int): Dimension for grouping. Default is 1.
            use_full_range (bool): Enables full range for activations. Default is False.
            use_mse_search (bool): Enables mean squared error (MSE) search. Default is False.
            use_layer_wise (bool): Enables quantize model per layer. Defaults to False.
            model_path (str): Model path that is used to load state_dict per layer.
            use_double_quant (bool): Enables double quantization. Default is False.
            double_quant_dtype (str): Data type for double_quant scale. Default is "int".
            double_quant_bits (int): Number of bits used to represent double_quant scale. Default is 4.
            double_quant_use_sym (bool): Indicates whether double_quant scale are symmetric. Default is True.
            double_quant_group_size (int): Size of double_quant groups. Default is 32.
            quant_lm_head (bool): Indicates whether quantize the lm_head layer in transformers。 Default is False.
            white_list (Optional[List[OP_NAME_OR_MODULE_TYPE]]): White list of operator names or module types.
                Default is DEFAULT_WHITE_LIST.
        """
        super().__init__(white_list=white_list)
        self.dtype = dtype
        self.bits = bits
        self.use_sym = use_sym
        self.group_size = group_size
        self.group_dim = group_dim
        self.use_full_range = use_full_range
        self.use_mse_search = use_mse_search
        self.use_layer_wise = use_layer_wise
        self.model_path = model_path
        # double quant
        self.use_double_quant = use_double_quant
        self.double_quant_bits = double_quant_bits
        self.double_quant_dtype = double_quant_dtype
        self.double_quant_use_sym = double_quant_use_sym
        self.double_quant_group_size = double_quant_group_size
        self.quant_lm_head = quant_lm_head
        self._post_init()  # initialize global & local configuration

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        """Register supported configurations for RTN.

        Returns:
            List[OperatorConfig]: List of supported operator configurations.
        """
        supported_configs = []
        linear_rtn_config = RTNConfig(
            dtype=[
                "int",
                "int8",
                "int4",
                "nf4",
                "fp4",
                "fp4_e2m1_bnb",
                "fp4_e2m1",
                "fp8_e5m2",
                "fp8_e5m2fnuz",
                "fp8_e4m3fn",
                "fp8_e4m3fnuz",
            ],
            bits=[4, 1, 2, 3, 5, 6, 7, 8],
            use_sym=[True, False],
            group_size=[32, -1, 1, 4, 8, 16, 64, 128, 256, 512, 1024],
            group_dim=[1, 0],
            use_full_range=[False, True],
            use_mse_search=[False, True],
            use_layer_wise=[False, True],
            use_double_quant=[False, True],
            double_quant_bits=[4, 1, 2, 3, 5, 6, 7, 8],
            double_quant_dtype=["int"],
            double_quant_use_sym=[True, False],
            double_quant_group_size=[32, -1, 1, 4, 8, 16, 64, 128, 256, 512, 1024],
            quant_lm_head=[False, True],
        )
        operators = list(WOQ_WHITE_LIST)
        supported_configs.append(OperatorConfig(config=linear_rtn_config, operators=operators))
        cls.supported_configs = supported_configs

    def to_config_mapping(
        self, config_list: List[BaseConfig] = None, model_info: List[Tuple[str, str]] = None
    ) -> OrderedDictType[Union[str, str], OrderedDictType[str, BaseConfig]]:
        """Convert the configuration to a mapping.

        Args:
            config_list (List[BaseConfig]): List of base configurations. Default is None.
            model_info (List[Tuple[str, str]]): List of tuples containing the name and type of each module in the model.
                Default is None.

        Returns:
            OrderedDictType[Union[str, str], OrderedDictType[str, BaseConfig]]: The configuration mapping.
        """
        if not self.quant_lm_head:
            self.set_local(
                LM_HEAD_NAMES, RTNConfig(dtype="fp32", use_layer_wise=self.use_layer_wise, model_path=self.model_path)
            )
        config_mapping = super().to_config_mapping(config_list, model_info)
        return config_mapping

    @staticmethod
    def get_model_info(model: torch.nn.Module) -> List[Tuple[str, Callable]]:
        """Get information about the model.

        Args:
            model (torch.nn.Module): The model.

        Returns:
            List[Tuple[str, Callable]]: List of tuples containing the name and type of each module in the model.
        """
        filter_result = []
        for op_name, module in model.named_modules():
            if isinstance(module, WOQ_WHITE_LIST):
                pair = (op_name, type(module).__name__)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "RTNConfig", List["RTNConfig"]]:
        """Get the default configuration set for tuning.

        Returns:
            Union[None, "RTNConfig", List["RTNConfig"]]: The configuration set for tuning.
        """
        return RTNConfig(
            dtype=["int4", "nf4"], use_sym=[True, False], group_size=[32, 128], use_mse_search=[False, True]
        )

    @classmethod
    def get_predefined_configs(cls) -> Dict[torch_utils.ProcessorType, "RTNConfig"]:
        """Get the predefined configuration set.

        Returns:
            Dict[torch_utils.ProcessorType, "RTNConfig"]: The configuration of RTN.
        """
        pre_defined_configs: Dict[torch_utils.ProcessorType, RTNConfig] = {}
        pre_defined_configs[torch_utils.ProcessorType.Client] = cls(use_layer_wise=True)
        pre_defined_configs[torch_utils.ProcessorType.Server] = cls()
        return pre_defined_configs


def get_default_rtn_config(processor_type: Optional[Union[str, torch_utils.ProcessorType]] = None) -> RTNConfig:
    """Get the default configuration of RTN.

    Args:
        processor_type (Optional[Union[str, torch_utils.ProcessorType]], optional): The user-specified processor type.
            Defaults to None.

    Returns:
        RTNConfig: RTNConfig
    """
    process_type = torch_utils.get_processor_type_from_user_config(processor_type)
    return RTNConfig.get_predefined_configs()[process_type]


def get_default_double_quant_config(type="BNB_NF4"):
    """Get the default configuration of double quant.

    Args:
        type (str, optional): double quant type. Defaults to "BNB_NF4".

    Returns:
        dict: double quant config.
    """
    from neural_compressor.torch.utils.constants import DOUBLE_QUANT_CONFIGS

    assert type in DOUBLE_QUANT_CONFIGS, "Supported double quant configs: {}".format(list(DOUBLE_QUANT_CONFIGS.keys()))
    return RTNConfig.from_dict(DOUBLE_QUANT_CONFIGS[type])


######################## GPTQ Config ###############################
@register_config(framework_name=FRAMEWORK_NAME, algo_name=GPTQ, priority=PRIORITY_GPTQ)
class GPTQConfig(TorchBaseConfig):
    """Config class for GPTQ.

    GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.
    https://arxiv.org/abs/2210.17323
    """

    name = GPTQ
    supported_configs: List[OperatorConfig] = []
    params_list = [
        "dtype",
        "bits",
        "use_sym",
        "group_size",
        "use_mse_search",
        "use_double_quant",
        "double_quant_dtype",
        "double_quant_bits",
        "double_quant_use_sym",
        "double_quant_group_size",
        # layer wise params
        "use_layer_wise",
        "model_path",
        # quant lm_head
        "quant_lm_head",
        # gptq params
        "act_order",
        "percdamp",
        "block_size",
        "static_groups",
        "true_sequential",
    ]

    def __init__(
        self,
        dtype: str = "int",
        bits: int = 4,
        use_sym: bool = True,
        group_size: int = 32,
        use_mse_search: bool = False,
        # layer wise
        use_layer_wise: bool = False,
        model_path: str = "",
        # double quant
        use_double_quant: bool = False,
        double_quant_dtype: str = "int",
        double_quant_bits: int = 8,  # not available when double_quant_dtype is not 'int'
        double_quant_use_sym: bool = False,
        double_quant_group_size: int = 256,
        # double quant
        quant_lm_head: bool = False,
        # gptq params
        act_order: bool = False,
        percdamp: float = 0.01,
        block_size: int = 2048,
        static_groups: bool = False,
        true_sequential: bool = False,
        # Tuning space
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
        **kwargs,
    ):
        """Init GPTQ weight-only quantization config.

        Args:
            dtype (str): Data type for weights. Default is "int".
            bits (int): Number of bits used to represent weights. Default is 4.
            use_sym (bool): Indicates whether weights are symmetric. Default is True.
            group_size (int): Size of weight groups. Default is 32.
            use_mse_search (bool): Enables mean squared error (MSE) search. Default is False.
            use_layer_wise (bool): Enables quantize model per layer. Defaults to False.
            model_path (str): Model path that is used to load state_dict per layer.
            use_double_quant (bool): Enables double quantization. Default is False.
            double_quant_dtype (str): Data type for double_quant scale. Default is "int".
            double_quant_bits (int): Number of bits used to represent double_quant scale. Default is 4.
            double_quant_use_sym (bool): Indicates whether double_quant scale are symmetric. Default is True.
            double_quant_group_size (int): Size of double_quant groups. Default is 32.
            quant_lm_head (bool): Indicates whether quantize the lm_head layer in transformers。 Default is False.
            act_order (bool): Whether to sort Hessian's diagonal values to rearrange channel-wise
                              quantization order. Default is False.
            percdamp (float): Percentage of Hessian's diagonal values' average, which will be added to
                              Hessian's diagonal to increase numerical stability. Default is 0.01.
            block_size (int): Execute GPTQ quantization per block, block shape = [C_out, block_size].
                              Default is 128.
            static_groups (bool): Whether to calculate group wise quantization parameters in advance.
                                  This option mitigate actorder's extra computational requirements.
                                  Default is False.
            true_sequential (bool): Whether to quantize layers within a transformer block in their original order.
                                  This can lead to higher accuracy but slower overall quantization process.
                                  Default is False.
            white_list (Optional[List[OP_NAME_OR_MODULE_TYPE]]): White list of operator names or module types.
                                                                 Default is DEFAULT_WHITE_LIST.
        """
        super().__init__(white_list=white_list)
        self.dtype = dtype
        self.bits = bits
        self.use_sym = use_sym
        self.group_size = group_size
        self.use_mse_search = use_mse_search
        # layer wise
        self.use_layer_wise = use_layer_wise
        self.model_path = model_path
        # double quant
        self.use_double_quant = use_double_quant
        self.double_quant_bits = double_quant_bits
        self.double_quant_dtype = double_quant_dtype
        self.double_quant_use_sym = double_quant_use_sym
        self.double_quant_group_size = double_quant_group_size
        # gptq
        self.act_order = act_order
        self.percdamp = percdamp
        self.block_size = block_size
        self.static_groups = static_groups
        self.true_sequential = true_sequential
        self.quant_lm_head = quant_lm_head
        self._post_init()  # initialize global & local configuration

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        """Register supported configurations for GPTQ.

        Returns:
            List[OperatorConfig]: List of supported operator configurations.
        """
        supported_configs = []
        # TODO(Yi)
        linear_gptq_config = GPTQConfig()
        operators = list(WOQ_WHITE_LIST)
        supported_configs.append(OperatorConfig(config=linear_gptq_config, operators=operators))
        cls.supported_configs = supported_configs

    def to_config_mapping(
        self, config_list: List[BaseConfig] = None, model_info: List[Tuple[str, str]] = None
    ) -> OrderedDictType[Union[str, str], OrderedDictType[str, BaseConfig]]:
        """Convert the configuration to a mapping.

        Args:
            config_list (List[BaseConfig]): List of base configurations. Default is None.
            model_info (List[Tuple[str, str]]): List of tuples containing the name and type of each module in the model.
                Default is None.

        Returns:
            OrderedDictType[Union[str, str], OrderedDictType[str, BaseConfig]]: The configuration mapping.
        """
        if not self.quant_lm_head:
            self.set_local(
                LM_HEAD_NAMES, GPTQConfig(dtype="fp32", use_layer_wise=self.use_layer_wise, model_path=self.model_path)
            )
        config_mapping = super().to_config_mapping(config_list, model_info)
        return config_mapping

    @staticmethod
    def get_model_info(model: torch.nn.Module) -> List[Tuple[str, Callable]]:
        """Get information about the model.

        Args:
            model (torch.nn.Module): The model.

        Returns:
            List[Tuple[str, Callable]]: List of tuples containing the name and type of each module in the model.
        """
        filter_result = []
        for op_name, module in model.named_modules():
            if isinstance(module, WOQ_WHITE_LIST):
                pair = (op_name, type(module).__name__)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "GPTQConfig", List["GPTQConfig"]]:
        """Get the default configuration set for tuning.

        Returns:
            Union[None, "GPTQConfig", List["GPTQConfig"]]: The configuration set for tuning.
        """
        # TODO fwk owner needs to update it.
        return GPTQConfig(act_order=[True, False], use_sym=[False, True])

    @classmethod
    def get_predefined_configs(cls) -> Dict[torch_utils.ProcessorType, "GPTQConfig"]:
        """Get the predefined configuration set.

        Returns:
            Dict[torch_utils.ProcessorType, "GPTQConfig"]: The configuration of GPTQ.
        """
        pre_defined_configs: Dict[torch_utils.ProcessorType, GPTQConfig] = {}
        pre_defined_configs[torch_utils.ProcessorType.Client] = cls(
            use_layer_wise=True
        )  # , model_path=self.model_path)
        pre_defined_configs[torch_utils.ProcessorType.Server] = cls()
        return pre_defined_configs


def get_default_gptq_config(processor_type: Optional[Union[str, torch_utils.ProcessorType]] = None) -> GPTQConfig:
    """Get the default configuration of GPTQ.

    Args:
        processor_type (Optional[Union[str, torch_utils.ProcessorType]], optional): The user-specified processor type.
            Defaults to None.

    Returns:
        GPTQConfig: GPTQConfig
    """
    process_type = torch_utils.get_processor_type_from_user_config(processor_type)
    return GPTQConfig.get_predefined_configs()[process_type]


######################## AWQ Config ###############################
@register_config(framework_name=FRAMEWORK_NAME, algo_name=AWQ, priority=PRIORITY_AWQ)
class AWQConfig(TorchBaseConfig):
    """Config class for AWQ.

    AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.
    https://arxiv.org/abs/2306.00978
    """

    supported_configs: List[OperatorConfig] = []
    params_list = [
        "dtype",
        "bits",
        "group_size",
        "group_dim",
        "use_sym",
        "use_full_range",
        "use_mse_search",
        "use_layer_wise",
        "use_double_quant",
        "double_quant_dtype",
        "double_quant_bits",
        "double_quant_use_sym",
        "double_quant_group_size",
        # quant_lm_head
        "quant_lm_head",
        # AWQ params
        "use_auto_scale",
        "use_auto_clip",
        "folding",
        "absorb_layer_dict",
    ]
    name = AWQ

    def __init__(
        self,
        dtype: str = "int",
        bits: int = 4,
        use_sym: bool = True,
        group_size: int = 32,
        group_dim: int = 1,
        use_full_range: bool = False,
        use_mse_search: bool = False,
        use_layer_wise: bool = False,
        model_path: str = "",
        # double quant
        use_double_quant: bool = False,
        double_quant_dtype: str = "int",
        double_quant_bits: int = 8,  # not available when double_quant_dtype is not 'int'
        double_quant_use_sym: bool = True,
        double_quant_group_size: int = 256,
        # quant lm_head
        quant_lm_head: bool = False,
        # awq
        use_auto_scale: bool = True,
        use_auto_clip: bool = True,
        folding: bool = False,
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
        absorb_layer_dict: dict = {},
        **kwargs,
    ):
        """Init AWQ weight-only quantization config.

        Args:
            dtype (str): Data type for weights, default is "int".
            bits (int): Number of bits used to represent weights, default is 4.
            use_sym (bool): Indicates whether weights are symmetric, default is True.
            group_size (int): Size of weight groups, default is 32.
            group_dim (int): Dimension for grouping, default is 1.
            use_full_range (bool): Enables full range for activations, default is False.
            use_mse_search (bool): Enables mean squared error (MSE) search, default is False.
            use_layer_wise (bool): Enables quantize model per layer. Defaults to False.
            model_path (str): Model path that is used to load state_dict per layer.
            use_double_quant (bool): Enables double quantization, default is False.
            double_quant_dtype (str): Data type for double_quant scale, default is "int".
            double_quant_bits (int): Number of bits used to represent double_quant scale, default is 4.
            double_quant_use_sym (bool): Indicates whether double_quant scale are symmetric, default is True.
            double_quant_group_size (int): Size of double_quant groups, default is 32.
            quant_lm_head (bool): Indicates whether quantize the lm_head layer in transformer, default is False.
            use_auto_scale (bool): Enables best scales search based on activation distribution, default is True.
            use_auto_clip (bool):  Enables clip range search. Defaults to True.
            folding(bool): Allow insert mul before linear when the scale cannot be absorbed by last layer,
              default is False.
            absorb_layer_dict (dict): The layer dict that scale can be absorbed, default is {}.
            white_list (Optional[List[OP_NAME_OR_MODULE_TYPE]]): White list of operator names or module types.
              Default is DEFAULT_WHITE_LIST.
        """
        super().__init__(white_list=white_list)
        self.dtype = dtype
        self.bits = bits
        self.use_sym = use_sym
        self.group_size = group_size
        self.group_dim = group_dim
        self.use_full_range = use_full_range
        self.use_mse_search = use_mse_search
        self.use_layer_wise = use_layer_wise
        self.model_path = model_path
        # double quant
        self.use_double_quant = use_double_quant
        self.double_quant_bits = double_quant_bits
        self.double_quant_dtype = double_quant_dtype
        self.double_quant_use_sym = double_quant_use_sym
        self.double_quant_group_size = double_quant_group_size
        self.quant_lm_head = quant_lm_head
        self.use_auto_scale = use_auto_scale
        self.use_auto_clip = use_auto_clip
        self.folding = folding
        self.absorb_layer_dict = absorb_layer_dict
        self._post_init()

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        """Register supported configurations for AWQ.

        Returns:
            List[OperatorConfig]: List of supported operator configurations.
        """
        supported_configs = []
        # TODO(Yi)
        linear_awq_config = AWQConfig()
        operators = list(WOQ_WHITE_LIST)
        supported_configs.append(OperatorConfig(config=linear_awq_config, operators=operators))
        cls.supported_configs = supported_configs

    def to_config_mapping(
        self, config_list: List[BaseConfig] = None, model_info: List[Tuple[str, str]] = None
    ) -> OrderedDictType[Union[str, str], OrderedDictType[str, BaseConfig]]:
        """Convert the configuration to a mapping.

        Args:
            config_list (List[BaseConfig]): List of base configurations. Default is None.
            model_info (List[Tuple[str, str]]): List of tuples containing the name and type of each module in the model.
                Default is None.

        Returns:
            OrderedDictType[Union[str, str], OrderedDictType[str, BaseConfig]]: The configuration mapping.
        """
        if not self.quant_lm_head:
            self.set_local(
                LM_HEAD_NAMES, AWQConfig(dtype="fp32", use_layer_wise=self.use_layer_wise, model_path=self.model_path)
            )
        config_mapping = super().to_config_mapping(config_list, model_info)
        return config_mapping

    @staticmethod
    def get_model_info(model: torch.nn.Module) -> List[Tuple[str, Callable]]:
        """Get information about the model.

        Args:
            model (torch.nn.Module): The model.

        Returns:
            List[Tuple[str, Callable]]: List of tuples containing the name and type of each module in the model.
        """
        filter_result = []
        for op_name, module in model.named_modules():
            if isinstance(module, WOQ_WHITE_LIST):
                pair = (op_name, type(module).__name__)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "AWQConfig", List["AWQConfig"]]:
        """Get the default configuration set for tuning.

        Returns:
            Union[None, "AWQConfig", List["AWQConfig"]]: The configuration set for tuning.
        """
        # TODO fwk owner needs to update it.
        return AWQConfig(bits=[4, 6])


def get_default_awq_config() -> AWQConfig:
    """Generate the default awq config.

    Returns:
        the default awq config.
    """
    return AWQConfig()


######################## TEQ Config ###############################
@register_config(framework_name=FRAMEWORK_NAME, algo_name=TEQ, priority=PRIORITY_TEQ)
class TEQConfig(TorchBaseConfig):
    """Config class for TEQ.

    TEQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.
    https://arxiv.org/abs/2306.00978
    """

    supported_configs: List[OperatorConfig] = []
    params_list = [
        "dtype",
        "bits",
        "group_size",
        "group_dim",
        "use_sym",
        "use_full_range",
        "use_mse_search",
        "use_layer_wise",
        "use_double_quant",
        "double_quant_dtype",
        "double_quant_bits",
        "double_quant_use_sym",
        "double_quant_group_size",
        # quant_lm_head
        "quant_lm_head",
        # TEQ params
        "absorb_to_layer",
        "folding",
    ]
    name = TEQ

    def __init__(
        self,
        dtype: str = "int",
        bits: int = 4,
        use_sym: bool = True,
        group_size: int = 32,
        group_dim: int = 1,
        use_full_range: bool = False,
        use_mse_search: bool = False,
        use_layer_wise: bool = False,
        # double quant
        use_double_quant: bool = False,
        double_quant_dtype: str = "int",
        double_quant_bits: int = 8,  # not available when double_quant_dtype is not 'int'
        double_quant_use_sym: bool = True,
        double_quant_group_size: int = 256,
        # quant lm_head
        quant_lm_head: bool = False,
        # teq
        absorb_to_layer: dict = {},
        folding: bool = True,
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
        **kwargs,
    ):
        """Init TEQ weight-only quantization config.

        Args:
            dtype (str): Data type for weights, default is "int".
            bits (int): Number of bits used to represent weights, default is 4.
            use_sym (bool): Indicates whether weights are symmetric, default is True.
            group_size (int): Size of weight groups, default is 32.
            group_dim (int): Dimension for grouping, default is 1.
            use_full_range (bool): Enables full range for activations, default is False.
            use_mse_search (bool): Enables mean squared error (MSE) search, default is False.
            use_layer_wise (bool): Enables quantize model per layer. Defaults to False.
            use_double_quant (bool): Enables double quantization, default is False.
            double_quant_dtype (str): Data type for double_quant scale, default is "int".
            double_quant_bits (int): Number of bits used to represent double_quant scale, default is 4.
            double_quant_use_sym (bool): Indicates whether double_quant scale are symmetric, default is True.
            double_quant_group_size (int): Size of double_quant groups, default is 32.
            quant_lm_head (bool): Indicates whether quantize the lm_head layer in transformers。 Default is False.
            absorb_to_layer (dict): The layer dict that scale can be absorbed, default is {}.
            folding(bool): Allow insert mul before linear when the scale cannot be absorbed by last layer,
              default is False.
            white_list (Optional[List[OP_NAME_OR_MODULE_TYPE]]): White list of operator names or module types.
              Default is DEFAULT_WHITE_LIST.
        """
        super().__init__(white_list=white_list)
        self.dtype = dtype
        self.bits = bits
        self.use_sym = use_sym
        self.group_size = group_size
        self.group_dim = group_dim
        self.use_full_range = use_full_range
        self.use_mse_search = use_mse_search
        self.use_layer_wise = use_layer_wise
        # double quant
        self.use_double_quant = use_double_quant
        self.double_quant_bits = double_quant_bits
        self.double_quant_dtype = double_quant_dtype
        self.double_quant_use_sym = double_quant_use_sym
        self.double_quant_group_size = double_quant_group_size
        self.quant_lm_head = quant_lm_head
        self.absorb_to_layer = absorb_to_layer
        self.folding = folding
        self._post_init()

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        """Register supported configurations for TEQ.

        Returns:
            List[OperatorConfig]: List of supported operator configurations.
        """
        supported_configs = []
        # TODO(Yi)
        linear_teq_config = TEQConfig()
        operators = list(WOQ_WHITE_LIST)
        supported_configs.append(OperatorConfig(config=linear_teq_config, operators=operators))
        cls.supported_configs = supported_configs

    def to_config_mapping(
        self, config_list: List[BaseConfig] = None, model_info: List[Tuple[str, str]] = None
    ) -> OrderedDictType[Union[str, str], OrderedDictType[str, BaseConfig]]:
        """Convert the configuration to a mapping.

        Args:
            config_list (List[BaseConfig]): List of base configurations. Default is None.
            model_info (List[Tuple[str, str]]): List of tuples containing the name and type of each module in the model.
                Default is None.

        Returns:
            OrderedDictType[Union[str, str], OrderedDictType[str, BaseConfig]]: The configuration mapping.
        """
        if not self.quant_lm_head:
            self.set_local(LM_HEAD_NAMES, TEQConfig(dtype="fp32"))
        config_mapping = super().to_config_mapping(config_list, model_info)
        return config_mapping

    @staticmethod
    def get_model_info(model: torch.nn.Module) -> List[Tuple[str, Callable]]:
        """Get information about the model.

        Args:
            model (torch.nn.Module): The model.

        Returns:
            List[Tuple[str, Callable]]: List of tuples containing the name and type of each module in the model.
        """
        filter_result = []
        for op_name, module in model.named_modules():
            if isinstance(module, WOQ_WHITE_LIST):
                pair = (op_name, type(module).__name__)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "TEQConfig", List["TEQConfig"]]:
        """Get the default configuration set for tuning.

        Returns:
            Union[None, "TEQConfig", List["TEQConfig"]]: The configuration set for tuning.
        """
        # TODO fwk owner needs to update it.
        return TEQConfig(bits=[4, 6])


def get_default_teq_config() -> TEQConfig:
    """Generate the default teq config.

    Returns:
        the default teq config.
    """
    return TEQConfig()


######################## AUTOROUND Config ###############################
@register_config(framework_name=FRAMEWORK_NAME, algo_name=AUTOROUND, priority=PRIORITY_AUTOROUND)
class AutoRoundConfig(TorchBaseConfig):
    """Config class for AUTOROUND.

    AUTOROUND: Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs.
    https://arxiv.org/abs/2309.05516
    code: https://github.com/intel/auto-round
    """

    supported_configs: List[OperatorConfig] = []
    params_list = [
        "dtype",
        "bits",
        "group_size",
        "use_sym",
        # autoround params
        "enable_full_range",
        "batch_size",
        "enable_minmax_tuning",
        "lr",
        "minmax_lr",
        "iters",
        "seqlen",
        "nsamples",
        "nblocks",
        "gradient_accumulate_steps",
        "not_use_best_mse",
        "dynamic_max_gap",
    ]
    name = AUTOROUND

    def __init__(
        self,
        dtype: str = "int",
        bits: int = 4,
        use_sym: bool = False,
        group_size: int = 128,
        # AUTOROUND
        act_bits: int = 32,
        act_group_size: int = None,
        act_sym: bool = None,
        act_dynamic: bool = True,
        enable_full_range: bool = False,
        batch_size: int = 8,
        lr_scheduler=None,
        enable_quanted_input: bool = True,
        enable_minmax_tuning: bool = True,
        lr: float = None,
        minmax_lr: float = None,
        low_gpu_mem_usage: bool = False,
        iters: int = 200,
        seqlen: int = 2048,
        nsamples: int = 128,
        sampler: str = "rand",
        seed: int = 42,
        nblocks: int = 1,
        gradient_accumulate_steps: int = 1,
        not_use_best_mse: bool = False,
        dynamic_max_gap: int = -1,
        scale_dtype: str = "fp16",
        use_layer_wise: bool = False,
        to_quant_block_names: list = None,
        export_format: str = "itrex",
        # v0.4
        enable_norm_bias_tuning: bool = False,
        enable_torch_compile: bool = None,
        # mllm
        is_mllm: bool = False,
        quant_nontext_module: Union[str, list] = None,
        extra_data_dir: str = None,
        processor=None,
        image_processor=None,
        template=None,
        truncation: bool = False,
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
        **kwargs,
    ):
        """Init AUTOROUND weight-only quantization config.

        Args:
            dtype (str): Data type for weights, default is "int".
            bits (int): Number of bits used to represent weights, default is 4.
            use_sym (bool): Indicates whether weights are symmetric, default is False.
            group_size (int): Size of weight groups, default is 128.
            act_bits (int): Number of bits for activation quantization. Default is 32.
            act_group_size (int): Group size for activation quantization. Default is None.
            act_sym (bool): Whether to use symmetric activation quantization. Default is None.
            act_dynamic (bool): Whether to use dynamic activation quantization. Default is True.
            enable_full_range (bool): Whether to enable full range quantization (default is False).
            batch_size (int): Batch size for training (default is 8).
            lr_scheduler: The learning rate scheduler to be used.
            enable_quanted_input (bool): Whether to use quantized input data (default is True).
            enable_minmax_tuning (bool): Whether to enable min-max tuning (default is True).
            lr (float): The learning rate (default is 0.005).
            minmax_lr (float): The learning rate for min-max tuning (default is None).
            low_gpu_mem_usage (bool): Whether to use low GPU memory (default is False).
            iters (int): Number of iterations (default is 200).
            seqlen (int): Length of the sequence.
            nsamples (int): Number of samples (default is 512).
            sampler (str): The sampling method (default is "rand").
            seed (int): The random seed (default is 42).
            nblocks (int): Number of blocks (default is 1).
            gradient_accumulate_steps (int): Number of gradient accumulation steps (default is 1).
            not_use_best_mse (bool): Whether to use mean squared error (default is False).
            dynamic_max_gap (int): The dynamic maximum gap (default is -1).
            scale_dtype (str): The data type of quantization scale to be used (default is "float16"), different kernels
              have different choices.
            use_layer_wise (bool): Enables quantize model per layer. Defaults to False.
            to_quant_block_names (list): A list whose elements are list of block's layer names to be quantized.
            export_format (str, optional): The format used for exporting the quantized model. Defaults to "itrex".
            enable_norm_bias_tuning (bool): Whether to enable fast norm/layer_bias tuning.
            enable_torch_compile (bool): Whether to enable torch compile to optimize quant_block/layer, torch>=2.6 True.
            quant_nontext_module (Union[str, list]): Whether to quantize nontext module.
            extra_data_dir (str): The path for extra data such as images, audio or videos.
            is_mllm (bool): Indicates whether the model to be quantized is a multi-modal model (MLLM).
            processor (transformers.AutoProcessor): Any multi-modal model will require an object to encode or
              decode the data that groups several modalities (among text, vision and audio).
              This is handled by objects called processors, which group together two or more processing objects such
              as tokenizers (for the text modality), image processors (for vision) and feature extractors (for audio).
            image_processor (Processor): Image processor for special model like llava.
            template (Template): The template to specify process for different mllms.
            truncation (bool): Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            white_list (Optional[List[OP_NAME_OR_MODULE_TYPE]]): White list of operator names or module types.
              Default is DEFAULT_WHITE_LIST.
        """
        super().__init__(white_list=white_list)
        self.dtype = dtype
        self.bits = bits
        self.use_sym = use_sym
        self.group_size = group_size
        self.act_bits = act_bits
        self.act_group_size = act_group_size
        self.act_sym = act_sym
        self.act_dynamic = act_dynamic
        self.enable_full_range = enable_full_range
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.enable_quanted_input = enable_quanted_input
        self.enable_minmax_tuning = enable_minmax_tuning
        self.lr = lr
        self.minmax_lr = minmax_lr
        self.low_gpu_mem_usage = low_gpu_mem_usage
        self.iters = iters
        self.seqlen = seqlen
        self.nsamples = nsamples
        self.sampler = sampler
        self.seed = seed
        self.nblocks = nblocks
        self.gradient_accumulate_steps = gradient_accumulate_steps
        self.not_use_best_mse = not_use_best_mse
        self.dynamic_max_gap = dynamic_max_gap
        self.scale_dtype = scale_dtype
        self.use_layer_wise = use_layer_wise
        self.to_quant_block_names = to_quant_block_names
        self.export_format = export_format
        self.enable_norm_bias_tuning = enable_norm_bias_tuning
        self.enable_torch_compile = enable_torch_compile
        self.is_mllm = is_mllm
        self.quant_nontext_module = quant_nontext_module
        self.extra_data_dir = extra_data_dir
        self.processor = processor
        self.image_processor = image_processor
        self.template = template
        self.truncation = truncation
        self._post_init()

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        """Register supported configurations for AutoRound.

        Returns:
            List[OperatorConfig]: List of supported operator configurations.
        """
        supported_configs = []
        # TODO(Yi)
        linear_AUTOROUND_config = AutoRoundConfig()
        operators = [torch.nn.Linear, torch.nn.functional.linear]
        supported_configs.append(OperatorConfig(config=linear_AUTOROUND_config, operators=operators))
        cls.supported_configs = supported_configs

    @staticmethod
    def get_model_info(model: torch.nn.Module) -> List[Tuple[str, Callable]]:
        """Get information about the model.

        Args:
            model (torch.nn.Module): The model.

        Returns:
            List[Tuple[str, Callable]]: List of tuples containing the name and type of each module in the model.
        """
        filter_result = []
        for op_name, module in model.named_modules():
            if isinstance(module, WOQ_WHITE_LIST):
                pair = (op_name, type(module).__name__)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "AutoRoundConfig", List["AutoRoundConfig"]]:
        """Get the default configuration set for tuning.

        Returns:
            Union[None, "AutoRoundConfig", List["AutoRoundConfig"]]: The configuration set for tuning.
        """
        # TODO fwk owner needs to update it.
        return AutoRoundConfig(bits=[4, 6])

    @classmethod
    def get_predefined_configs(cls) -> Dict[torch_utils.ProcessorType, "AutoRoundConfig"]:
        """Get the predefined configuration set.

        Returns:
            Dict[torch_utils.ProcessorType, "AutoRoundConfig"]: The configuration of AutoRound.
        """
        pre_defined_configs: Dict[torch_utils.ProcessorType, AutoRoundConfig] = {}
        pre_defined_configs[torch_utils.ProcessorType.Client] = cls(use_layer_wise=True)
        pre_defined_configs[torch_utils.ProcessorType.Server] = cls()
        return pre_defined_configs


def get_default_AutoRound_config(processor_type: Optional[Union[str, torch_utils.ProcessorType]] = None) -> RTNConfig:
    """Get the default configuration of AutoRound.

    Args:
        processor_type (Optional[Union[str, torch_utils.ProcessorType]], optional): The user-specified processor type.
            Defaults to None.

    Returns:
        AutoRoundConfig: AutoRoundConfig
    """
    process_type = torch_utils.get_processor_type_from_user_config(processor_type)
    return AutoRoundConfig.get_predefined_configs()[process_type]


######################## MX Config ###############################
@register_config(framework_name=FRAMEWORK_NAME, algo_name=MX_QUANT)
class MXQuantConfig(TorchBaseConfig):
    """Config class for MX quantization."""

    supported_configs: List[OperatorConfig] = []
    params_list = [
        "w_dtype",
        "act_dtype",
        "out_dtype",
        "blocksize",
        "round_method",
        "weight_only",
    ]
    name = MX_QUANT

    def __init__(
        self,
        w_dtype: str = "int8",
        act_dtype: str = "int8",
        out_dtype: str = "bfloat16",
        blocksize: int = 32,
        round_method: str = "nearest",
        weight_only: bool = False,
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
        **kwargs,
    ):
        """Init MX quantization config.

        Args:
            w_dtype (str): Data type for weights, default is "int8".
            act_dtype (str): Data type for activations, default is "int8".
            out_dtype (str): Data type for outputs, default is "bfloat16".
            blocksize (int): Granularity to share the scale, default is 32.
            round_method (str): Round method, default is "nearest".
            weight_only (bool): Whether implement weight_only, default is False.
            white_list (Optional[List[OP_NAME_OR_MODULE_TYPE]]): White list of operator names or module types.
              Default is DEFAULT_WHITE_LIST.
        """
        super().__init__(white_list=white_list)
        self.w_dtype = w_dtype
        self.act_dtype = act_dtype
        self.out_dtype = out_dtype
        self.blocksize = blocksize
        self.round_method = round_method
        self.weight_only = weight_only
        self._post_init()

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        """Register supported configurations."""
        supported_configs = []
        linear_mx_config = MXQuantConfig(
            w_dtype=[
                "int8",
                "int4",
                "int2",
                "fp8_e5m2",
                "fp8_e4m3",
                "fp6_e3m2",
                "fp6_e2m3",
                "fp4",
                "float16",
                "bfloat16",
                "float32",
            ],
            act_dtype=[
                "int8",
                "int4",
                "int2",
                "fp8_e5m2",
                "fp8_e4m3",
                "fp6_e3m2",
                "fp6_e2m3",
                "fp4",
                "float16",
                "bfloat16",
                "float32",
            ],
            out_dtype=["bfloat16", "float16", "float32"],
            blocksize=[2, 4, 8, 16, 32, 64, 128, 256, 512],
            round_method=["nearest", "dither", "floor", "even"],
            weight_only=[True, False],
        )
        operators = [torch.nn.Linear, torch.nn.functional.linear]
        supported_configs.append(OperatorConfig(config=linear_mx_config, operators=operators))
        cls.supported_configs = supported_configs

    @staticmethod
    def get_model_info(model: torch.nn.Module) -> List[Tuple[str, Callable]]:
        """Get information about the model.

        Args:
            model (torch.nn.Module): The model.

        Returns:
            List[Tuple[str, Callable]]: List of tuples containing the name and type of each module in the model.
        """
        white_list = (
            torch.nn.Linear,
            torch.nn.functional.linear,
        )

        filter_result = []
        for op_name, module in model.named_modules():
            if module.__class__ in white_list:
                pair = (op_name, type(module).__name__)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "MXQuantConfig", List["MXQuantConfig"]]:
        """Get the default configuration set for tuning."""
        return MXQuantConfig(weight_only=[False, True])


def get_default_mx_config() -> MXQuantConfig:
    """Generate the default mx config.

    Returns:
        the default rtn config.
    """
    return MXQuantConfig()


######################## Dynamic Quant Config ###############################
@register_config(framework_name=FRAMEWORK_NAME, algo_name=PT2E_DYNAMIC_QUANT)
class DynamicQuantConfig(TorchBaseConfig):
    """Config class for dynamic quantization."""

    name = PT2E_DYNAMIC_QUANT
    params_list = [
        "w_dtype",
        "w_sym",
        "w_granularity",
        "w_algo",
        "act_dtype",
        "act_sym",
        "act_granularity",
        "act_algo",
    ]
    supported_configs: List[OperatorConfig] = []

    def __init__(
        self,
        w_dtype: str = "int8",
        w_sym: bool = True,
        w_granularity: str = "per_tensor",
        w_algo: str = "minmax",
        act_dtype: str = "uint8",
        act_sym: bool = False,
        act_granularity: str = "per_tensor",
        act_algo: str = "kl",
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
        **kwargs,
    ):
        """Init Dynamic Quant Configs."""
        super().__init__(white_list=white_list)
        self.w_dtype = w_dtype
        self.w_sym = w_sym
        self.w_granularity = w_granularity
        self.w_algo = w_algo
        self.act_dtype = act_dtype
        self.act_sym = act_sym
        self.act_granularity = act_granularity
        self.act_algo = act_algo
        self._post_init()

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        """Register supported configurations."""
        supported_configs = []
        linear_static_config = cls()
        operators = [torch.nn.Linear]
        supported_configs.append(OperatorConfig(config=linear_static_config, operators=operators))
        cls.supported_configs = supported_configs

    @staticmethod
    def get_model_info(model: torch.nn.Module, example_inputs=None):
        """Get information about the model.

        Args:
            model (torch.nn.Module): The model.

        Returns:
            List[Tuple[str, Callable]]: List of tuples containing the name and type of each module in the model.
        """
        return None

    def to_config_mapping(
        self, config_list: List[BaseConfig] = None, model_info: List[Tuple[str, str]] = None
    ) -> OrderedDictType[Union[str, str], OrderedDictType[str, BaseConfig]]:
        """Convert the configuration to a mapping.

        Args:
            config_list (List[BaseConfig]): List of base configurations. Default is None.
            model_info (List[Tuple[str, str]]): List of tuples containing the name and type of each module in the model.
                Default is None.

        Returns:
            OrderedDictType[Union[str, str], OrderedDictType[str, BaseConfig]]: The configuration mapping.
        """
        config_mapping = OrderedDict({self.name: self})
        return config_mapping

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "DynamicQuantConfig", List["DynamicQuantConfig"]]:
        """Get the default configuration set for tuning."""
        return cls(act_sym=[True, False], act_algo=["kl", "minmax"])


def get_default_dynamic_config() -> DynamicQuantConfig:
    """Generate the default dynamic quant config.

    Returns:
        the default dynamic quant config.
    """
    return DynamicQuantConfig()


######################## Static Quant Config ###############################
@register_config(framework_name=FRAMEWORK_NAME, algo_name=STATIC_QUANT)
class INT8StaticQuantConfig(TorchBaseConfig):
    """Config class for static quantization."""

    name = STATIC_QUANT
    params_list = [
        "w_dtype",
        "w_sym",
        "w_granularity",
        "w_algo",
        "act_dtype",
        "act_sym",
        "act_granularity",
        "act_algo",
        "excluded_precisions",
    ]
    supported_configs: List[OperatorConfig] = []

    def __init__(
        self,
        w_dtype: str = "int8",
        w_sym: bool = True,
        w_granularity: str = "per_channel",
        w_algo: str = "minmax",
        act_dtype: str = "uint8",
        act_sym: bool = False,
        act_granularity: str = "per_tensor",
        act_algo: str = "minmax",
        excluded_precisions: list = [],
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
        model_info: Optional[List[Tuple[str, Callable]]] = None,
        **kwargs,
    ):
        """Init StaticQuant Config.

        Args:
            w_dtype (str): Data type for weights, default is "int8".
            w_sym (bool): Whether to use symmetric quantization for weights, default is True.
            w_granularity (str): Level of quantization granularity for weights, default is "per_channel".
            w_algo (str): Quatization algorithm used to compute parameters for weights, default is "minmax".
            act_dtype (str): Data type for activations, default is "uint8".
            act_sym (bool): Whether to use symmetric quantization for activations, default is False.
            act_granularity (str): Level of quantization granularity for activations, default is "per_channel".
            act_algo (str): Quatization algorithm used to compute parameters for activations, default is "minmax".
            excluded_precisions (list): Precisions to be excluded, Default value is empty list.
            white_list (Optional[List[OP_NAME_OR_MODULE_TYPE]]): White list of operator names or module types.
                                                                 Default is DEFAULT_WHITE_LIST.
            model_info (Optional): used to keep model info for XPU device.  # TODO: should be removed from input arguments
        """
        super().__init__(white_list=white_list)
        self.w_dtype = w_dtype
        self.w_sym = w_sym
        self.w_granularity = w_granularity
        self.w_algo = w_algo
        self.act_dtype = act_dtype
        self.act_sym = act_sym
        self.act_granularity = act_granularity
        self.act_algo = act_algo
        self.excluded_precisions = excluded_precisions
        self.model_info = model_info
        self._post_init()

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        """Register supported configurations."""
        supported_configs = []
        linear_static_config = INT8StaticQuantConfig()
        operators = [torch.nn.Linear]
        supported_configs.append(OperatorConfig(config=linear_static_config, operators=operators))
        cls.supported_configs = supported_configs

    @staticmethod
    def get_model_info_for_ipex(model: torch.nn.Module, example_inputs) -> List[Tuple[str, Callable]]:
        """Get information about the model.

        Args:
            model (torch.nn.Module): The model.

        Returns:
            List[Tuple[str, Callable]]: List of tuples containing the name and type of each module in the model.
        """
        from neural_compressor.torch.algorithms.static_quant import get_quantizable_ops_recursively

        _, _, _, _, model_info = get_quantizable_ops_recursively(model, example_inputs=example_inputs)
        return model_info

    def get_model_info_for_ipex_xpu(self, model: torch.nn.Module) -> List[Tuple[str, Callable]]:  # pragma: no cover
        """Get information about the model.

        Args:
            model (torch.nn.Module): The model.

        Returns:
            List[Tuple[str, Callable]]: List of tuples containing the name and type of each module in the model.
        """
        if self.model_info:
            return self.model_info
        else:
            white_list = torch.quantization.quantization_mappings.get_default_qconfig_propagation_list()
            filter_result = []
            for op_name, module in model.named_modules():
                if type(module) in white_list:
                    pair = (op_name, type(module).__name__)
                    filter_result.append(pair)
            logger.debug(f"Get model info: {filter_result}")
            self.model_info = filter_result
            return filter_result

    def get_model_info(self, model: torch.nn.Module, example_inputs=None) -> List[Tuple[str, Callable]]:
        """Get information about the model.

        Args:
            model (torch.nn.Module): The model.

        Returns:
            List[Tuple[str, Callable]]: List of tuples containing the name and type of each module in the model.
        """
        from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator

        if is_ipex_imported():
            if auto_detect_accelerator().current_device() == "cpu":
                return INT8StaticQuantConfig.get_model_info_for_ipex(model, example_inputs)
            else:
                return INT8StaticQuantConfig.get_model_info_for_ipex_xpu(self, model)

    def to_config_mapping(
        self, config_list: List[BaseConfig] = None, model_info: List[Tuple[str, str]] = None
    ) -> OrderedDictType[Union[str, str], OrderedDictType[str, BaseConfig]]:
        """Convert the configuration to a mapping.

        Args:
            config_list (List[BaseConfig]): List of base configurations. Default is None.
            model_info (List[Tuple[str, str]]): List of tuples containing the name and type of each module in the model.
                Default is None.

        Returns:
            OrderedDictType[Union[str, str], OrderedDictType[str, BaseConfig]]: The configuration mapping.
        """
        if is_ipex_imported():
            return super().to_config_mapping(config_list, model_info)
        config_mapping = OrderedDict({self.name: self})
        return config_mapping

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "INT8StaticQuantConfig", List["INT8StaticQuantConfig"]]:
        """Get the default configuration set for tuning."""
        return INT8StaticQuantConfig(act_sym=[True, False], act_algo=["kl", "minmax"])


def get_default_static_config() -> INT8StaticQuantConfig:
    """Generate the default static quant config.

    Returns:
        the default static quant config.
    """
    return INT8StaticQuantConfig()


######################## Smooth Quant Config ###############################
@register_config(framework_name=FRAMEWORK_NAME, algo_name=SMOOTH_QUANT)
class SmoothQuantConfig(TorchBaseConfig):
    """Config class for smooth quantization."""

    name = SMOOTH_QUANT
    params_list = [
        "w_dtype",
        "w_sym",
        "w_granularity",
        "w_algo",
        "act_dtype",
        "act_sym",
        "act_granularity",
        "act_algo",
        "excluded_precisions",
        "alpha",
        "folding",
        "scale_sharing",
        "auto_alpha_args",
    ]
    supported_configs: List[OperatorConfig] = []

    def __init__(
        self,
        w_dtype: str = "int8",
        w_sym: bool = True,
        w_granularity: str = "per_channel",
        w_algo: str = "minmax",
        act_dtype: str = "uint8",
        act_sym: bool = False,
        act_granularity: str = "per_tensor",
        act_algo: str = "minmax",
        excluded_precisions: list = [],
        alpha: float = 0.5,
        folding: bool = False,
        # below for autotune
        scale_sharing: bool = False,
        init_alpha: float = 0.5,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        alpha_step: float = 0.1,
        shared_criterion: str = "max",
        do_blockwise: bool = False,
        auto_alpha_args: dict = None,
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
        **kwargs,
    ):
        """Init SmoothQuant Config.

        Args:
            w_dtype (str): Data type for weights, default is "int8".
            w_sym (bool): Whether to use symmetric quantization for weights, default is True.
            w_granularity (str): Level of quantization granularity for weights, default is "per_channel".
            w_algo (str): Quatization algorithm used to compute parameters for weights, default is "minmax".
            act_dtype (str): Data type for activations, default is "uint8".
            act_sym (bool): Whether to use symmetric quantization for activations, default is False.
            act_granularity (str): Level of quantization granularity for activations, default is "per_channel".
            act_algo (str): Quatization algorithm used to compute parameters for activations, default is "minmax".
            excluded_precisions (list): Precisions to be excluded, Default value is empty list.
            alpha (float): Value to balance input and weight quantization error, between 0 and 1, default is 0.5.
            folding (bool): Whether to fold mul into the previous layer, default is False.
            scale_sharing (bool): Whether share the same scale for layers with the same input, default is False.
            init_alpha (float): Value to get baseline quantization error for auto-tuning, default is 0.5.
            alpha_min (float): Min value of auto-tuning alpha search space, default is 0.0.
            alpha_max (float): Max value of auto-tuning alpha search space, default is 1.0.
            alpha_step (float): Step_size of auto-tuning alpha search space, default is 0.1.
            shared_criterion (str): Criterion for input LayerNorm op of a transformer block, default is "max".
            do_blockwise (bool): Whether to enable block-wise auto-tuning, default is False.
            auto_alpha_args (bool): Arguments for auto alpha searching, default is None.
            white_list (Optional[List[OP_NAME_OR_MODULE_TYPE]]): White list of operator names or module types.
                                                                 Default is DEFAULT_WHITE_LIST.
        """
        super().__init__(white_list=white_list)
        self.w_dtype = w_dtype
        self.w_sym = w_sym
        self.w_granularity = w_granularity
        self.w_algo = w_algo
        self.act_dtype = act_dtype
        self.act_sym = act_sym
        self.act_granularity = act_granularity
        self.act_algo = act_algo
        self.excluded_precisions = excluded_precisions
        self.alpha = alpha
        self.folding = folding
        # below for autotune
        self.scale_sharing = scale_sharing
        self.init_alpha = init_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_step = alpha_step
        self.shared_criterion = shared_criterion
        self.do_blockwise = do_blockwise
        self.auto_alpha_args = {
            "init_alpha": self.init_alpha,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "alpha_step": self.alpha_step,
            "shared_criterion": self.shared_criterion,
            "do_blockwise": self.do_blockwise,
        }
        self._post_init()

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        """Register supported configurations."""
        supported_configs = []
        # TODO(Yi)
        linear_sq_config = SmoothQuantConfig()
        operators = [torch.nn.Linear]
        supported_configs.append(OperatorConfig(config=linear_sq_config, operators=operators))
        cls.supported_configs = supported_configs

    def get_model_info(self, model: torch.nn.Module, example_inputs) -> List[Tuple[str, Callable]]:
        """Get information about the model.

        Args:
            model (torch.nn.Module): The model.

        Returns:
            List[Tuple[str, Callable]]: List of tuples containing the name and type of each module in the model.
        """
        from neural_compressor.torch.algorithms.smooth_quant import get_quantizable_ops_recursively

        model_info, cfgs, op_infos_from_cfgs, output_tensor_id_op_name = get_quantizable_ops_recursively(
            model, example_inputs, alpha=self.alpha, act_algo=self.act_algo, inplace=True
        )
        model.cfgs, model.op_infos_from_cfgs, model.output_tensor_id_op_name = (
            cfgs,
            op_infos_from_cfgs,
            output_tensor_id_op_name,
        )
        return model_info

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "SmoothQuantConfig", List["SmoothQuantConfig"]]:
        """Get the default configuration set for tuning."""
        import numpy as np

        return SmoothQuantConfig(
            alpha=np.arange(0.1, 1.0, 0.1).tolist(),
            folding=[True, False],
            scale_sharing=[True, False],
            excluded_precisions=[["bf16"]],
        )


def get_default_sq_config() -> SmoothQuantConfig:
    """Generate the default smoothquant config.

    Returns:
        the default smoothquant config.
    """
    return SmoothQuantConfig()


######################## HQQ Config ###############################
@register_config(framework_name=FRAMEWORK_NAME, algo_name=HQQ, priority=PRIORITY_HQQ)
class HQQConfig(TorchBaseConfig):
    """Configuration class for Half-Quadratic Quantization (HQQ).

    HQQ is a quantization algorithm that reduces the precision of weights and activations in neural networks.
    For more details, refer to the blog: https://mobiusml.github.io/hqq_blog/
    and the code: https://github.com/mobiusml/hqq
    """

    name = HQQ
    params_list = [
        "bits",
        "group_size",
        "quant_zero",
        "quant_scale",
        "scale_quant_group_size",
        "quant_lm_head",
    ]
    supported_configs: List[OperatorConfig] = []

    def __init__(
        self,
        dtype: str = "int",
        bits: int = 4,
        group_size: int = 64,
        quant_zero: bool = True,
        quant_scale: bool = False,
        scale_quant_group_size: int = 128,
        quant_lm_head: bool = False,
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
        **kwargs,
    ):
        """Initialize HQQConfig.

        Args:
            dtype (str): Data type for quantization. Default is "int".
            bits (int): Number of bits for quantization. Default is 4.
            group_size (int): Group size for quantization. Default is 64.
            quant_zero (bool): Whether to quantize zero values. Default is True.
            quant_scale (bool): Whether to quantize scale values. Default is False.
            scale_quant_group_size (int): Group size for scale quantization. Default is 128.
            quant_lm_head (bool): Whether to quantize the language model head. Default is False.
            white_list (Optional[List[OP_NAME_OR_MODULE_TYPE]]): White list of operator names or module types.
                Default is DEFAULT_WHITE_LIST.
        """
        super().__init__(white_list=white_list)
        self.dtype = dtype
        self.bits = bits
        self.group_size = group_size
        self.quant_zero = quant_zero
        self.quant_scale = quant_scale
        self.scale_quant_group_size = scale_quant_group_size
        self.quant_lm_head = quant_lm_head
        self._post_init()

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        """Register supported configurations for HQQ.

        Returns:
            List[OperatorConfig]: List of supported operator configurations.
        """
        supported_configs = []
        linear_hqq_config = HQQConfig()
        operators = list(WOQ_WHITE_LIST)
        supported_configs.append(OperatorConfig(config=linear_hqq_config, operators=operators))
        cls.supported_configs = supported_configs

    @staticmethod
    def get_model_info(model: torch.nn.Module) -> List[Tuple[str, Callable]]:
        """Get information about the model.

        Args:
            model (torch.nn.Module): The model.

        Returns:
            List[Tuple[str, Callable]]: List of tuples containing the name and type of each module in the model.
        """
        filter_result = []
        for op_name, module in model.named_modules():
            if isinstance(module, WOQ_WHITE_LIST):
                pair = (op_name, type(module).__name__)
                filter_result.append(pair)
        return filter_result

    def to_config_mapping(
        self, config_list: List[BaseConfig] = None, model_info: List[Tuple[str, str]] = None
    ) -> OrderedDictType[Union[str, str], OrderedDictType[str, BaseConfig]]:
        """Convert the configuration to a mapping.

        Args:
            config_list (List[BaseConfig]): List of base configurations. Default is None.
            model_info (List[Tuple[str, str]]): List of tuples containing the name and type of each module in the model.
                Default is None.

        Returns:
            OrderedDictType[Union[str, str], OrderedDictType[str, BaseConfig]]: The configuration mapping.
        """
        if not self.quant_lm_head:
            self.set_local(LM_HEAD_NAMES, HQQConfig(dtype="fp32"))
        config_mapping = super().to_config_mapping(config_list, model_info)
        return config_mapping

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "HQQConfig", List["HQQConfig"]]:
        """Get the configuration set for tuning.

        Returns:
            Union[None, "HQQConfig", List["HQQConfig"]]: The configuration set for tuning.
        """
        return HQQConfig(bits=[4, 8])


def get_default_hqq_config() -> HQQConfig:
    """Generate the default HQQ config.

    Returns:
        the default HQQ config.
    """
    return HQQConfig()


######################## FP8 Quant Config ###############################
if is_hpex_available():
    from neural_compressor.torch.algorithms.fp8_quant._core.common import get_white_list
else:
    get_white_list = lambda: []


@register_config(framework_name=FRAMEWORK_NAME, algo_name=FP8_QUANT)
class FP8Config(TorchBaseConfig):
    """Config class for FP8 quantization."""

    name = FP8_QUANT

    def __init__(
        self,
        dump_stats_path: str = "./hqt_output/measure",
        fp8_config: str = "E4M3",
        hp_dtype: str = "bf16",
        blocklist: dict = {"names": [], "types": ()},
        allowlist: dict = {"names": [], "types": get_white_list()},
        mode: str = "AUTO",
        scale_method: str = "maxabs_hw",
        scale_params: dict = {},
        observer: str = "maxabs",
        mod_dict: dict = {},
        measure_exclude: str = "OUTPUT",
        fake_quant: bool = False,
        use_qdq: bool = False,
        scale_format: str = "scalar",
        measure_on_hpu: bool = True,
        **kwargs,
    ):
        """Initializing FP8Config.

        Args:
            dump_stats_path (str, optional): The file folder and file prefix to save measurement info. Defaults to "./hqt_output/measure".
            fp8_config (str, optional): The data type of fp8. Defaults to "E4M3".
            hp_dtype (str, optional): The high precision data type used in fp8 quantization. Defaults to "bf16".
            blocklist (dict, optional): Whether to skip fp8 quantization for specific op names or types, name could be substring. Defaults to {"names": [], "types": ()}.
            allowlist (dict, optional): Whether to execute fp8 quantization for specific op names or types. Defaults to {"names": [], "types": FP8_WHITE_LIST}.
            mode (str, optional): Choose the quantization mode. Defaults to "AUTO".
            scale_method (str, optional): Select method used to generate scale from calibration info. Defaults to "maxabs_hw".
            scale_params (dict, optional): _description_. Defaults to {}.
            observer (str, optional): Params of scales. Defaults to "maxabs".
            mod_dict (dict, optional): The dict of modules to quantize. Defaults to {}.
            measure_exclude (str, optional): Select INPUT/OUTPUT to be exculded by measurement. Defaults to "OUTPUT".
            fake_quant (bool, optional): whether to execute fake quantization, a little bit different with use_qdq, used for training. Defaults to False.
            use_qdq (bool, optional): whether to execute Q/DQ quantization. Defaults to False.
        """
        super().__init__()
        self.dump_stats_path = dump_stats_path
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
        self.fake_quant = str(fake_quant)
        self.use_qdq = str(use_qdq)
        self.scale_format = scale_format
        self.measure_on_hpu = measure_on_hpu

    @property
    def measure(self):
        """Check whether the mode is for measurement."""
        return self.mode == "MEASURE"

    @property
    def quantize(self):
        """Check whether the mode is for quantization."""
        return self.mode == "QUANTIZE"

    @property
    def json_file(self):
        """Get the path of json file."""
        return self._json_file

    @json_file.setter
    def json_file(self, json_file):
        """Set the path of json file."""
        self._json_file = json_file

    @classmethod
    def from_json_file(cls, filename):
        """Set configuration from json file."""
        with open(filename, "r", encoding="utf-8") as file:
            config_dict = json.load(file)
        config = cls.from_dict(config_dict)
        config.json_file = filename
        return config

    def save_temp_json_file(self):
        """Save configuration to a temporary json file."""
        import tempfile
        from pathlib import Path

        json_file_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.to_json_file(json_file_tmp.name)
        self._json_file = json_file_tmp.name

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "FP8Config", List["FP8Config"]]:
        """Get the configuration set for tuning."""
        # just a simple example here
        # usually write parameter combinations that are more suitable to tune based on experience.
        return FP8Config()

    @classmethod
    def register_supported_configs(cls) -> List:
        pass

    @staticmethod
    def get_model_info(model: torch.nn.Module) -> List[Tuple[str, Callable]]:
        """Get information about the model.

        Args:
            model (torch.nn.Module): The model.

        Returns:
            List[Tuple[str, Callable]]: List of tuples containing the name and type of each module in the model.
        """
        filter_result = []
        for op_name, module in model.named_modules():
            if (
                module.__class__.__name__ in get_white_list()
                or module.__class__.__name__.split("Patched")[-1] in get_white_list()
            ):
                pair = (op_name, module.__class__.__name__)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    def to_config_mapping(self, config_list: List[BaseConfig] = None, model_info: List[Tuple[str, str]] = None):
        """Convert the configuration to a mapping.

        Args:
            config_list (List[BaseConfig]): List of base configurations. Default is None.
            model_info (List[Tuple[str, str]]): List of tuples containing the name and type of each module in the model.
                Default is None.

        Returns:
            OrderedDictType[Union[str, str], OrderedDictType[str, BaseConfig]]: The configuration mapping.
        """
        if self.json_file is None:
            self.save_temp_json_file()
        config_mapping = OrderedDict()
        if config_list is None:
            config_list = [self]
        for config in config_list:
            for op_name, op_type in model_info:
                config_mapping[(op_name, op_type)] = self
        return config_mapping


def get_default_fp8_config() -> FP8Config:
    """Generate the default fp8 config.

    Returns:
        the default fp8 config.
    """
    return FP8Config()


def get_default_fp8_config_set() -> FP8Config:
    """Generate the default fp8 config set.

    Returns:
        the default fp8 config.
    """
    return FP8Config.get_config_set_for_tuning()


######################## MixedPrecision Config ###############################
@register_config(framework_name=FRAMEWORK_NAME, algo_name=MIXED_PRECISION)
class MixedPrecisionConfig(TorchBaseConfig):
    """Config class for mixed-precision."""

    name = MIXED_PRECISION
    supported_configs: List[OperatorConfig] = []
    params_list = [
        "dtype",
    ]
    supported_half_precision_ops = (
        torch.nn.Linear,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
    )

    def __init__(
        self,
        dtype: Union[str, List[str]] = "fp16",
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
        **kwargs,
    ):
        """Init MixedPrecision config.

        Args:
            dtype (str or list): The data type of mixed precision, default is fp16.
            white_list (list): White list of operator names or module types, default is DEFAULT_WHITE_LIST.
        """
        super().__init__(white_list=white_list)
        self.dtype = dtype
        self._post_init()

    @classmethod
    def register_supported_configs(cls) -> List[OperatorConfig]:
        """Register supported configs."""
        supported_configs = []
        mixed_precision_config = MixedPrecisionConfig(
            dtype=["fp16", "bf16", "fp32"],
        )
        operators = cls.supported_half_precision_ops
        supported_configs.append(OperatorConfig(config=mixed_precision_config, operators=operators))
        cls.supported_configs = supported_configs

    @staticmethod
    def get_model_info(model: torch.nn.Module) -> List[Tuple[str, Callable]]:
        """Get information about the model.

        Args:
            model (torch.nn.Module): The model.

        Returns:
            List[Tuple[str, Callable]]: List of tuples containing the name and type of each module in the model.
        """
        white_list = tuple(MixedPrecisionConfig.supported_half_precision_ops)
        filter_result = []
        for op_name, module in model.named_modules():
            if isinstance(module, white_list):
                pair = (op_name, type(module).__name__)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "MixedPrecisionConfig", List["MixedPrecisionConfig"]]:
        """Get a default config set for tuning."""
        return MixedPrecisionConfig(dtype=["fp16", "bf16", "fp32"])


def get_default_mixed_precision_config() -> MixedPrecisionConfig:
    """Generate the default mixed-precision config.

    Returns:
        the default mixed-precision config.
    """
    return MixedPrecisionConfig()


def get_default_mixed_precision_config_set() -> MixedPrecisionConfig:
    """Generate the default mixed-precision config set.

    Returns:
        the default mixed-precision config.
    """
    return MixedPrecisionConfig.get_config_set_for_tuning()


##################### Algo Configs End ###################################


register_supported_configs_for_fwk(fwk_name=FRAMEWORK_NAME)


def get_all_registered_configs() -> Dict[str, BaseConfig]:
    """Get all registered configs."""
    registered_configs = config_registry.get_all_configs()
    return registered_configs.get(FRAMEWORK_NAME, {})


# =============================================================================
# Tuning Config
# =============================================================================


######################## WOQ Tuning Config ###############################
def get_woq_tuning_config() -> list:
    """Generate the config set for WOQ tuning.

    Returns:
        the list of WOQ quant config.
    """
    RTN_G32ASYM = RTNConfig(use_sym=False, group_size=32)
    AUTO_ROUND_CONFIG = AutoRoundConfig(use_sym=False, group_size=32, seqlen=512)
    GPTQ_G32ASYM = GPTQConfig(use_sym=False, group_size=32)
    AWQ_G32ASYM = AWQConfig(use_sym=False, group_size=32)
    return [RTN_G32ASYM, AUTO_ROUND_CONFIG, GPTQ_G32ASYM, AWQ_G32ASYM]


CONFIGS_FOR_STATIC_QUANT_MAPPING = OrderedDict(
    [
        # Configs for static quant mapping
        (STATIC_QUANT, INT8StaticQuantConfig),
        (FP8_QUANT, FP8Config),
    ]
)


class StaticQuantConfig(TorchBaseConfig):
    _model_mapping = CONFIGS_FOR_STATIC_QUANT_MAPPING

    def __new__(
        self,
        *args,
        **kwargs,
    ):
        dtype = kwargs.get("fp8_config", None)
        if dtype is not None:
            config_cls = self._model_mapping[FP8_QUANT]
        else:
            config_cls = self._model_mapping[STATIC_QUANT]
        return config_cls(*args, **kwargs)
