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

import enum
import inspect
import itertools
import json
import pathlib
import re
from abc import ABC
from abc import abstractmethod

import numpy as np
import onnx
import pydantic
from onnxruntime import quantization
from typing_extensions import Self

from onnx_neural_compressor import constants
from onnx_neural_compressor import data_reader
from onnx_neural_compressor import logger
from onnx_neural_compressor import utility

from collections import OrderedDict  # isort: skip
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Type, Union, _GenericAlias  # isort: skip


class ParamLevel(enum.Enum):
    OP_LEVEL = enum.auto()
    OP_TYPE_LEVEL = enum.auto()
    MODEL_LEVEL = enum.auto()


class TuningParam:
    """Define the tunable parameter for the algorithm.

    Example:
        Class FakeAlgoConfig(config.BaseConfig):
            '''Fake algo config.'''.

            params_list = [
                ...
                # For simple tunable types, like a list of int, giving
                # the param name is enough. `config.BaseConfig` class will
                # create the `TuningParam` implicitly.
                "simple_attr"

                # For complex tunable types, like a list of lists,
                # developers need to create the `TuningParam` explicitly.
                TuningParam("complex_attr", tunable_type=List[List[str]])

                # The default parameter level is `ParamLevel.OP_LEVEL`.
                # If the parameter is at a different level, developers need
                # to specify it explicitly.
                TuningParam("model_attr", level=ParamLevel.MODEL_LEVEL)

            ...

    # TODO: more examples to explain the usage of `TuningParam`.
    """

    def __init__(
        self,
        name: str,
        default_val: Any = None,
        tunable_type=None,
        options=None,
        level: ParamLevel = ParamLevel.OP_LEVEL,
    ) -> None:
        self.name = name
        self.default_val = default_val
        self.tunable_type = tunable_type
        self.options = options
        self.level = level

    @staticmethod
    def create_input_args_model(expect_args_type: Any) -> type:
        """Dynamically create an InputArgsModel based on the provided type hint.

        Parameters:
        - expect_args_type (Any): The user-provided type hint for input_args.

        Returns:
        - type: The dynamically created InputArgsModel class.
        """

        class DynamicInputArgsModel(pydantic.BaseModel):
            input_args: expect_args_type

        return DynamicInputArgsModel

    def is_tunable(self, value: Any) -> bool:
        # Use `Pydantic` to validate the input_args.
        # TODO: refine the implementation in further.
        assert isinstance(self.tunable_type, _GenericAlias), f"Expected a type hint, got {self.tunable_type} instead."
        DynamicInputArgsModel = TuningParam.create_input_args_model(self.tunable_type)
        try:
            new_args = DynamicInputArgsModel(input_args=value)
            return True
        except Exception as e:
            logger.debug(f"Failed to validate the input_args: {e}")
            return False


# Config registry to store all registered configs.
class ConfigRegistry(object):
    registered_configs = {}
    _config_registry = None

    def __new__(cls) -> Self:
        if cls._config_registry is None:
            cls._config_registry = super(ConfigRegistry, cls).__new__(cls)

        return cls._config_registry

    @classmethod
    def register_config_impl(cls, algo_name: str, priority: Union[float, int] = 0):
        """Register config decorator.

        The register the configuration classes for different algorithms.

        Usage example:
            @ConfigRegistry.register_config(algo_name=ExampleAlgorithm, priority=100)
            class ExampleAlgorithmConfig:
                # Configuration details for the ExampleAlgorithm

        Args:
            algo_name: the algorithm name.
            priority: priority: the priority of the configuration. A larger number indicates a higher priority,
                which will be tried first at the auto-tune stage. Defaults to 0.
        """

        def decorator(config_cls):
            cls.registered_configs[algo_name] = {"priority": priority, "cls": config_cls}
            return config_cls

        return decorator

    @classmethod
    def get_all_configs(cls) -> Dict[str, Dict[str, Dict[str, object]]]:
        """Get all registered configurations."""
        return cls.registered_configs

    @classmethod
    def get_sorted_configs(cls) -> Dict[str, OrderedDict[str, Dict[str, object]]]:
        """Get registered configurations sorted by priority."""
        return OrderedDict(sorted(cls.registered_configs.items(), key=lambda x: x[1]["priority"], reverse=True))

    @classmethod
    def get_cls_configs(cls) -> Dict[str, Dict[str, object]]:
        """Get registered configurations without priority."""
        cls_configs = {}
        for algo_name, config_data in cls.registered_configs.items():
            cls_configs[algo_name] = config_data["cls"]
        return cls_configs

    @classmethod
    def get_all_config_cls(cls) -> List[Type[BaseConfig]]:
        configs_cls = []
        for algo_name, config_pairs in cls.registered_configs.items():
            configs_cls.append(config_pairs["cls"])
        return configs_cls


config_registry = ConfigRegistry()


def register_config(algo_name: str, priority: Union[float, int] = 0):
    """Register config decorator.

    The register the configuration classes for different algorithms.

    Usage example:
        @register_config(algo_name=ExampleAlgorithm, priority=100)
        class ExampleAlgorithmConfig:
            # Configuration details for the ExampleAlgorithm

    Args:
        algo_name: the algorithm name.
        priority: the priority of the configuration. A larger number indicates a higher priority,
            which will be tried first at the auto-tune stage. Defaults to 0.
    """

    return config_registry.register_config_impl(algo_name=algo_name, priority=priority)


class BaseConfig(ABC):
    """The base config for all algorithm configs."""

    name = constants.BASE_CONFIG
    params_list: List[Union[str, TuningParam]] = []

    def __init__(
        self,
        white_list: Optional[Union[Union[str, Callable], List[Union[str, Callable]]]] = constants.DEFAULT_WHITE_LIST,
    ) -> None:
        self._global_config: Optional[BaseConfig] = None
        # For PyTorch, operator_type is the collective name for module type and functional operation type,
        # for example, `torch.nn.Linear`, and `torch.nn.functional.linear`.
        # local config is the collections of operator_type configs and operator configs
        self._local_config: Dict[str, Optional[BaseConfig]] = {}
        self._white_list = white_list

    def _post_init(self):
        if self.white_list == constants.DEFAULT_WHITE_LIST:
            global_config = self.get_params_dict()
            self._global_config = self.__class__(**global_config, white_list=None)
        elif isinstance(self.white_list, list) and len(self.white_list) > 0:
            for op_name_or_type in self.white_list:
                global_config = self.get_params_dict()
                tmp_config = self.__class__(**global_config, white_list=None)
                self.set_local(op_name_or_type, tmp_config)
        elif self.white_list == constants.EMPTY_WHITE_LIST:
            return
        else:
            raise NotImplementedError(
                f"The white list should be one of {constants.DEFAULT_WHITE_LIST}, {constants.EMPTY_WHITE_LIST},"
                " a not empty list, but got {self.white_list}"
            )

    @property
    def white_list(self):
        return self._white_list

    @white_list.setter
    def white_list(self, op_name_or_type_list: Optional[List[Union[str, Callable]]]):
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

    def to_dict(self):
        result = {}
        global_config = self.get_params_dict()
        if bool(self.local_config):
            result[constants.LOCAL] = {}
            for op_name, config in self.local_config.items():
                result[constants.LOCAL][op_name] = config.to_dict()
            if self.global_config:
                result[constants.GLOBAL] = global_config
        else:
            result = global_config
        return result

    def get_params_dict(self):
        result = dict()
        for param, value in self.__dict__.items():
            if param not in ["_global_config", "_local_config", "_white_list"]:
                result[param] = value
        return result

    @classmethod
    def from_dict(cls, config_dict):
        """Construct config from a dict.

        Args:
            config_dict: _description_

        Returns:
            The constructed config.
        """
        if constants.GLOBAL not in config_dict and constants.LOCAL not in config_dict:
            config = cls(**config_dict)
            return config
        else:
            config = cls(**config_dict.get(constants.GLOBAL, {}))
            operator_config = config_dict.get(constants.LOCAL, {})
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

    def to_json_string(self, use_diff: bool = False) -> Union[str, Dict]:
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
        except Exception as e:
            logger.error("Failed to serialize the config to JSON string: %s", e)
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
        # TODO validate the user config
        pass

    def __add__(self, other: BaseConfig) -> BaseConfig:
        if isinstance(other, type(self)):
            for op_name, config in other.local_config.items():
                self.set_local(op_name, config)
            return self
        else:
            return ComposableConfig(configs=[self, other])

    @staticmethod
    def get_the_default_value_of_param(config: BaseConfig, param: str) -> Any:
        # Get the signature of the __init__ method
        signature = inspect.signature(config.__init__)

        # Get the parameters and their default values
        parameters = signature.parameters
        return parameters.get(param).default

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
        # TODO to support the expansion of config with `local`
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
        config = self
        tuning_param_list = []
        not_tuning_param_pair = {}  # key is the param name, value is the user specified value
        for param in params_list:
            # Create `tuning.TuningParam` for each param
            # There are two cases:
            # 1. The param is a string.
            # 2. The param is a `tuning.TuningParam` instance.
            if isinstance(param, str):
                default_param = self.get_the_default_value_of_param(config, param)
                tuning_param = TuningParam(name=param, tunable_type=List[type(default_param)])
            elif isinstance(param, TuningParam):
                tuning_param = param
            else:
                raise ValueError(f"Unsupported param type: {param}")
            # Assign the options to the `tuning.TuningParam` instance
            param_val = getattr(config, tuning_param.name)
            if param_val is not None:
                if tuning_param.is_tunable(param_val):
                    tuning_param.options = param_val
                    tuning_param_list.append(tuning_param)
                else:
                    not_tuning_param_pair[tuning_param.name] = param_val
        logger.debug("Tuning param list: %s", tuning_param_list)
        logger.debug("Not tuning param pair: %s", not_tuning_param_pair)
        if len(tuning_param_list) == 0:
            config_list = [config]
        else:
            tuning_param_name_lst = [tuning_param.name for tuning_param in tuning_param_list]
            for params_values in itertools.product(*[tuning_param.options for tuning_param in tuning_param_list]):
                tuning_param_pair = dict(zip(tuning_param_name_lst, params_values))
                tmp_params_dict = {**not_tuning_param_pair, **tuning_param_pair}
                new_config = self.__class__(**tmp_params_dict)
                logger.info(new_config.to_dict())
                config_list.append(new_config)
        logger.info("Expanded the %s and got %d configs.", self.__class__.name, len(config_list))
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
        self, config_list: Optional[List[BaseConfig]] = None, model_info: List[Tuple[str, str]] = None
    ) -> OrderedDict[Tuple[str, str], OrderedDict[str, BaseConfig]]:
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
                    if isinstance(op_name, str) and re.match(op_name_pattern, op_name):
                        config_mapping[(op_name, op_type)] = op_name_config_dict[op_name_pattern]
                    elif op_name_pattern == op_name:
                        config_mapping[(op_name, op_type)] = op_name_config_dict[op_name_pattern]
        return config_mapping

    @staticmethod
    def _is_op_type(name: str) -> bool:
        # * Ort and TF may override this method.
        return not isinstance(name, str)

    @classmethod
    @abstractmethod
    def get_config_set_for_tuning(cls):
        raise NotImplementedError


class ComposableConfig(BaseConfig):
    name = constants.COMPOSABLE_CONFIG

    def __init__(self, configs: List[BaseConfig]) -> None:
        self.config_list = configs

    def __add__(self, other: BaseConfig) -> BaseConfig:
        if isinstance(other, type(self)):
            self.config_list.extend(other.config_list)
        else:
            self.config_list.append(other)
        return self

    def to_dict(self):
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
        self, config_list: List[BaseConfig] = None, model_info: Dict[str, Any] = None
    ) -> OrderedDict[str, BaseConfig]:
        config_mapping = OrderedDict()
        for config in self.config_list:
            op_type_config_dict, op_name_config_dict = config._get_op_name_op_type_config()
            single_config_model_info = model_info.get(config.name, None)
            for op_name, op_type in single_config_model_info:
                if op_type in op_type_config_dict:
                    config_mapping[(op_name, op_type)] = op_name_config_dict[op_type]
                for op_name_pattern in op_name_config_dict:
                    if re.match(op_name_pattern, op_name):
                        config_mapping[(op_name, op_type)] = op_name_config_dict[op_name_pattern]
        return config_mapping

    @classmethod
    def register_supported_configs(cls):
        """Add all supported configs."""
        raise NotImplementedError

    @classmethod
    def get_config_set_for_tuning(cls) -> None:
        # TODO (Yi) handle the composable config in `tuning_config`
        return None

    def get_model_info(self, model, *args, **kwargs):
        model_info_dict = dict()
        for config in self.config_list:
            model_info_dict.update({config.name: config.get_model_info(model, *args, **kwargs)})
        return model_info_dict


def get_all_config_set_from_config_registry() -> List[BaseConfig]:
    all_registered_config_cls: List[Type[BaseConfig]] = config_registry.get_all_config_cls()
    config_set = []
    for config_cls in all_registered_config_cls:
        config_set.append(config_cls.get_config_set_for_tuning())
    return config_set


def register_supported_configs():
    """Register supported configs."""
    all_registered_config_cls: List[Type[BaseConfig]] = config_registry.get_all_config_cls()
    for config_cls in all_registered_config_cls:
        config_cls.register_supported_configs()


class _OperatorConfig(NamedTuple):
    config: BaseConfig
    operators: List[Union[str, Callable]]
    valid_func_list: List[Callable] = []


######################## RNT Config ###############################


@register_config(algo_name=constants.RTN, priority=constants.PRIORITY_RTN)
class RTNConfig(BaseConfig):
    """Config class for round-to-nearest weight-only quantization."""

    supported_configs: List[_OperatorConfig] = []
    params_list: List[Union[str, TuningParam]] = [
        "weight_dtype",
        "weight_bits",
        "weight_group_size",
        "weight_sym",
        "act_dtype",
        "accuracy_level",
        "ratios",
    ]
    model_params_list: List[str] = [
        "providers",
        "layer_wise_quant",
    ]
    name: str = constants.RTN

    def __init__(
        self,
        weight_dtype: str = "int",
        weight_bits: int = 4,
        weight_group_size: int = 32,
        weight_sym: bool = True,
        act_dtype: str = "fp32",
        accuracy_level: int = 0,
        ratios: dict = {},
        providers: List[str] = ["CPUExecutionProvider"],
        layer_wise_quant: bool = False,
        quant_last_matmul: bool = True,
        white_list: List[Union[str, Callable]] = constants.DEFAULT_WHITE_LIST,
    ):
        """Init RTN weight-only quantization config.

        Args:
            weight_dtype (str, optional): Data type for weights, default is "int".
            weight_bits (int, optional): Number of bits used to represent weights, default is 4.
            weight_group_size (int, optional): Size of weight groups, default is 32.
            weight_sym (bool, optional): Indicates whether weights are symmetric, default is True.
            act_dtype (str, optional): Data type for activations, default is "fp32".
            accuracy_level (int, optional): accuracy level. Support 0 (unset), 1(fp32 compute type of jblas kernel),
                2 (fp16 compute type of jblas kernel), 3 (bf16 compute type of jblas kernel),
                4 (int8 compute type of jblas kernel). Defaults to 0.
            ratios (dict, optional): percentile of clip. Defaults to {}.
            providers (list, optional): execution providers to use. Defaults to ["CPUExecutionProvider"].
            layer_wise_quant (bool, optional): whether to quantize model layer by layer to save memory footprint.
                Check below link for details
                https://github.com/intel/neural-compressor/blob/master/docs/source/quantization_layer_wise.md,
                default is False.
            quant_last_matmul (bool, optional): whether to quantize the last matmul of the model, default is True.
            white_list (list, optional): op in white_list will be applied current config.
                Defaults to constants.DEFAULT_WHITE_LIST.
        """
        super().__init__(white_list=white_list)
        self.weight_bits = weight_bits
        self.weight_dtype = weight_dtype
        self.weight_group_size = weight_group_size
        self.weight_sym = weight_sym
        self.act_dtype = act_dtype
        self.accuracy_level = accuracy_level
        self.ratios = ratios
        self.providers = providers
        self.layer_wise_quant = layer_wise_quant
        self.quant_last_matmul = quant_last_matmul
        self._post_init()

    def get_model_params_dict(self):
        result = dict()
        for param in self.model_params_list:
            result[param] = getattr(self, param)
        return result

    @classmethod
    def register_supported_configs(cls) -> None:
        supported_configs = []
        linear_rtn_config = RTNConfig(
            weight_dtype=["int"],
            weight_bits=[1, 2, 3, 4, 5, 6, 7, 8],
            weight_group_size=[32, -1, 1, 16, 64, 128, 256, 512, 1024],
            weight_sym=[True, False],
            act_dtype=["fp32"],
        )
        operators = ["MatMul"]
        supported_configs.append(_OperatorConfig(config=linear_rtn_config, operators=operators))
        cls.supported_configs = supported_configs

    def to_config_mapping(self, config_list: List[BaseConfig] = None, model_info: list = None):
        config_mapping = OrderedDict()
        if config_list is None:
            config_list = [self]
        for config in config_list:
            # update model level setting
            config_mapping.update(config.get_model_params_dict())

            # update node level setting
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
        if not self.quant_last_matmul:
            config_mapping[model_info[-1]] = {
                "weight": {"dtype": "fp32"},
                "activation": {"dtype": "fp32", "quant_mode": "fp32"},
            }
        return config_mapping

    @staticmethod
    def get_model_info(model: Union[onnx.ModelProto, pathlib.Path, str]) -> list:
        if not isinstance(model, onnx.ModelProto):
            model = onnx.load(model, load_external_data=False)
        white_list = ["MatMul"]
        filter_result = []
        for node in model.graph.node:
            if node.op_type in white_list:
                pair = (node.name, node.op_type)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "RTNConfig", List["RTNConfig"]]:  # pragma: no cover
        return RTNConfig(weight_bits=[4, 8], weight_sym=[True, False])


def get_default_rtn_config() -> RTNConfig:
    """Generate the default rtn config.

    Returns:
        the default rtn config.
    """
    return RTNConfig()


######################## GPTQ Config ###############################


@register_config(algo_name=constants.GPTQ, priority=constants.PRIORITY_GPTQ)
class GPTQConfig(BaseConfig):
    """Config class for gptq weight-only quantization."""

    supported_configs: List[_OperatorConfig] = []
    params_list: List[Union[str, TuningParam]] = [
        "weight_dtype",
        "weight_bits",
        "weight_group_size",
        "weight_sym",
        "act_dtype",
        "accuracy_level",
    ]
    model_params_list: List[Union[str, TuningParam]] = [
        "percdamp",
        "blocksize",
        "actorder",
        "mse",
        "perchannel",
        "providers",
        "layer_wise_quant",
    ]
    name: str = constants.GPTQ

    def __init__(
        self,
        weight_dtype: str = "int",
        weight_bits: int = 4,
        weight_group_size: int = 32,
        weight_sym: bool = True,
        act_dtype: str = "fp32",
        accuracy_level: int = 0,
        percdamp: float = 0.01,
        blocksize: int = 128,
        actorder: bool = False,
        mse: bool = False,
        perchannel: bool = True,
        providers: List[str] = ["CPUExecutionProvider"],
        layer_wise_quant: bool = False,
        quant_last_matmul: bool = True,
        white_list: List[Union[str, Callable]] = constants.DEFAULT_WHITE_LIST,
    ):
        """Init GPTQ weight-only quantization config.

        Args:
            weight_dtype (str, optional): data type for weights. Defaults to "int".
            weight_bits (int, optional): number of bits used to represent weights. Defaults to 4.
            weight_group_size (int, optional): size of weight groups. Defaults to 32.
            weight_sym (bool, optional): indicates whether weights are symmetric. Defaults to True.
            act_dtype (str, optional): data type for activations. Defaults to "fp32".
            accuracy_level (int, optional): accuracy level. Support 0 (unset), 1(fp32 compute type of jblas kernel),
                2 (fp16 compute type of jblas kernel), 3 (bf16 compute type of jblas kernel),
                4 (int8 compute type of jblas kernel). Defaults to 0.
            percdamp (float, optional): percentage of Hessian's diagonal values' average, which will be added
                to Hessian's diagonal to increase numerical stability. Defaults to 0.01.
            blocksize (int, optional): execute GPTQ quantization per block. Defaults to 128.
            actorder (bool, optional): whether to sort Hessian's diagonal values to rearrange channel-wise
                quantization order. Defaults to False.
            mse (bool, optional): whether get scale and zero point with mse error. Defaults to False.
            perchannel (bool, optional): whether quantize weight per-channel. Defaults to True.
            providers (list, optional): execution providers to use. Defaults to ["CPUExecutionProvider"].
            layer_wise_quant (bool, optional): whether to quantize model layer by layer to save memory footprint.
                Check below link for details
                https://github.com/intel/neural-compressor/blob/master/docs/source/quantization_layer_wise.md,
                default is False.
            quant_last_matmul (bool, optional): whether to quantize the last matmul of the model, default is True.
            white_list (list, optional): op in white_list will be applied current config.
                Defaults to constants.DEFAULT_WHITE_LIST.
        """
        super().__init__(white_list=white_list)
        self.weight_bits = weight_bits
        self.weight_dtype = weight_dtype
        self.weight_group_size = weight_group_size
        self.weight_sym = weight_sym
        self.act_dtype = act_dtype
        self.accuracy_level = accuracy_level
        self.percdamp = percdamp
        self.blocksize = blocksize
        self.actorder = actorder
        self.mse = mse
        self.perchannel = perchannel
        self.providers = providers
        self.layer_wise_quant = layer_wise_quant
        self.quant_last_matmul = quant_last_matmul
        self._post_init()

    def get_model_params_dict(self):
        result = dict()
        for param in self.model_params_list:
            result[param] = getattr(self, param)
        return result

    @classmethod
    def register_supported_configs(cls) -> None:
        supported_configs = []
        linear_gptq_config = GPTQConfig(
            weight_dtype=["int"],
            weight_bits=[1, 2, 3, 4, 5, 6, 7, 8],
            weight_group_size=[32, -1, 1, 16, 64, 128, 256, 512, 1024],
            weight_sym=[True, False],
            act_dtype=["fp32"],
            actorder=[True, False],
            mse=[True, False],
            perchannel=[True, False],
        )
        operators = ["MatMul"]
        supported_configs.append(_OperatorConfig(config=linear_gptq_config, operators=operators))
        cls.supported_configs = supported_configs

    def to_config_mapping(self, config_list: list = None, model_info: list = None) -> OrderedDict:
        config_mapping = OrderedDict()
        if config_list is None:
            config_list = [self]
        for config in config_list:
            # update model level setting
            config_mapping.update(config.get_model_params_dict())

            # update node level setting
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
        if not self.quant_last_matmul:
            config_mapping[model_info[-1]] = {
                "weight": {"dtype": "fp32"},
                "activation": {"dtype": "fp32", "quant_mode": "fp32"},
            }
        return config_mapping

    @staticmethod
    def get_model_info(model: Union[onnx.ModelProto, pathlib.Path, str]) -> list:
        if not isinstance(model, onnx.ModelProto):
            model = onnx.load(model, load_external_data=False)
        white_list = ["MatMul"]
        filter_result = []
        for node in model.graph.node:
            if node.op_type in white_list:
                pair = (node.name, node.op_type)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "GPTQConfig", List["GPTQConfig"]]:  # pragma: no cover
        return GPTQConfig(
            weight_bits=[4, 8],
            weight_sym=[True, False],
            actorder=[True, False],
            mse=[True, False],
            perchannel=[True, False],
        )


def get_default_gptq_config() -> GPTQConfig:
    """Generate the default gptq config.

    Returns:
        the default gptq config.
    """
    return GPTQConfig()


######################## AWQ Config ###############################


@register_config(algo_name=constants.AWQ, priority=constants.PRIORITY_AWQ)
class AWQConfig(BaseConfig):
    """Config class for awq weight-only quantization."""

    supported_configs: List[_OperatorConfig] = []
    params_list: List[str] = [
        "weight_dtype",
        "weight_bits",
        "weight_group_size",
        "weight_sym",
        "act_dtype",
        "accuracy_level",
    ]
    model_params_list: List[str] = [
        "enable_auto_scale",
        "enable_mse_search",
        "providers",
    ]
    name: str = constants.AWQ

    def __init__(
        self,
        weight_dtype: str = "int",
        weight_bits: int = 4,
        weight_group_size: int = 32,
        weight_sym: bool = True,
        act_dtype: str = "fp32",
        accuracy_level: int = 0,
        enable_auto_scale: bool = True,
        enable_mse_search: bool = True,
        providers: List[str] = ["CPUExecutionProvider"],
        quant_last_matmul: bool = True,
        white_list: List[Union[str, Callable]] = constants.DEFAULT_WHITE_LIST,
    ):
        """Init AWQ weight-only quantization config.

        Args:
            weight_dtype (str, optional): data type for weights. Defaults to "int".
            weight_bits (int, optional): number of bits used to represent weights. Defaults to 4.
            weight_group_size (int, optional): size of weight groups. Defaults to 32.
            weight_sym (bool, optional): indicates whether weights are symmetric. Defaults to True.
            act_dtype (str, optional): data type for activations. Defaults to "fp32".
            accuracy_level (int, optional): accuracy level. Support 0 (unset), 1(fp32 compute type of jblas kernel),
                2 (fp16 compute type of jblas kernel), 3 (bf16 compute type of jblas kernel),
                4 (int8 compute type of jblas kernel). Defaults to 0.
            enable_auto_scale (bool, optional): whether to search for best scales based on activation distribution.
                Defaults to True.
            enable_mse_search (bool, optional): whether to search for the best clip range from range
                [0.91, 1.0, 0.01]. Defaults to True.
            providers (list, optional): execution providers to use. Defaults to ["CPUExecutionProvider"].
            quant_last_matmul (bool, optional): whether to quantize the last matmul of the model, default is True.
            white_list (list, optional): op in white_list will be applied current config.
                Defaults to constants.DEFAULT_WHITE_LIST.
        """
        super().__init__(white_list=white_list)
        self.weight_bits = weight_bits
        self.weight_dtype = weight_dtype
        self.weight_group_size = weight_group_size
        self.weight_sym = weight_sym
        self.act_dtype = act_dtype
        self.accuracy_level = accuracy_level
        self.enable_auto_scale = enable_auto_scale
        self.enable_mse_search = enable_mse_search
        self.providers = providers
        self.quant_last_matmul = quant_last_matmul
        self._post_init()

    def get_model_params_dict(self):
        result = dict()
        for param in self.model_params_list:
            result[param] = getattr(self, param)
        return result

    @classmethod
    def register_supported_configs(cls) -> List[_OperatorConfig]:
        supported_configs = []
        linear_awq_config = AWQConfig(
            weight_dtype=["int"],
            weight_bits=[1, 2, 3, 4, 5, 6, 7, 8],
            weight_group_size=[32, -1, 1, 16, 64, 128, 256, 512, 1024],
            weight_sym=[True, False],
            act_dtype=["fp32"],
            enable_auto_scale=[True, False],
            enable_mse_search=[True, False],
        )
        operators = ["MatMul"]
        supported_configs.append(_OperatorConfig(config=linear_awq_config, operators=operators))
        cls.supported_configs = supported_configs

    def to_config_mapping(self, config_list: list = None, model_info: list = None) -> OrderedDict:
        config_mapping = OrderedDict()
        if config_list is None:
            config_list = [self]
        for config in config_list:
            # update model level setting
            config_mapping.update(config.get_model_params_dict())

            # update node level setting
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
        if not self.quant_last_matmul:
            config_mapping[model_info[-1]] = {
                "weight": {"dtype": "fp32"},
                "activation": {"dtype": "fp32", "quant_mode": "fp32"},
            }
        return config_mapping

    @staticmethod
    def get_model_info(model: Union[onnx.ModelProto, pathlib.Path, str]) -> list:
        if not isinstance(model, onnx.ModelProto):
            model = onnx.load(model, load_external_data=False)
        white_list = ["MatMul"]
        filter_result = []
        for node in model.graph.node:
            if node.op_type in white_list:
                pair = (node.name, node.op_type)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "AWQConfig", List["AWQConfig"]]:  # pragma: no cover
        return AWQConfig(
            weight_bits=[4, 8],
            weight_sym=[True, False],
            enable_auto_scale=[True, False],
            enable_mse_search=[True, False],
        )


def get_default_awq_config() -> AWQConfig:
    """Generate the default awq config.

    Returns:
        the default awq config.
    """
    return AWQConfig()


######################## SmoohQuant Config ###############################


@register_config(algo_name=constants.SMOOTH_QUANT, priority=constants.PRIORITY_SMOOTH_QUANT)
class SmoothQuantConfig(BaseConfig, quantization.StaticQuantConfig):
    """Smooth quant quantization config."""

    supported_configs: List[_OperatorConfig] = []
    params_list: List[str] = [
        # smooth parameters
        "alpha",
        "folding",
        "auto_alpha_args",
        "calib_iter",
        "scales_per_op",
    ]
    name: str = constants.SMOOTH_QUANT

    def __init__(
        self,
        alpha: float = 0.5,
        folding: bool = True,
        op_types: List[str] = ["Gemm", "Conv", "MatMul", "FusedConv"],
        calib_iter: int = 100,
        scales_per_op: bool = True,
        auto_alpha_args: dict = {"alpha_min": 0.3, "alpha_max": 0.7, "alpha_step": 0.05, "attn_method": "min"},
        providers: List[str] = ["CPUExecutionProvider"],
        white_list: List[Union[str, Callable]] = constants.DEFAULT_WHITE_LIST,
        **kwargs,
    ):
        """Init smooth quant config.

        Args:
            alpha (float, optional): alpha value to balance the quantization difficulty of activation and weight.
                Defaults to 0.5.
            folding (bool, optional): whether fold those foldable Mul which are inserted for smooth quant.
                Defaults to True.
            op_types (list, optional): the op type to be smooth quantized.
                Defaults to ["Gemm", "Conv", "MatMul", "FusedConv"].
            calib_iter (int, optional): iteration num for calibration. Defaults to 100.
            scales_per_op (bool, optional): True, each op will have an individual scale, mainlyfor accuracy.
                False, ops with the same input will share a scale, mainly for performance. Defaults to True.
            auto_alpha_args (dict, optional): settings for alpha tuning.
                Defaults to {"alpha_min": 0.3, "alpha_max": 0.7, "alpha_step": 0.05, "attn_method": "min"}.
            providers (list, optional): providers used for inference.
                Defaults to ["CPUExecutionProvider"].
            white_list (list, optional): op in white_list will be applied current config.
                Defaults to constants.DEFAULT_WHITE_LIST.
            kwargs (dict): kwargs in below link are supported except calibration_data_reader:
                https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/quantize.py#L78
        """
        BaseConfig.__init__(self)
        kwargs.update({"calibration_data_reader": None})
        quantization.StaticQuantConfig.__init__(self, **kwargs)
        self.alpha = alpha
        self.folding = folding
        self.op_types = op_types
        self.calib_iter = calib_iter
        self.scales_per_op = scales_per_op
        self.auto_alpha_args = auto_alpha_args
        self.providers = providers
        self.white_list = white_list
        self.weight_type = self.weight_type.value if isinstance(self.weight_type, enum.Enum) else self.weight_type
        self.activation_type = (
            self.activation_type.value if isinstance(self.activation_type, enum.Enum) else self.activation_type
        )
        self.calibrate_method = (
            self.calibrate_method.value if isinstance(self.calibrate_method, enum.Enum) else self.calibrate_method
        )
        self.quant_format = self.quant_format.value if isinstance(self.quant_format, enum.Enum) else self.quant_format
        self._post_init()

    @classmethod
    def register_supported_configs(cls) -> List[_OperatorConfig]:
        supported_configs = []
        smooth_quant_config = SmoothQuantConfig()
        operators = ["Gemm", "Conv", "MatMul", "FusedConv"]
        supported_configs.append(_OperatorConfig(config=smooth_quant_config, operators=operators))
        cls.supported_configs = supported_configs

    @staticmethod
    def get_model_info(model) -> list:
        white_list = ["Gemm", "Conv", "MatMul", "FusedConv"]
        filter_result = []
        for node in model.graph.node:
            if node.op_type in white_list:
                pair = (node.name, node.op_type)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(
        cls,
    ) -> Union[None, "SmoothQuantConfig", List["SmoothQuantConfig"]]:  # pragma: no cover
        return SmoothQuantConfig(alpha=np.arange(0.3, 0.7, 0.05))

    def convert_to_ort_config(self):
        self.activation_type = quantization.QuantType(self.activation_type)
        self.weight_type = quantization.QuantType(self.weight_type)
        self.weight_type = quantization.QuantType(self.weight_type)
        self.calibrate_method = quantization.CalibrationMethod(self.calibrate_method)
        self.quant_format = quantization.QuantFormat(self.quant_format)


def get_default_sq_config() -> SmoothQuantConfig:
    """Generate the default smooth quant config.

    Returns:
        the default smooth quant config.
    """
    return SmoothQuantConfig()


######################## WOQ Tuning Config ###############################


def get_woq_tuning_config() -> list:
    """Generate the config set for WOQ tuning.

    Returns:
        the list of WOQ quant config.
    """
    RTN_G32ASYM = RTNConfig(weight_sym=False)
    GPTQ_G32ASYM = GPTQConfig(weight_sym=False)
    GPTQ_G32ASYM_DISABLE_LAST_MATMUL = GPTQConfig(weight_sym=False, quant_last_matmul=False)
    GPTQ_G128ASYM = GPTQConfig(weight_group_size=128, weight_sym=False)
    AWQ_G32ASYM = AWQConfig(weight_sym=False)
    return [RTN_G32ASYM, GPTQ_G32ASYM, GPTQ_G32ASYM_DISABLE_LAST_MATMUL, GPTQ_G128ASYM, AWQ_G32ASYM]


##################### INC Algo Configs End ###################################

register_supported_configs()

##################### Config for ONNXRuntime-like user-facing API ############


class StaticQuantConfig(quantization.StaticQuantConfig):

    def __init__(self, calibration_data_reader: data_reader.CalibrationDataReader, extra_options=None, *args, **kwargs):
        """This is a class for static Quant Configuration.

        Inherit from StaticQuantConfig:
        https://github.com/microsoft/onnxruntime/blob/v1.17.1/onnxruntime/python/tools/quantization/quantize.py#L78
        extra_options:
            Support smoothquant args.
            - SmoothQuant = True/False :
                Default is False. If enabled, SmoothQuant algorithm will be applied before quantization to do
                fake input channel quantization.
            - SmoothQuantAlpha = float :
                Default is 0.5. It only works if SmoothQuant is True. It controls the difficulty of weight
                and activation quantization. A larger alpha value could be used on models with more significant
                activation outliers to migrate more quantization difficulty to weights.
            - SmoothQuantFolding = True/False :
                Default is True. It only works if SmoothQuant is True. If enabled, inserted Mul ops during
                SmoothQuant will be folded into the previous op if the previous op is foldable.
            - SmoothQuantOpTypes = list (new args):
                Default is ["Gemm", "Conv", "MatMul", "FusedConv"]. It only works if SmoothQuant is True.
                It controls the op types to be smooth quantized.
            - SmoothQuantCalibIter = int (new args):
                Default is 100. It only works if SmoothQuant is True. It controls the iteration num for calibration.
            - SmoothQuantScalesPerOp = True/False (new args) :
                Default is True. It only works if SmoothQuant is True.
                If enabled, each op will have an individual scale, mainlyfor accuracy.
                If not enabled,  ops with the same input will share a scale, mainly for performance.
        """
        super().__init__(calibration_data_reader=calibration_data_reader, extra_options=extra_options, *args, **kwargs)

    def to_dict(self):
        return self.__dict__


class DynamicQuantConfig(quantization.DynamicQuantConfig):
    """This is a class for dynamic Quant Configuration.

    Inherit from DynamicQuantConfig:
        https://github.com/microsoft/onnxruntime/blob/v1.17.1/onnxruntime/python/tools/quantization/quantize.py#L206
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def generate_nc_sq_config(quant_config: quantization.StaticQuantConfig):
    extra_options = quant_config.extra_options
    quant_kwargs = {
        "alpha": extra_options.get("SmoothQuantAlpha", 0.5),
        "folding": extra_options.get("SmoothQuantFolding", True),
        "op_types": extra_options.get("SmoothQuantOpTypes", ["Gemm", "Conv", "MatMul", "FusedConv"]),
        "calib_iter": extra_options.get("SmoothQuantCalibIter", 100),
        "scales_per_op": extra_options.get("SmoothQuantScalesPerOp", True),
    }
    quant_config.extra_options["SmoothQuant"] = False
    quant_config_dict = quant_config.to_dict()
    nc_sq_config = SmoothQuantConfig(**quant_kwargs, **quant_config_dict)
    return nc_sq_config
