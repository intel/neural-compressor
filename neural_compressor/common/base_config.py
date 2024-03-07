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

import inspect
import json
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from typing_extensions import Self

from neural_compressor.common.tuning_param import TuningParam
from neural_compressor.common.utils import (
    BASE_CONFIG,
    COMPOSABLE_CONFIG,
    DEFAULT_WHITE_LIST,
    DEFAULT_WORKSPACE,
    EMPTY_WHITE_LIST,
    GLOBAL,
    LOCAL,
    OP_NAME_OR_MODULE_TYPE,
    logger,
)

__all__ = [
    "options",
    "register_config",
    "get_all_config_set_from_config_registry",
    "register_supported_configs_for_fwk",
    "BaseConfig",
    "ConfigRegistry",
    "ComposableConfig",
]


# Config registry to store all registered configs.
class ConfigRegistry(object):
    registered_configs = {}
    _config_registry = None

    def __new__(cls) -> Self:
        if cls._config_registry is None:
            cls._config_registry = super(ConfigRegistry, cls).__new__(cls)

        return cls._config_registry

    @classmethod
    def register_config_impl(cls, framework_name: str, algo_name: str, priority: Union[float, int] = 0):
        """Register config decorator.

        The register the configuration classes for different algorithms within specific frameworks.

        Usage example:
            @ConfigRegistry.register_config(framework_name=FRAMEWORK_NAME, algo_name=ExampleAlgorithm, priority=100)
            class ExampleAlgorithmConfig:
                # Configuration details for the ExampleAlgorithm

        Args:
            framework_name: the framework name.
            algo_name: the algorithm name.
            priority: priority: the priority of the configuration. A larger number indicates a higher priority,
                which will be tried first at the auto-tune stage. Defaults to 0.
        """

        def decorator(config_cls):
            cls.registered_configs.setdefault(framework_name, {})
            cls.registered_configs[framework_name][algo_name] = {"priority": priority, "cls": config_cls}
            return config_cls

        return decorator

    @classmethod
    def get_all_configs(cls) -> Dict[str, Dict[str, Dict[str, object]]]:
        """Get all registered configurations."""
        return cls.registered_configs

    @classmethod
    def get_sorted_configs(cls) -> Dict[str, OrderedDict[str, Dict[str, object]]]:
        """Get registered configurations sorted by priority."""
        sorted_configs = OrderedDict()
        for framework_name, algos in sorted(cls.registered_configs.items()):
            sorted_configs[framework_name] = OrderedDict(
                sorted(algos.items(), key=lambda x: x[1]["priority"], reverse=True)
            )
        return sorted_configs

    @classmethod
    def get_cls_configs(cls) -> Dict[str, Dict[str, object]]:
        """Get registered configurations without priority."""
        cls_configs = {}
        for framework_name, algos in cls.registered_configs.items():
            cls_configs[framework_name] = {}
            for algo_name, config_data in algos.items():
                cls_configs[framework_name][algo_name] = config_data["cls"]
        return cls_configs

    @classmethod
    def get_all_config_cls_by_fwk_name(cls, fwk_name: str) -> List[Type[BaseConfig]]:
        configs_cls = []
        for algo_name, config_pairs in cls.registered_configs.get(fwk_name, {}).items():
            configs_cls.append(config_pairs["cls"])
        return configs_cls


config_registry = ConfigRegistry()


def register_config(framework_name: str, algo_name: str, priority: Union[float, int] = 0):
    """Register config decorator.

    The register the configuration classes for different algorithms within specific frameworks.

    Usage example:
        @register_config(framework_name=FRAMEWORK_NAME, algo_name=ExampleAlgorithm, priority=100)
        class ExampleAlgorithmConfig:
            # Configuration details for the ExampleAlgorithm

    Args:
        framework_name: the framework name.
        algo_name: the algorithm name.
        priority: the priority of the configuration. A larger number indicates a higher priority,
            which will be tried first at the auto-tune stage. Defaults to 0.
    """

    return config_registry.register_config_impl(framework_name=framework_name, algo_name=algo_name, priority=priority)


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

    def to_dict(self):
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
        config = self
        tuning_param_list = []
        not_tuning_param_pair = {}  # key is the param name, value is the user specified value
        for param in params_list:
            # Create `TuningParam` for each param
            # There are two cases:
            # 1. The param is a string.
            # 2. The param is a `TuningParam` instance.
            if isinstance(param, str):
                default_param = self.get_the_default_value_of_param(config, param)
                tuning_param = TuningParam(name=param, tunable_type=List[type(default_param)])
            elif isinstance(param, TuningParam):
                tuning_param = param
            else:
                raise ValueError(f"Unsupported param type: {param}")
            # Assign the options to the `TuningParam` instance
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
            for params_values in product(*[tuning_param.options for tuning_param in tuning_param_list]):
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
        # * Ort and TF may override this method.
        return not isinstance(name, str)

    @classmethod
    @abstractmethod
    def get_config_set_for_tuning(cls):
        raise NotImplementedError


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


def get_all_config_set_from_config_registry(fwk_name: str) -> Union[BaseConfig, List[BaseConfig]]:
    all_registered_config_cls: List[BaseConfig] = config_registry.get_all_config_cls_by_fwk_name(fwk_name)
    config_set = []
    for config_cls in all_registered_config_cls:
        config_set.append(config_cls.get_config_set_for_tuning())
    return config_set


def register_supported_configs_for_fwk(fwk_name: str):
    """Register supported configs for specific framework.

    Args:
        fwk_name: the framework name.
    """
    all_registered_config_cls: List[BaseConfig] = config_registry.get_all_config_cls_by_fwk_name(fwk_name)
    for config_cls in all_registered_config_cls:
        config_cls.register_supported_configs()


#######################################################
####   Options
#######################################################


def _check_value(name, src, supported_type, supported_value=[]):
    """Check if the given object is the given supported type and in the given supported value.

    Example::

        from neural_compressor.common.base_config import _check_value

        def datatype(self, datatype):
            if _check_value("datatype", datatype, list, ["fp32", "bf16", "uint8", "int8"]):
                self._datatype = datatype
    """
    if isinstance(src, list) and any([not isinstance(i, supported_type) for i in src]):
        assert False, "Type of {} items should be {} but not {}".format(
            name, str(supported_type), [type(i) for i in src]
        )
    elif not isinstance(src, list) and not isinstance(src, supported_type):
        assert False, "Type of {} should be {} but not {}".format(name, str(supported_type), type(src))

    if len(supported_value) > 0:
        if isinstance(src, str) and src not in supported_value:
            assert False, "{} is not in supported {}: {}. Skip setting it.".format(src, name, str(supported_value))
        elif (
            isinstance(src, list)
            and all([isinstance(i, str) for i in src])
            and any([i not in supported_value for i in src])
        ):
            assert False, "{} is not in supported {}: {}. Skip setting it.".format(src, name, str(supported_value))

    return True


class Options:
    """Option Class for configs.

    This class is used for configuring global variables. The global variable options is created with this class.
    If you want to change global variables, you should use functions from neural_compressor.common.utils.utility.py:
        set_random_seed(seed: int)
        set_workspace(workspace: str)
        set_resume_from(resume_from: str)
        set_tensorboard(tensorboard: bool)

    Args:
        random_seed(int): Random seed used in neural compressor.
                          Default value is 1978.
        workspace(str): The directory where intermediate files and tuning history file are stored.
                        Default value is:
                            "./nc_workspace/{}/".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")).
        resume_from(str): The directory you want to resume tuning history file from.
                          The tuning history was automatically saved in the workspace directory
                               during the last tune process.
                          Default value is None.
        tensorboard(bool): This flag indicates whether to save the weights of the model and the inputs of each layer
                               for visual display.
                           Default value is False.

    Example::

        from neural_compressor.common import set_random_seed, set_workspace, set_resume_from, set_tensorboard
        set_random_seed(2022)
        set_workspace("workspace_path")
        set_resume_from("workspace_path")
        set_tensorboard(True)
    """

    def __init__(self, random_seed=1978, workspace=DEFAULT_WORKSPACE, resume_from=None, tensorboard=False):
        """Init an Option object."""
        self.random_seed = random_seed
        self.workspace = workspace
        self.resume_from = resume_from
        self.tensorboard = tensorboard

    @property
    def random_seed(self):
        """Get random seed."""
        return self._random_seed

    @random_seed.setter
    def random_seed(self, random_seed):
        """Set random seed."""
        if _check_value("random_seed", random_seed, int):
            self._random_seed = random_seed

    @property
    def workspace(self):
        """Get workspace."""
        return self._workspace

    @workspace.setter
    def workspace(self, workspace):
        """Set workspace."""
        if _check_value("workspace", workspace, str):
            self._workspace = workspace

    @property
    def resume_from(self):
        """Get resume_from."""
        return self._resume_from

    @resume_from.setter
    def resume_from(self, resume_from):
        """Set resume_from."""
        if resume_from is None or _check_value("resume_from", resume_from, str):
            self._resume_from = resume_from

    @property
    def tensorboard(self):
        """Get tensorboard."""
        return self._tensorboard

    @tensorboard.setter
    def tensorboard(self, tensorboard):
        """Set tensorboard."""
        if _check_value("tensorboard", tensorboard, bool):
            self._tensorboard = tensorboard


options = Options()
