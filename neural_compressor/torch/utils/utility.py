# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Callable, Dict, List, Tuple

from neural_compressor.common.logger import Logger

logger = Logger().get_logger()

# Dictionary to store a mapping between algorithm names and corresponding algo implementation(function)
algos_mapping: Dict[str, Callable] = {}

import torch

# All constants for torch
WHITE_MODULE_LIST = [torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d]


def register_algo(name):
    """Decorator function to register algorithms in the algos_mapping dictionary.

    Usage example:
        @register_algo(name=example_algo)
        def example_algo(model: torch.nn.Module, quant_config: RTNWeightQuantConfig) -> torch.nn.Module:
            ...

    Args:
        name (str): The name under which the algorithm function will be registered.

    Returns:
        decorator: The decorator function to be used with algorithm functions.
    """

    def decorator(algo_func):
        algos_mapping[name] = algo_func
        return algo_func

    return decorator


def fetch_module(model, op_name):
    """Get module with a given op name.

    Args:
        model (object): the input model.
        op_name (str): name of op.

    Returns:
        module (object).
    """
    module = model
    name_list = op_name.split(".")
    for name in name_list:
        if hasattr(module, name):
            module = getattr(module, name)
        else:
            logger.warning(f"The {op_name} is not present in the model.")
            return None
    return module


def set_module(model, op_name, new_module):
    """Set module with a given op name.

    Args:
        model (object): the input model.
        op_name (str): name of op.
        new_module (object): the input model.

    Returns:
        module (object).
    """
    name_list = op_name.split(".")
    if len(name_list) == 1:
        setattr(model, name_list[-1], new_module)
        return
    else:
        second_last_module = fetch_module(model, ".".join(name_list[:-1]))
        if second_last_module is None:
            logger.warning(f"Setting skipped as the {op_name} is not present in the model.")
            return None
        else:
            setattr(second_last_module, name_list[-1], new_module)


def get_model_info(model: torch.nn.Module, white_module_list: List[Callable]) -> List[Tuple[str, Callable]]:
    module_dict = dict(model.named_modules())
    filter_result = []
    filter_result_set = set()
    for op_name, module in module_dict.items():
        if isinstance(module, tuple(white_module_list)):
            pair = (op_name, type(module))
            if pair not in filter_result_set:
                filter_result_set.add(pair)
                filter_result.append(pair)
    logger.debug(f"Get model info: {filter_result}")
    return filter_result


def get_double_quant_config(double_quant_type, weight_sym=True):
    from neural_compressor.torch.utils.constants import DOUBLE_QUANT_CONFIGS

    if double_quant_type is None:
        return {}
    assert double_quant_type in DOUBLE_QUANT_CONFIGS, "Supported double quant configs: {}".format(
        list(DOUBLE_QUANT_CONFIGS.keys())
    )
    DOUBLE_QUANT_CONFIGS[double_quant_type]["weight_sym"] = weight_sym
    return DOUBLE_QUANT_CONFIGS[double_quant_type]


# pylint:disable=import-error
try:
    import deepspeed
    import habana_frameworks.torch.hpex

    _hpex_avalible = True
except:
    _hpex_avalible = False


def is_hpex_avaliable():
    return _hpex_avalible
