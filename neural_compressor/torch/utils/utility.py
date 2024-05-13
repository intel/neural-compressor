# Copyright (c) 2024 Intel Corporation
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


from enum import Enum
from typing import Callable, Dict, List, Tuple, Union

import torch
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.observer import HistogramObserver, MinMaxObserver
from torch.ao.quantization.quantizer import QuantizationSpec
from torch.ao.quantization.quantizer.x86_inductor_quantizer import QuantizationConfig, X86InductorQuantizer
from typing_extensions import TypeAlias

from neural_compressor.common import logger

OP_NAME_AND_TYPE_TUPLE_TYPE: TypeAlias = Tuple[str, Union[torch.nn.Module, Callable]]

# Dictionary to store a mapping between algorithm names and corresponding algo implementation(function)
algos_mapping: Dict[str, Callable] = {}


# All constants for torch
WHITE_MODULE_LIST = [torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d]


WEIGHT_NAME = "quantized_model.pt"
QCONFIG_NAME = "qconfig.json"


def register_algo(name):
    """Decorator function to register algorithms in the algos_mapping dictionary.

    Usage example:
        @register_algo(name=example_algo)
        def example_algo(model: torch.nn.Module, quant_config: RTNConfig) -> torch.nn.Module:
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


def get_model_info(model: torch.nn.Module, white_module_list: List[Callable]) -> List[Tuple[str, str]]:
    module_dict = dict(model.named_modules())
    filter_result = []
    filter_result_set = set()
    for op_name, module in module_dict.items():
        if isinstance(module, tuple(white_module_list)):
            pair = (op_name, type(module).__name__)
            if pair not in filter_result_set:
                filter_result_set.add(pair)
                filter_result.append(pair)
    logger.debug(f"Get model info: {filter_result}")
    return filter_result


def get_double_quant_config(double_quant_type):
    from neural_compressor.torch.utils.constants import DOUBLE_QUANT_CONFIGS

    if double_quant_type is None:
        return {}
    assert double_quant_type in DOUBLE_QUANT_CONFIGS, "Supported double quant configs: {}".format(
        list(DOUBLE_QUANT_CONFIGS.keys())
    )
    return DOUBLE_QUANT_CONFIGS[double_quant_type]


class Mode(Enum):
    PREPARE = "prepare"
    CONVERT = "convert"
    QUANTIZE = "quantize"


def get_quantizer(model, quantizer_cls, quant_config=None, *args, **kwargs):
    """Get the quantizer.

    Initialize a quantizer or get `quantizer` attribute from model.

    Args:
        model (torch.nn.Module): pytorch model.
        quantizer_cls (Quantizer): quantizer class of a specific algorithm.
        quant_config (dict, optional): Specifies how to apply the algorithm on the given model.
            Defaults to None.

    Returns:
        quantizer object.
    """
    if not hasattr(model, "quantizer"):
        quantizer = quantizer_cls(quant_config=quant_config, *args, **kwargs)
        return quantizer
    else:
        return model.quantizer


def postprocess_model(model, mode, quantizer):
    """Process `quantizer` attribute of model according to current phase.

    In `prepare` phase, the `quantizer` is set as an attribute of the model
    to avoid redundant initialization during `convert` phase.

    In 'convert' or 'quantize' phase, the unused `quantizer` attribute is removed.

    Args:
        model (torch.nn.Module): pytorch model.
        mode (Mode): The mode of current phase, including 'prepare', 'convert' and 'quantize'.
        quantizer (Quantizer): quantizer object.
    """
    if mode == Mode.PREPARE:
        model.quantizer = quantizer
    elif mode == Mode.CONVERT or mode == Mode.QUANTIZE:
        if getattr(model, "quantizer", False):
            del model.quantizer


def create_quant_spec_from_config(dtype, sym, granularity, algo) -> QuantizationSpec:
    dtype_mapping: Dict[str, torch.dtype] = {"int8": torch.int8, "uint8": torch.uint8}
    qscheme_mapping = {
        "per_channel": {True: torch.per_channel_symmetric, False: torch.per_tensor_affine},
        "per_tensor": {True: torch.per_tensor_symmetric, False: torch.per_tensor_affine},
    }
    observer_mapping = {
        "minmax": MinMaxObserver,
        "kl": HistogramObserver,
    }
    # algo
    observer_or_fake_quant_ctr = observer_mapping[algo]
    # qscheme
    qscheme = qscheme_mapping[granularity][sym]
    quantization_spec = QuantizationSpec(
        dtype=dtype_mapping[dtype], observer_or_fake_quant_ctr=observer_or_fake_quant_ctr, qscheme=qscheme
    )
    return quantization_spec


def _map_inc_config_to_torch_quant_config(inc_config) -> QuantizationConfig:
    default_quant_config = xiq.get_default_x86_inductor_quantization_config()
    input_act_quant_spec = create_quant_spec_from_config(
        inc_config.act_dtype, inc_config.act_sym, inc_config.act_granularity, inc_config.act_algo
    )
    weight_quant_spec = create_quant_spec_from_config(
        inc_config.w_dtype, inc_config.w_sym, inc_config.w_granularity, inc_config.w_algo
    )
    quant_config = QuantizationConfig(
        input_activation=input_act_quant_spec,
        output_activation=default_quant_config.output_activation,
        weight=weight_quant_spec,
        bias=default_quant_config.bias,
        is_qat=False,
    )
    return quant_config


def create_xiq_quantizer_from_pt2e_config(config) -> X86InductorQuantizer:
    quantizer = xiq.X86InductorQuantizer()
    # set global
    global_config = _map_inc_config_to_torch_quant_config(config)
    quantizer.set_global(global_config)
    # set local
    for module_or_func_name, local_config in config.local_config.items():
        local_quant_config = _map_inc_config_to_torch_quant_config(local_config)
        if isinstance(module_or_func_name, torch.nn.Module):
            quantizer.set_module_type_qconfig(module_or_func_name, local_quant_config)
        else:
            quantizer.set_function_type_qconfig(module_or_func_name, local_quant_config)
    return quantizer
