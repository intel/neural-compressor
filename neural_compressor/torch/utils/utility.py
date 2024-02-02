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

import torch

from neural_compressor.common.utils import logger

# Dictionary to store a mapping between algorithm names and corresponding algo implementation(function)
algos_mapping: Dict[str, Callable] = {}


# All constants for torch
WHITE_MODULE_LIST = [torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d]


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


def get_depth(d) -> int:
    """Query the depth of the dict."""
    if isinstance(d, dict):
        return 1 + max(get_depth(v) for v in d.values())
    return 0


def get_dict_at_depth(d, target_depth, result, depth=0):
    """Get all sub-dicts that are at a specified depth in a nested dict."""
    if depth == target_depth:
        result.append(d)
        return
    elif depth < target_depth and isinstance(d, dict):
        for k, v in d.items():
            get_dict_at_depth(v, target_depth, result, depth=depth + 1)


def get_element_under_depth(d, ops_lst):
    """Get all values in a nested dict."""
    if isinstance(d, dict):
        for k, v in d.items():
            get_element_under_depth(v, ops_lst)
    else:
        ops_lst.append(d)


def paser_cfgs(cfgs):  # pragma: no cover
    """Parse configs.

    Args:
        cfgs (dict): the input configs.


    Returns:
        ops_name (list): list of op names.
        tune_cfg (dict): dictionary of quantization configuration.
        op_infos_from_cfgs (dict): op infos from configs.
        output_tensor_ids_op_name (dict): dictionary of output tensor op names.
    """
    ops_name = []
    layer_output_infos_ids = []
    op_infos_from_cfgs = {}
    # record input_tensor_id and op_name
    # {"0": [(" ", "q_op_infos", "0"), (" ", "q_op_infos", "1")]}
    input_tensor_ids_op_name = {}
    output_tensor_ids_op_name = {}
    for module_key in cfgs.keys():
        for state in cfgs[module_key]:
            if state == "layer_output_infos":
                for index, op_info in enumerate(cfgs[module_key][state]):
                    name = (module_key, state, index)
                    ops_name.append(name)
                    layer_output_infos_ids.append(op_info["id"])
                    op_infos_from_cfgs[name] = op_info
                continue
            for op_cfg_id in cfgs[module_key][state].keys():
                op_info = cfgs[module_key][state][op_cfg_id]
                name = (module_key, state, op_cfg_id)
                if name not in ops_name:
                    ops_name.append(name)
                else:
                    assert False, "Please check IPEX int8 configure json whether have the same name ops"
                op_infos_from_cfgs[name] = op_info
                input_tensors = op_info["input_tensor_infos"]
                for input_tensor in input_tensors:
                    if "id" not in input_tensor.keys():
                        continue
                    else:
                        input_tensor_id = input_tensor["id"]
                    if input_tensor_id not in input_tensor_ids_op_name.keys():
                        input_tensor_ids_op_name[input_tensor_id] = [name]
                    else:
                        input_tensor_ids_op_name[input_tensor_id].append(name)
                output_tensors = op_info["output_tensor_infos"]
                for output_tensor in output_tensors:
                    if "id" not in output_tensor.keys():
                        continue
                    else:
                        output_tensor_id = output_tensor["id"]
                    if output_tensor_id not in output_tensor_ids_op_name.keys():
                        output_tensor_ids_op_name[output_tensor_id] = [name]
                    else:
                        output_tensor_ids_op_name[output_tensor_id].append(name)
    return ops_name, op_infos_from_cfgs, input_tensor_ids_op_name, output_tensor_ids_op_name


def get_quantizable_ops_from_cfgs(ops_name, op_infos_from_cfgs, input_tensor_ids_op_name):  # pragma: no cover
    """Get quantizable ops from configs, combine fused ops as one op.

    Args:
        ops_name (list): list of op names.
        op_infos_from_cfgs (dict): op infos from configs.
        input_tensor_ids_op_name (dict): dictionary of input tensor op names.

    Returns:
        cfgs (dict).
    """
    quantizable_ops = []
    seen_ops = []
    for name in ops_name:
        start = True
        if name in seen_ops:
            continue
        elif name[1] not in ["q_op_infos"]:
            continue
        else:
            # judge fuse ops the first op
            op_info = op_infos_from_cfgs[name]
            output_tensors = op_info["output_tensor_infos"]
            input_tensors = op_info["input_tensor_infos"]
            start = any(
                [
                    input_tensor["inf_dtype"] != "torch.float32"
                    for input_tensor in input_tensors
                    if "inf_dtype" in input_tensor.keys()
                ]
            )
            if not start:
                continue
            # add quantizable ops, include op and fuse ops.
            q_ops, stack = [], [(name, [])]
            while stack:
                cur_name, cur = stack.pop()
                seen_ops.append(cur_name)
                if cur_name[1] not in ["q_op_infos"]:
                    q_ops.append(cur)
                    break
                op_info = op_infos_from_cfgs[cur_name]
                output_tensors = op_info["output_tensor_infos"]
                for output_tensor in output_tensors:
                    if output_tensor["inf_dtype"] == "torch.qint8" or output_tensor["inf_dtype"] == "torch.quint8":
                        q_ops.append(cur + [cur_name])
                        break
                    try:
                        next_op_names = input_tensor_ids_op_name[output_tensor["id"]]
                        for next_op_name in next_op_names:
                            stack.append((next_op_name, cur + [cur_name]))
                    except:
                        next_op_name = None
                    if next_op_name is None:
                        q_ops.append(cur + [cur_name])
            for q_op in q_ops:
                quantizable_ops.append(q_op)
    return quantizable_ops
