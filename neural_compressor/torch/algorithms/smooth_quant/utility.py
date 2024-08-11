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
"""Utility functions for Smooth quantization."""


import copy
import json
import os
import re
from collections import UserDict

import intel_extension_for_pytorch as ipex
import numpy
import torch
from packaging.version import Version
from tqdm import tqdm

from neural_compressor.torch.algorithms.static_quant import (
    CpuInfo,
    Statistics,
    TransformerBasedModelBlockPatternDetector,
    generate_activation_observer,
    get_quantizable_ops_from_cfgs,
    ipex_config_path,
    parse_cfgs,
    simple_inference,
    unify_op_type_mapping_ipex,
)
from neural_compressor.torch.utils import get_ipex_version, get_torch_version, logger

version = get_torch_version()
ipex_ver = get_ipex_version()


def get_quantizable_ops_recursively(model, example_inputs, alpha, act_algo, inplace=True):  # pragma: no cover
    """Get all quantizable ops from model.

    Args:
        model (object): input model
        example_inputs (dict|list|tuple|torch.Tensor): used to trace torch model.
        alpha (float|str): smoothquant alpha.
        act_algo (str): activation algorithm, minmax or kl.
        inplace (bool): whether to carry out model transformations in-place. Defaults to True.

    Returns:
        quantizable_ops (list): list of tuples of op_name and op_type.
        cfgs (dict): dict of configuration.
        op_infos_from_cfgs (dict): op infos from configs.
        output_tensor_ids_op_name (dict): dictionary of output tensor op names.
    """
    quantizable_ops = []
    # group ops by position for transform-based model
    detector = TransformerBasedModelBlockPatternDetector(model)
    detect_result = detector.detect_block()
    attention_block = detect_result.get("attention_blocks", None)
    ffn_blocks = detect_result.get("ffn_blocks", None)
    logger.info(f"Attention Blocks: {len(attention_block)}")
    logger.info(f"FFN Blocks: {len(ffn_blocks)}")
    if not os.path.exists(ipex_config_path):
        assert isinstance(model, torch.nn.Module), "The model passed in is not the instance of torch.nn.Module"

    if hasattr(model, "save_qconf_summary"):
        os.makedirs(os.path.dirname(ipex_config_path), exist_ok=True)
        model.save_qconf_summary(qconf_summary=ipex_config_path)
    else:  # pragma: no cover
        model.eval()

        # create a quantization config file for intel pytorch extension model
        os.makedirs(os.path.dirname(ipex_config_path), exist_ok=True)
        assert example_inputs is not None, "IPEX need q_dataloader or example_inputs to prepare the model"

        from torch.ao.quantization import MinMaxObserver

        if alpha == "auto":  # for quantize API
            alpha = 0.5

        if ipex_ver.release >= Version("2.1.1").release:
            static_qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(
                alpha=alpha, act_observer=MinMaxObserver
            )
        else:  # pragma: no cover
            if act_algo == "minmax":
                static_qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(
                    alpha=alpha, act_observer=MinMaxObserver()
                )
                logger.warning(
                    "The int8 model accuracy will be close to 0 with MinMaxobserver, "
                    + "the suggested IPEX version is higher or equal than 2.1.100+cpu."
                )
            else:
                static_qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=alpha)

        if isinstance(example_inputs, dict):
            model = ipex.quantization.prepare(
                model, static_qconfig, example_kwarg_inputs=example_inputs, inplace=inplace
            )
        else:
            model = ipex.quantization.prepare(model, static_qconfig, example_inputs=example_inputs, inplace=inplace)

        simple_inference(model, example_inputs, iterations=1)
        model.save_qconf_summary(qconf_summary=ipex_config_path)

    map_op_name_to_fqn = {}
    with open(ipex_config_path, "r") as f:
        cfgs = json.load(f)
        (
            ops_name,
            op_infos_from_cfgs,
            input_tensor_id_op_name,
            output_tensor_id_op_name,
        ) = parse_cfgs(cfgs)
        quantizable_op_names = get_quantizable_ops_from_cfgs(ops_name, op_infos_from_cfgs, input_tensor_id_op_name)
        for name in quantizable_op_names:
            # name : list
            if len(name) == 1:
                module_key = name[0][0]
                op_cfg_id = name[0][2]
                ipex_op_type = cfgs[module_key]["q_op_infos"][op_cfg_id]["op_type"]
                module_fqn = cfgs[module_key]["q_op_infos"][op_cfg_id].get("fqn", None)

                if ipex_op_type in unify_op_type_mapping_ipex:
                    quantizable_ops.append((tuple(name), unify_op_type_mapping_ipex[ipex_op_type]))
                    map_op_name_to_fqn[(tuple(name), ipex_op_type)] = module_fqn
                else:
                    re_flag = False
                    for pattern, unify_op_type in unify_op_type_mapping_ipex["re"].items():
                        if re.match(pattern, ipex_op_type):
                            re_flag = True
                            quantizable_ops.append((tuple(name), unify_op_type))
                            map_op_name_to_fqn[(tuple(name), unify_op_type)] = module_fqn
                            break
                    if not re_flag:
                        quantizable_ops.append((tuple(name), ipex_op_type))
                        map_op_name_to_fqn[(tuple(name), ipex_op_type)] = module_fqn
            else:  # pragma: no cover
                op_type = ""
                for op_name in name:
                    module_key = op_name[0]
                    op_cfg_id = op_name[2]
                    single_op_type = cfgs[module_key]["q_op_infos"][op_cfg_id]["op_type"]
                    if single_op_type in unify_op_type_mapping_ipex:
                        single_op_type = unify_op_type_mapping_ipex[single_op_type]
                    op_type += "&" + single_op_type if op_type else single_op_type
                quantizable_ops.append((tuple(name), op_type))
                _module_key = name[0][0]
                _op_cfg_id = name[0][2]
                module_fqn = cfgs[_module_key]["q_op_infos"][_op_cfg_id]["fqn"]
                map_op_name_to_fqn[(tuple(name), op_type)] = module_fqn

    logger.debug("Map op name to fqn: ")
    logger.debug(map_op_name_to_fqn)
    logger.info("Attention Blocks : ")
    logger.info(attention_block)
    logger.info("FFN Blocks : ")
    logger.info(ffn_blocks)
    return quantizable_ops, cfgs, op_infos_from_cfgs, output_tensor_id_op_name


def check_cfg_and_qconfig(
    tune_cfg, cfgs, op_infos_from_cfgs, output_tensor_ids_op_name, alpha=0.5, smooth_quant=True
):  # pragma: no cover
    """Check configs and quantization configs.

    Args:
        tune_cfg (dict): dictionary of quantization configuration.
        cfgs (dict): the input configs.
        op_infos_from_cfgs (dict): op infos from configs.
        output_tensor_ids_op_name (dict): dictionary of output tensor op names.
        alpha (float): Value to balance input and weight quantization error,
            between 0 and 1, default is 0.5.
        smooth_quant (bool, optional): whether to use smooth quant.

    Returns:
        cfgs (dict).
    """
    for op_name in tune_cfg:
        inc_op_cfg = tune_cfg[op_name]
        for i, name in enumerate(op_name[0]):
            # to int8
            ipex_op_cfg = op_infos_from_cfgs[name]
            input_tensor_infos = ipex_op_cfg["input_tensor_infos"]
            if op_name[1] == "Linear" or op_name[1] == "Linear&add":  # record op_name for possible op-wise fallback
                logger.debug(f"ipex_op_cfg['fqn'] - op_name {ipex_op_cfg['fqn']}  {op_name}")
            for index, input_tensor_info in enumerate(input_tensor_infos):
                if "force_dtype" not in input_tensor_info.keys():
                    continue
                if (
                    input_tensor_info["force_dtype"] == "torch.qint8"
                    or input_tensor_info["force_dtype"] == "torch.quint8"
                ):
                    # int8 -> int8
                    if inc_op_cfg["weight"]["dtype"] == "int8":
                        inc_scheme = inc_op_cfg["activation"]["scheme"]
                        inc_algorithm = inc_op_cfg["activation"]["algorithm"]
                        ipex_op_cfg["input_tensor_infos"] = input_tensor_infos
                        if (
                            "op_type" in ipex_op_cfg
                            and ipex_op_cfg["op_type"] == "<class 'torch.nn.modules.linear.Linear'>"
                        ):
                            smooth_quant_enable = True
                        else:
                            smooth_quant_enable = False
                        activation_observer = generate_activation_observer(
                            inc_scheme, inc_algorithm, smooth_quant, smooth_quant_enable, alpha
                        )
                        if not smooth_quant:
                            if inc_scheme == "sym":
                                input_tensor_infos[index]["force_dtype"] = "torch.qint8"
                            if inc_scheme == "asym":
                                input_tensor_infos[index]["force_dtype"] = "torch.quint8"
                        ipex_op_cfg["activation_observer"] = activation_observer
                    # int8 -> fp32
                    else:
                        input_tensor_infos[index]["force_dtype"] = "torch.float32"
                    # modify pre_op output inf_dtype
                    if i == 0:
                        input_tensor_id = input_tensor_info["id"]
                        input_tensor_dtype = input_tensor_info["force_dtype"]
                        if input_tensor_id in output_tensor_ids_op_name.keys():
                            pre_op_name = output_tensor_ids_op_name[input_tensor_id]
                            pre_op_module = pre_op_name[0][0]
                            pre_op_state = pre_op_name[0][1]
                            pre_op_index = pre_op_name[0][2]
                            pre_op_infos = cfgs[pre_op_module][pre_op_state][pre_op_index]
                            pre_op_output_infos = pre_op_infos["output_tensor_infos"]
                            for index, pre_op_output in enumerate(pre_op_output_infos):
                                if pre_op_output["id"] == input_tensor_id:
                                    pre_op_output_infos[index]["inf_dtype"] = input_tensor_dtype
                                else:
                                    pass
                            pre_op_infos["output_tensor_infos"] = pre_op_output_infos
                            cfgs[pre_op_module][pre_op_state][pre_op_index] = pre_op_infos
                        else:
                            pass
            cfgs[name[0]][name[1]][name[2]] = ipex_op_cfg
    return cfgs


def cfg_to_qconfig(
    tune_cfg, cfgs, op_infos_from_cfgs, output_tensor_id_op_name, alpha=0.5, smooth_quant=True
):  # pragma: no cover
    """Check configs and quantization configs.

    Args:
        user_cfg (dict): quantization configuration for ops.
        cfgs (dict): configs loaded from ipex config path.
        op_infos_from_cfgs (dict): dict containing configs that have been parsed for each op.
        output_tensor_ids_op_name (dict): dict containing op names corresponding to 'op_infos_from_cfgs'.
        alpha (float): Value to balance input and weight quantization error,
            between 0 and 1, default is 0.5.
        smooth_quant (bool, optional): whether to use smooth quant.

    Returns:
        cfgs (dict): updated configs.
    """
    assert cfgs is not None, "No configure for IPEX int8 model..."
    op_infos = copy.deepcopy(op_infos_from_cfgs)
    cfgs = check_cfg_and_qconfig(tune_cfg["op"], cfgs, op_infos, output_tensor_id_op_name, alpha, smooth_quant)
    with open(ipex_config_path, "w") as write_f:
        json.dump(cfgs, write_f, indent=4)
    return None


def dump_model_op_stats(user_cfg):
    """This is a function to dump quantizable ops of model to user.

    Args:
        user_cfg (dict): quantization config.

    Returns:
        None
    """
    res = dict()
    for k, v in user_cfg.items():
        op_type_list = k[-1].split("><")
        op_type = ""
        for op in op_type_list:
            if "class" in op:
                op_type = (
                    op[op.rfind(".") + 1 : op.rfind("'")]
                    if op_type == ""
                    else op_type + "&" + op[op.rfind(".") + 1 : op.rfind("'")]
                )
            elif "method" in op:
                start = op.find("'") + 1
                if start > 1:
                    op_type = (
                        op[start : op.find("'", start)]
                        if op_type == ""
                        else op_type + "&" + op[start : op.find("'", start)]
                    )
                else:
                    start = op.find("method") + 7
                    op_type = (
                        op[start : op.find(" ", start)]
                        if op_type == ""
                        else op_type + "&" + op[start : op.find(" ", start)]
                    )
            else:
                op_type = op if op_type == "" else op_type + "&" + op
        if op_type not in res.keys():
            res[op_type] = {"INT8": 0, "BF16": 0, "FP32": 0}
        if v["weight"]["dtype"] == "int8":
            res[op_type]["INT8"] += 1
        elif v["weight"]["dtype"] == "fp32":
            res[op_type]["FP32"] += 1

    output_data = [
        [op_type, sum(res[op_type].values()), res[op_type]["INT8"], res[op_type]["BF16"], res[op_type]["FP32"]]
        for op_type in res.keys()
    ]

    Statistics(
        output_data, header="Mixed Precision Statistics", field_names=["Op Type", "Total", "INT8", "BF16", "FP32"]
    ).print_stat()


def get_parent(node, all_parents=False):  # pragma: no cover
    """Get the parent node(s) of a given node.

    Args:
        node (Node): The node whose parent(s) are to be retrieved.
        all_parents (bool, optional): Whether to return all parents or just the first one. Defaults to False.

    Returns:
        list: The parent node if `all_parents` is False, otherwise a list of all parent nodes.
            Returns None if no parents are found.
    """
    if node.inputs() is None:
        return None
    elif len(list(node.inputs())) == 0:
        return None
    if not all_parents:
        return list(node.inputs())[0].node()
    else:
        return list(node.inputs())


def get_module(model, key):  # pragma: no cover
    """Get module from model by key name.

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
    """
    module = model
    name_list = key.split(".")
    for name in name_list:
        if hasattr(module, name):
            module = getattr(module, name)
        elif hasattr(module, "sq_linear"):  # for peft models
            module = getattr(module, "sq_linear")
            module = getattr(module, name)
        elif hasattr(module, "orig_layer"):  # for peft models and auto alpha
            module = getattr(module, "orig_layer")
            module = getattr(module, name)
        else:
            module = module
    return module


def set_module(model, key, new_module):  # pragma: no cover
    """Set new module into model by key name.

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
        new_module (torch.nn.Module): new module to be inserted
    """
    module = model
    name_list = key.split(".")
    for name in name_list[:-1]:
        if hasattr(module, name):
            module = getattr(module, name)
        elif hasattr(module, ("sq_linear")):  # for peft models that Linears are contained in Linear
            module = getattr(module, "sq_linear")
            module = getattr(module, name)
        elif hasattr(module, ("orig_layer")):  # for peft models and auto alpha
            module = getattr(module, "orig_layer")
            module = getattr(module, name)
        else:
            module = module

    if hasattr(module, "sq_linear") and name_list[-1] != "sq_linear":  # for peft models
        module = getattr(module, "sq_linear")
    if hasattr(module, "orig_layer") and name_list[-1] != "orig_layer":  # for peft models and auto alpha
        module = getattr(module, "orig_layer")
    setattr(module, name_list[-1], new_module)


def update_sq_scale(ipex_config_path, smoothquant_scale_info):  # pragma: no cover
    """Update ipex_config.json with smoothquant scale info generated by our algorithm.

    Args:
        ipex_config_path (str): a path to temporary ipex_config.json file.
        smoothquant_scale_info (dict): a dict contains smoothquant scale info.
    """
    with open(ipex_config_path, "r") as f:
        ipex_config = json.load(f)
        for module_name, v in ipex_config.items():
            if "q_op_infos" in v and v["q_op_infos"]:
                for op_num, v1 in v["q_op_infos"].items():
                    # update alpha data instead of updating weight scale
                    op_name = v1["fqn"]  # fqn always exists even it's empty.
                    if op_name in smoothquant_scale_info and v1["op_type_is_module"]:
                        input_scale_for_mul = smoothquant_scale_info[op_name]["input_scale_for_mul"].tolist()
                        input_scale_after_mul = smoothquant_scale_info[op_name]["input_scale_after_mul"].tolist()
                        input_zero_point_after_mul = smoothquant_scale_info[op_name][
                            "input_zero_point_after_mul"
                        ].tolist()
                        weight_scale_for_mul = (1 / smoothquant_scale_info[op_name]["input_scale_for_mul"]).tolist()
                        weight_scale_after_mul = smoothquant_scale_info[op_name]["weight_scale_after_mul"].tolist()
                        v1["input_tensor_infos"][0]["scale"] = input_scale_after_mul
                        v1["input_tensor_infos"][0]["zero_point"] = input_zero_point_after_mul
                        v1["input_tensor_infos"][0]["smooth_quant_scaling_factor"] = input_scale_for_mul
                        v1["weight_tensor_infos"][0]["smooth_quant_scaling_factor"] = weight_scale_for_mul
                        v1["weight_tensor_infos"][0]["scale"] = weight_scale_after_mul
                        # # observers were overridden by the fallback step, setting it back.
        f.close()
    # overwrite ipex_config_path
    with open(ipex_config_path, "w") as f1:
        json.dump(ipex_config, f1, indent=4)
        f1.close()


def enough_memo_store_scale(device, need_space):  # pragma: no cover
    """Check if there is enough memory available to store a specified amount of data.

    Args:
        device (str): The device type ('cuda' for GPU or 'cpu' for CPU).
        need_space (int): The amount of memory needed, in bytes.

    Returns:
        bool: True if there is enough memory available, False otherwise.
    """
    if device == "cuda":  # pragma: no cover
        current_gpu_index = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(current_gpu_index).total_memory
        used_memory = torch.cuda.memory_allocated(current_gpu_index)
        free_space = total_memory - used_memory
    else:
        import psutil

        free_space = psutil.virtual_memory().free
    return free_space >= need_space


def move_input_to_device(input, device=torch.device("cpu")):  # pragma: no cover
    """Move the input data to the specified device.

    Args:
        input (dict, list, tuple, or torch.Tensor): The input data to be moved.
            Can be a dictionary, list, tuple, or a tensor.
        device (torch.device, optional): The device to which the input should be moved.
            Defaults to CPU.

    Returns:
        The input data moved to the specified device,
            with the same type as the input (dict, list, tuple, or tensor).
    """
    if isinstance(input, dict) or isinstance(input, UserDict):
        tmp_input = {}
        for k, inp in input.items():
            tmp_input[k] = move_input_to_device(inp, device)
        input = tmp_input
    elif isinstance(input, list) or isinstance(input, tuple):
        is_tuple = isinstance(input, tuple)
        tmp_input = []
        for inp in input:
            tmp_input.append(move_input_to_device(inp, device))
        input = tuple(tmp_input) if is_tuple else tmp_input
    elif isinstance(input, torch.Tensor):
        input = input.to(device)  # pylint: disable=no-member
    return input


def forward_wrapper(model, input, device=torch.device("cpu")):  # pragma: no cover
    """Apply the model to the input data on the specified device.

    Args:
        model (torch.nn.Module): The model to be applied.
        input (dict, list, tuple, or zip): The input data to be fed to the model.
            Can be a dictionary, list, tuple, or a zip of arguments and keyword arguments.
        device (torch.device, optional): The device on which the model and input should be located.
            Defaults to CPU.

    Returns:
        The output of the model after applying it to the input data.

    Raises:
        Exception: Logs warnings if there are issues with moving the model or input to the device.
    """
    try:
        model = model.to(device)
        input = move_input_to_device(input, device)
    except Exception as e:
        logger.warning(e)
        logger.warning("Please check the input device if the error raised.")
    if isinstance(input, dict) or isinstance(input, UserDict):
        output = model(**input)
    elif isinstance(input, list) or isinstance(input, tuple):
        try:
            output = model(*input)
        except:
            output = model(input)
    elif isinstance(input, zip):
        for args, kwargs in input:
            output = model(*args, **kwargs)
    else:
        output = model(input)
    return output


def model_forward(model, dataloader, iters, device):  # pragma: no cover
    """Run the model on data from the dataloader for a specified number of iterations.

    Args:
        model (torch.nn.Module): The model to be used for forward passes.
        dataloader (DataLoader): The dataloader providing the input data and labels.
        iters (int): The maximum number of iterations to run.
            If -1, run until the dataloader is exhausted.
        device (torch.device): The device on which the model and data are located.

    Returns:
        None

    Raises:
        Exception: Handles exceptions during the forward pass and retries if needed.
    """
    try:
        cnt = 0
        for idx, (input, label) in enumerate(dataloader):
            output = forward_wrapper(model, input, device)
            cnt += 1
            if iters != -1 and cnt >= iters:
                break
    except Exception as e:
        cnt = 0
        for idx, input in enumerate(dataloader):
            output = forward_wrapper(model, input, device)
            cnt += 1
            if iters != -1 and cnt >= iters:
                break


def build_captured_dataloader(model, run_fn, calib_num=None):
    """Build a dataloader that captures input data and keyword arguments used in forward passes of the model.

    Args:
        model (torch.nn.Module): The model whose inputs will be captured.
        run_fn (function): A function to run the model, which will use the InputCaptureModule to collect inputs.
        calib_num (int, optional): The number of inputs to capture for calibration. If None, capture all inputs.

    Returns:
        torch.nn.Module: The original model.
        CapturedDataloader: A dataloader with the captured inputs and keyword arguments.
    """

    class CapturedDataloader:
        def __init__(self, args_list, kwargs_list) -> None:
            self.args_list = args_list
            self.kwargs_list = kwargs_list

        def __iter__(self):
            for args, kwargs in zip(self.args_list, self.kwargs_list):
                if not args:
                    yield kwargs
                elif not kwargs:
                    yield args
                else:
                    yield args, kwargs

    class InputCaptureModule(torch.nn.Module):
        def __init__(self, model) -> None:
            super().__init__()
            self.args_list = []
            self.kwargs_list = []
            self.orig_model = model
            self.iters = 0
            self.calib_num = calib_num

        def forward(self, *args, **kwargs):
            if self.iters < self.calib_num:
                self.args_list.append(args)
                self.kwargs_list.append(kwargs)
                self.iters += 1

    captured_model = InputCaptureModule(model)
    run_fn(captured_model)
    dataloader = CapturedDataloader(captured_model.args_list, captured_model.kwargs_list)
    model = captured_model.orig_model
    return model, dataloader


def cal_scale(input_max_abs, weights, alpha, weight_max_lb=1e-5):  # pragma: no cover
    """Calculate the scaling factor for weights based on the input max values and weight magnitudes.

    Args:
        input_max_abs (Tensor): The maximum absolute values of the inputs.
        weights (list of Tensor): The list of weight tensors to be concatenated and processed.
        alpha (float): A parameter to balance the scaling between inputs and weights.
        weight_max_lb (float, optional): The lower bound for weight magnitudes to avoid division by zero.
            Defaults to 1e-5.

    Returns:
        Tensor: The calculated scaling factors for the weights.
    """
    weights = torch.cat(weights, dim=0)
    weight_max = torch.max(torch.abs(weights), dim=0)[0]
    weight_max = torch.clip(weight_max, weight_max_lb)
    input_power = torch.pow(input_max_abs, alpha)
    logger.debug(f"{max(input_max_abs)}, {min(input_max_abs)}")
    weight_power = torch.pow(weight_max, 1 - alpha)
    weight_scale = torch.clip(input_power / weight_power, min=1e-5)
    weight_scale[input_power == 0] = 1.0
    return weight_scale


def model_forward_per_sample(model, sample, device):  # pragma: no cover
    """Perform a forward pass of the model on a single sample.

    Args:
        model (torch.nn.Module): The model to be applied.
        sample (Tensor or tuple): The input sample or a tuple of inputs to be passed to the model.
        device (torch.device): The device on which the model and input sample are located.

    Returns:
        Tensor: The output of the model after applying it to the sample.

    Raises:
        Exception: Handles exceptions during the forward pass and retries if needed.
    """
    try:
        output = forward_wrapper(model, sample, device)
        return output

    except Exception as e:
        output = forward_wrapper(model, sample[0], device)
        return output


def quant_dequant_w_v1(m, num_bits=8, scheme="sym"):  # pragma: no cover
    """Quantize and dequantize the weights of a layer.

    Args:
        m (torch.nn.Module): The layer whose weights are to be quantized and dequantized.
            Supports torch.nn.Linear and torch.nn.Conv2d.
        num_bits (int, optional): The number of bits for quantization.
            Defaults to 8.
        scheme (str, optional): The quantization scheme to use.
            Can be "sym" for symmetric or "asym" for asymmetric quantization. Defaults to "sym".

    Returns:
        Tensor: The quantized and dequantized weights of the layer.

    Raises:
        Warning: Logs a warning if the layer type is not supported.
    """
    eps = torch.finfo(torch.float32).eps
    if isinstance(m, torch.nn.Linear):
        x = m.weight
        tmp = torch.zeros(torch.max(x, dim=1).values.size())
        if scheme == "sym":
            q_min, q_max = -(2.0 ** (num_bits - 1)), 2.0 ** (num_bits - 1) - 1.0
            x_max = torch.max(torch.abs(x), dim=1).values
            scale = x_max / (float(q_max - q_min) / 2)
        else:
            q_min, q_max = 0, 2.0**num_bits - 1.0
            x_max = torch.maximum(torch.max(x, dim=1).values, tmp)
            x_min = torch.minimum(torch.min(x, dim=1).values, tmp)
            scale = (x_max - x_min) / (2**num_bits - 1)

        scale = torch.clip(scale, min=eps)

        if scheme == "sym":
            bias = 0
        else:
            bias = torch.round(0 - (torch.min(x, dim=1).values) / scale)
            bias = bias.unsqueeze(dim=-1)
        scale = scale.unsqueeze(dim=-1)
        q_x = torch.round(x / scale + bias)
        q_x.clamp_(q_min, q_max)
        return (q_x - bias) * scale
    elif isinstance(m, torch.nn.Conv2d):
        x = m.weight
        x = torch.permute(x, (0, 2, 3, 1))
        x = x.reshape(-1, x.shape[-1])
        tmp = torch.zeros(torch.max(x, dim=0).values.size())
        if scheme == "sym":
            q_min, q_max = -(2.0 ** (num_bits - 1)), 2.0 ** (num_bits - 1) - 1.0
            x_max = torch.max(torch.abs(x), dim=0).values
            scale = x_max / (2 ** (num_bits - 1) - 1)
        else:
            q_min, q_max = 0, 2.0**num_bits - 1.0
            x_max = torch.maximum(torch.max(x, dim=0).values, tmp)
            x_min = torch.minimum(torch.min(x, dim=0).values, tmp)
            scale = (x_max - x_min) / (2**num_bits - 1)
        scale = torch.clip(scale, min=eps)
        if scheme == "sym":
            bias = 0
        else:
            bias = torch.round(0 - (torch.min(x, dim=0).values) / scale)
            bias = bias.unsqueeze(dim=0)
        scale = scale.unsqueeze(dim=0)

        q_x = x / scale + bias
        q_x.clamp_(q_min, q_max).round_()
        q_dq_x = (q_x - bias) * scale
        q_dq_x = q_dq_x.view(m.weight.shape[0], m.weight.shape[2], m.weight.shape[3], m.weight.shape[1])
        q_dq_x = torch.permute(q_dq_x, (0, 3, 1, 2))
        return q_dq_x
    else:
        logger.warning("unsupported layer type, please have a check")


def quant_dequant_x_v1(x, min_x=None, max_x=None, num_bits=8):  # pragma: no cover
    """Quantize and dequantize a tensor.

    Args:
        x (Tensor): The input tensor to be quantized and dequantized.
        min_x (Tensor, optional): The minimum value of the input tensor.
            If None, it will be computed from x. Defaults to None.
        max_x (Tensor, optional): The maximum value of the input tensor.
            If None, it will be computed from x. Defaults to None.
        num_bits (int, optional): The number of bits for quantization. Defaults to 8.

    Returns:
        Tensor: The quantized and dequantized tensor.

    Raises:
        None: No specific exceptions are raised, but input values are clipped to avoid invalid operations.
    """
    eps = torch.finfo(torch.float32).eps
    q_min, q_max = 0, 2.0**num_bits - 1.0
    if max_x is None or min_x is None:
        max_x, min_x = torch.max(x), torch.min(x)
    else:
        max_x = torch.max(max_x)
        min_x = torch.min(min_x)
    scale = (max_x - min_x) / (2**num_bits - 1)
    scale = torch.clip(scale, min=eps)
    bias = torch.round((0 - min_x) / scale)
    q_x = torch.round(x / scale + bias)
    q_x.clamp_(q_min, q_max)
    return scale * (q_x - bias)


def reshape_scale_as_weight(layer, scale):  # pragma: no cover
    """Reshape the scale for weight input channel, depthwise output channel.

    Args:
        layer (torch.nn.Module): Torch module.
        scale (Tensor): Original scale.

    Returns:
        Tensor: Reshaped scale.
    """
    if hasattr(layer, "orig_layer"):
        layer = layer.orig_layer
    if isinstance(layer, torch.nn.Conv2d) and layer.groups > 1:  ##only depthwise conv could hit here
        scale = scale.view(scale.shape[0], 1, 1, 1)  ##mount on output channel

    elif isinstance(layer, torch.nn.Conv2d):
        scale = scale.view(1, scale.shape[0], 1, 1)

    elif isinstance(layer, torch.nn.Linear):
        scale = scale.view(1, scale.shape[0])

    return scale


def reshape_in_channel_to_last(layer_name, model):  # pragma: no cover
    """Move the input channel to the last dimension.

    Args:
        layer_name (str): Layer name.

    Returns:
        Tensor: The reshaped weight.
    """
    layer = get_module(model, layer_name)
    if layer.__class__.__name__ == "WrapperLayer":
        layer = layer.orig_layer

    weight = layer.weight  ##TODO oc*ic, support transposed conv
    if len(weight.shape) == 4:
        weight = weight.permute(0, 2, 3, 1)
        weight = weight.reshape(-1, weight.shape[-1])
    return weight


def reshape_scale_as_input(layer, scale):  # pragma: no cover
    """Reshape the scale for input feature in channel.

    Args:
        layer (torch.nn.Module): Torch module.
        scale (Tensor): Original scale.

    Returns:
        Tensor: Reshaped scale.
    """
    if hasattr(layer, "orig_layer"):
        layer = layer.orig_layer
    if isinstance(layer, torch.nn.Conv2d):
        scale = scale.view(1, scale.shape[0], 1, 1)

    elif isinstance(layer, torch.nn.Linear):
        scale = scale.view(1, scale.shape[0])

    return scale


TUNERS = {}


def register_autotune(name):  # pragma: no cover
    """Class decorator to register a SmoothQuant auto-tune subclass.

    Returns:
        type: The class of register.
    """

    def register(auto_tune):
        TUNERS[name] = auto_tune
        return auto_tune

    return register


class Calibration:  # pragma: no cover
    """Calibration class."""

    def __init__(self, model, dataloder=None, q_func=None, device="cpu"):
        """Initialize the Calibration class.

        Args:
            model (torch.nn.Module): The model to be calibrated.
            dataloder (DataLoader, optional): DataLoader providing the calibration data. Defaults to None.
            q_func (Callable, optional): A function for quantization. Defaults to None.
            device (str, optional): The device to perform calibration on. Defaults to "cpu".
        """
        self.model = model
        self.dataloader = dataloder
        self.q_func = q_func
        self.device = device

    @torch.no_grad()
    def _save_input_pc_hook(self, name):
        """A forward hook to save input max of a module.

        Args:
            name (str): The module name.

        Returns:
            function: A hook function.
        """

        def save_input_hook(module, inputs, outputs):
            input = inputs[0]
            ##TODO check input channel is correct
            if len(module.weight.shape) == 4:  ##conv3d or conv1d not supported now, need better way
                input = input.permute(0, 2, 3, 1)
            input = input.reshape(-1, input.shape[-1])
            max_tensor = torch.max(input, dim=0)[0]
            min_tensor = torch.min(input, dim=0)[0]
            if name not in self.input_maxes.keys():
                self.input_mins[name], self.input_maxes[name] = min_tensor, max_tensor
            else:
                self.input_mins[name] = torch.min(self.input_mins[name], min_tensor)
                self.input_maxes[name] = torch.max(self.input_maxes[name], max_tensor)

        return save_input_hook

    @torch.no_grad()
    def _add_min_max_observer(self, modules):
        """Insert observers into the given modules.

        Args:
            modules (list): The modules to which the observer will be inserted.

        Returns:
            None
        """
        self.hook_handles = []
        for key in modules.keys():
            hook_func = self._save_input_pc_hook(key)
            hook_handle = modules[key].register_forward_hook(hook_func)
            self.hook_handles.append(hook_handle)

    @torch.no_grad()
    def _remove_observer(self):
        """Remove the observer from the model.

        Returns:
            None
        """
        for hook_handle in self.hook_handles:
            hook_handle.remove()

    @torch.no_grad()
    def _dump_min_max(self, calib_iter=100):
        """Dump min-max per channel information; the min-max values will be saved in the input_maxes attribute.

        Args:
            calibration_method (str): Only supports 'min_max' currently.
            calib_iter (int): Sample size for calibration.

        Returns:
            None
        """
        logger.info("Calibrating...")
        if self.q_func:
            self.q_func(self.model)
        else:
            assert self.dataloader, "Please set dataloader for calibration."
            model_forward(self.model, self.dataloader, calib_iter, self.device)

    @torch.no_grad()
    def calibrate(self, calib_iter, op_types=[torch.nn.Conv2d, torch.nn.Linear]):  ##TODO transformers.conv1d
        """Process the absorb layer and smooth layers, then return the channel-wise max value info.

        Args:
            absorb_to_layer (dict): A dictionary where keys are absorb layers and values are lists
                of layers to be smoothed.
            calib_iter (int): Data size for calibration.

        Returns:
            dict: A dictionary containing the layer names and channel-wise max value information.
        """
        ##hook all the module
        self.input_mins = {}
        self.input_maxes = {}

        hook_modules = {}
        for n, module in self.model.named_modules():
            if isinstance(module, tuple(op_types)):
                hook_modules[n] = module

        self._add_min_max_observer(hook_modules)

        self._dump_min_max(calib_iter=calib_iter)
        self._remove_observer()
        return self.input_mins, self.input_maxes


class GraphTrace:  # pragma: no cover
    """GraphTrace Class."""

    def __init__(self):
        """Initialize the GraphTrace class with supported operations and layers.

        Attributes:
            supported_torch_module_to_aten (dict): A mapping from PyTorch module names
                to their corresponding ATen operation names.
            skip_ops_to_find_absorb (list of str): A list of ATen operations that should
                be skipped when searching for operations to absorb.
            could_absorb_layers (list of str): A list of ATen operations that are eligible
                for absorption during graph tracing.
        """
        self.supported_torch_module_to_aten = {
            "Linear": "aten::linear",
            "Conv2d": "aten::_convolution",
            "ConvTranspose2d": "aten::_convolution",
            "LayerNorm": "aten::layer_norm",
            "BatchNorm2d": "aten::batch_norm",
            "GroupNorm": "aten::group_norm",
            "InstanceNorm2d": "aten::instance_norm",
            "LlamaRMSNorm": "aten::mul",
            "T5LayerNorm": "aten::mul",
            "LPLayerNorm": "aten::layer_norm",  ##mpt_chat
        }

        ##TODO potential bug, need to check only have one bug
        ##TODO, must satisfy af(x)=f(ax),current skip layer may be incomplete
        self.skip_ops_to_find_absorb = ["aten::to", "aten::relu", "aten::leaky_relu", "aten::hardtanh"]

        self.could_absorb_layers = [
            "aten::layer_norm",
            "aten::batch_norm",
            "aten::linear",
            "aten::_convolution",
            "aten::group_norm",
            "aten::instance_norm",
            "aten::mul",
        ]  ##TODO,support more norm

    def trace(self, model, dummy_input):
        """Trace and freeze a model using TorchScript, handling various input formats and devices.

        Args:
            model (torch.nn.Module): The model to be traced and frozen.
            dummy_input (Tensor, dict, or tuple): A dummy input or a dictionary of inputs
                for tracing the model.

        Returns:
            torch.jit.ScriptModule or None: The traced and frozen model, or None if tracing failed.

        Raises:
            Exception: Logs warnings if tracing or freezing the model fails.
        """
        traced_model = None
        optimize_numerics = False
        orig_device = str(next(model.parameters()).device)
        if orig_device != "cpu" and orig_device != "meta":  # pragma: no cover
            model = model.to("cpu")
            dummy_input = move_input_to_device(dummy_input, "cpu")
        if isinstance(dummy_input, dict) or isinstance(dummy_input, UserDict):
            try:
                traced_model = torch.jit.trace(
                    model, example_kwarg_inputs=dict(dummy_input), strict=False, check_trace=False
                )
                traced_model = torch.jit.freeze(traced_model.eval(), optimize_numerics=optimize_numerics)
            except Exception as e:
                logger.warning(e)
                logger.warning("Jit trace in GraphTrace failed, absorb layer detection is skipped")
        else:
            try:
                traced_model = torch.jit.trace(model, dummy_input, strict=False)
                traced_model = torch.jit.freeze(traced_model.eval(), optimize_numerics=optimize_numerics)
            except:
                try:
                    traced_model = torch.jit.trace(model, dummy_input[0], strict=False)
                    traced_model = torch.jit.freeze(traced_model.eval(), optimize_numerics=optimize_numerics)
                except Exception as e:
                    logger.warning(e)
                    logger.warning("Jit trace in GraphTrace failed, absorb layer detection is skipped")
        model = model.to(orig_device)
        return traced_model

    def get_nodes(self, traced_model, op_types=["Linear"]):
        """Extract nodes of specified types from a traced model's computation graph.

        Args:
            traced_model (torch.jit.ScriptModule): The traced and frozen model.
            op_types (list or str, optional): The types of operations to extract nodes for.
                Defaults to ["Linear"].

        Returns:
            list of tuple: A list of tuples where each tuple contains a node
                and its operation type.
        """
        if isinstance(op_types, str):
            op_types = [op_types]
        nodes = []
        for node in traced_model.graph.nodes():
            node_type = node.kind()
            for op_type in op_types:
                if node_type == op_type:
                    nodes.append((node, op_type))
                    break
        return nodes

    def get_prev_absorb_layer(self, nodes):
        """Find previous layers that can be absorbed based on the given nodes.

        Args:
            nodes (list): A list of nodes for which to find absorbable previous layers.

        Returns:
            list: A list of previous layers that can be absorbed, or None if no suitable layer is found.
        """
        prev_absorb_layer = []
        for node in nodes:
            parent = get_parent(node)
            while 1:
                if parent.kind() in self.skip_ops_to_find_absorb:
                    parent = get_parent(parent)
                    continue
                if parent.kind() in self.could_absorb_layers:
                    parent_out_kinds = []
                    for val_user in list(parent.outputs())[0].uses():
                        next_node = val_user.user
                        parent_out_kinds.append(next_node.kind())
                    parent_out_kinds = set(parent_out_kinds)
                    parent_out_kinds.discard("aten::size")

                    if parent_out_kinds == parent_out_kinds.intersection(self.could_absorb_layers):
                        prev_absorb_layer.append(parent)
                    elif parent_out_kinds.intersection(self.skip_ops_to_find_absorb):
                        res = self.skip_op_absorb_helper(parent)
                        prev_absorb_layer.append(parent) if res else prev_absorb_layer.append(None)
                    else:  # When parent to multiple ops, sq transformation could be wrong.
                        prev_absorb_layer.append(None)
                else:
                    prev_absorb_layer.append(None)
                break
        return prev_absorb_layer

    def skip_op_absorb_helper(self, parent_node):
        """Helper function to determine if a node should be skipped for absorption based on its outputs.

        Args:
            parent_node (torch.jit.Node): The node to evaluate for absorption suitability.

        Returns:
            bool: True if the node can be absorbed, False otherwise.
        """
        for val_user in list(parent_node.outputs())[0].uses():
            next_node = val_user.user
            if next_node.kind() == "aten::size":
                continue
            elif next_node.kind() in self.could_absorb_layers:
                continue
            elif next_node.kind() in self.skip_ops_to_find_absorb:
                node_res = self.skip_op_absorb_helper(next_node)
                if not node_res:
                    return False
            else:
                return False
        return True

    def mapping_torch_module_to_aten(self, op_types):
        """Map specified torch module operation types to their corresponding ATen operation types.

        Args:
            op_types (list of str): A list of operation types to be mapped from torch module to ATen.

        Returns:
            list: A list of unique ATen operation types corresponding to the provided torch module operation types.
        """
        res = []
        for op in op_types:
            if op not in self.supported_torch_module_to_aten.keys():
                logger.warning(f"{op} is not supported in smooth quant, ignoring...")
                continue
            res.append(self.supported_torch_module_to_aten[op])
        res = list(set(res))
        return res

    def _check_valid_conv(self, module):
        """Remove group convolution layers except depthwise convolution.

        Args:
            module (torch.nn.Module): The module to process.

        Returns:
            None
        """
        if not isinstance(module, torch.nn.Conv2d):
            return True
        if module.groups > 1:
            if module.in_channels == module.out_channels and module.groups == module.in_channels:
                return True
            else:
                return False
        return True

    def get_absorb_to_layer(self, model, example_input, op_types, skip_unsupported_layers=True):
        """Determine which layers in the model can be absorbed by other layers and map them accordingly.

        Args:
            model (torch.nn.Module): The model to analyze for absorbable layers.
            example_input (Tensor, dict, or tuple): Example input to trace the model.
            op_types (list of str): List of operation types to be considered for absorption.
            skip_unsupported_layers (bool, optional): Whether to exclude layers that are not supported.
                Defaults to True.

        Returns:
            absorb_to_layer (dict): A dictionary mapping absorbable layer names to the layers they can absorb.
            no_absorb_layers (list): A list of layer names that could not be absorbed.
        """
        traced_model = self.trace(model, example_input)
        if traced_model is None:
            return None, None

        aten_op_types = self.mapping_torch_module_to_aten(op_types)
        nodes_types = self.get_nodes(traced_model, aten_op_types)
        nodes = [node_type[0] for node_type in nodes_types]
        nodes_prev_absorb = self.get_prev_absorb_layer(nodes)
        absorb_to_layer = {}
        no_absorb_layers = []
        for index, absorb in enumerate(nodes_prev_absorb):
            if absorb is None:
                no_absorb_layers.append(".".join(nodes[index].scopeName().split("/")[-1].split(".")[1:]))
                continue
            node = nodes[index]
            layer_name = ".".join(node.scopeName().split("/")[-1].split(".")[1:])
            absorb_name = ".".join(absorb.scopeName().split("/")[-1].split(".")[1:])
            if layer_name == "" or absorb_name == "":
                continue
            if absorb_name in absorb_to_layer.keys():
                absorb_to_layer[absorb_name].append(layer_name)
            else:
                absorb_to_layer[absorb_name] = [layer_name]
        if skip_unsupported_layers:
            absorb_to_layer = self.remove_unsupported_layers(model, absorb_to_layer, no_absorb_layers)
        return absorb_to_layer, no_absorb_layers

    def remove_unsupported_layers(self, model, absorb_to_layer, no_absorb_layers):
        """Filter out unsupported layers from the absorb-to-layer mapping based on model's layer types.

        Args:
            model (torch.nn.Module): The model containing layers to be checked.
            absorb_to_layer (dict): A dictionary mapping absorbable layer names to layers they can absorb.
            no_absorb_layers (list): A list to collect names of layers that cannot be absorbed.

        Returns:
            dict: A dictionary with only the supported layers, mapping absorbable layer names
                to valid layers they can absorb.
        """
        res = {}
        for key in absorb_to_layer.keys():
            absorb_layer = get_module(model, key)
            layer_type = absorb_layer.__class__.__name__
            if layer_type not in self.supported_torch_module_to_aten.keys():
                no_absorb_layers.extend(absorb_to_layer[key])
                continue
            supported = True
            for layer_name in absorb_to_layer[key]:
                layer = get_module(model, layer_name)
                layer_type = layer.__class__.__name__
                if (layer_type not in self.supported_torch_module_to_aten.keys()) or not self._check_valid_conv(layer):
                    supported = False
                    no_absorb_layers.extend(absorb_to_layer[key])
                    break
            if supported:
                res[key] = absorb_to_layer[key]
        return res


@register_autotune("version1")
class AutoAlpha:  # pragma: no cover
    """AutoAlpha Class."""

    def __init__(
        self,
        model,
        dataloader,
        absorb_to_layer,
        op_types,
        device,
        q_func,
        example_inputs,
        weight_clip=True,
        alpha_min=0.3,
        alpha_max=0.7,
        alpha_step=0.1,
        shared_criterion="mean",
        init_alpha=0.5,
        folding=False,
        do_blockwise=False,
        n_samples=32,
    ):
        """Initialize the AutoAlpha tuner with necessary parameters and components."""
        self.model = model.to("cpu")
        self.model.eval()
        self.dataloader = dataloader
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_step = alpha_step
        self.shared_criterion = shared_criterion
        self.init_alpha = init_alpha
        self.loss_type = "blockwise" if do_blockwise else "model_wise"
        self.calib_sample_num = n_samples if n_samples else 32
        self.op_types = op_types
        self.absorb_to_layer = absorb_to_layer
        self.weight_scale_dict = {}
        self.q_func = q_func
        self.folding = folding
        self.example_inputs = example_inputs
        self.max_value_info = {}  # to record max values for alpha tune
        self.weight_clip = weight_clip[0] if isinstance(weight_clip, tuple) else weight_clip
        self.input_maxes = {}
        self.input_mins = {}
        self.input_maxes_abs = {}
        self.device = device

    def tune(self):
        """The main entry of auto_alpha.

        Returns:
            tuple: Optimal alpha values and scales based on user-defined recipes.
        """
        calib = Calibration(self.model, self.dataloader, self.q_func, self.device)
        calib_iter = 100
        self.input_mins, self.input_maxes = calib.calibrate(calib_iter, self.op_types)
        for key in self.input_mins.keys():
            self.input_maxes_abs[key] = torch.max(torch.abs(self.input_mins[key]), torch.abs(self.input_maxes[key]))

        if not self.folding:
            diff_modules = set(self.absorb_to_layer.keys()).difference(self.input_mins.keys())
            for d in diff_modules:
                del self.absorb_to_layer[d]

        scale_memo_use = 0
        for key in self.absorb_to_layer:
            layer_name = self.absorb_to_layer[key][0]
            input_max = self.input_maxes_abs[layer_name]
            scale_memo_use += 4 * input_max.shape[0] * len(self.absorb_to_layer[key])
        alpha_space_len = (self.alpha_max - self.alpha_min) / self.alpha_step + 1
        scale_memo_use *= alpha_space_len
        self._save_scale = enough_memo_store_scale(self.device, scale_memo_use)

        if self.loss_type == "blockwise":
            self.block_names = self.get_blocks()
            logger.info("Blockwise auto-tuning will be performed")
            module_names = self._get_sq_layer_names()
            block_names, self.block_to_module = self.block_names, {}
            for block in block_names:
                self.block_to_module[block] = []
            for module in module_names:
                checked = False
                for block in block_names:
                    if block + "." in module:
                        self.block_to_module[block].append(module)
                        checked = True
                if not checked:
                    self.block_to_module[module] = [module]
            self.block_names = list(self.block_to_module.keys())
            logger.info(f"Blockwise auto-tuning: {len(self.block_names)} blocks found")
            logger.debug(f"Blockwise auto-tuning blocks info: {self.block_to_module}")
            return self._auto_tune_alpha_blockwise()
        else:
            return self._auto_tune_alpha()

    def get_blocks(self):
        """Obtain a list of blocks in block-wise tuning mode."""
        block_names = []
        for n, m in self.model.named_modules():
            if hasattr(type(m), "__name__") and "ModuleList" in type(m).__name__:
                for nn, mm in m.named_children():
                    block_name = n + "." + nn
                    block_names.append(block_name)
                break
        return block_names

    def _add_blockwise_observer(self, block_modules):
        """Insert observers into the block modules.

        Args:
            block_modules (list): The block modules to which the observer will be inserted.

        Returns:
            None
        """
        self.blockwise_hook_handles = []
        for key in block_modules.keys():
            hook_func = self._save_blockwise_hook(key)
            hook_handle = block_modules[key].register_forward_hook(hook_func)
            self.blockwise_hook_handles.append(hook_handle)

    def _save_blockwise_hook(self, name):
        """A forward hook to save inputs and outputs of a block.

        Args:
            name (str): The block name.

        Returns:
            function: A hook function.
        """

        def save_blockwise_hook(module, inputs, outputs):
            self.block_inputs[name] = inputs[0]
            self.block_outputs[name] = outputs[0]

        return save_blockwise_hook

    def _get_all_hook_module_names(self):
        """Obtain all the modules that could be hooked based on given op_types."""
        module_names = []
        for n, module in self.model.named_modules():
            if isinstance(module, tuple(self.op_types)):
                module_names.append(n)
        return module_names

    def _update_scales_for_auto(self, absorb_scales, weight_scales):
        """Apply activation and weight scales to the model."""
        for key in self.absorb_to_layer.keys():
            layer_names = self.absorb_to_layer[key]
            for layer_name in layer_names:
                layer = get_module(self.model, layer_name)
                input_scale = absorb_scales[key]
                weight_scale = weight_scales[layer_name]
                input_scale = reshape_scale_as_input(layer, input_scale)
                weight_scale = reshape_scale_as_weight(layer, weight_scale)
                layer.update_scale(input_scale, weight_scale)  ##FIXME

    def _change_qdq_for_auto(self, enable=True):
        """Change the option for qdq."""
        module_names = self._get_all_hook_module_names()
        for name in module_names:
            name = name.split(".orig_layer")[0]
            module = get_module(self.model, name)
            if not hasattr(module, "orig_layer"):  # skip module if it's not used in calibration
                continue
            if enable:
                module.enable_quant()
            else:
                module.disable_quant()

    def _qdq_model_wrapper_for_auto(self, save_q_input=False):
        """Wrap all the modules with QDQ (Quantize-Dequantize) operations.

        Returns:
            None
        """
        module_names = self._get_all_hook_module_names()
        self.to_unwrap_module_names = module_names
        for name in module_names:
            if name not in self.input_mins:  # skip module if it's not used in calibration
                continue
            module = get_module(self.model, name)
            new_module = WrapperLayer(module, self.input_mins[name], self.input_maxes[name], save_q_input=save_q_input)
            set_module(self.model, name, new_module)

    def _qdq_model_unwrapper_for_auto(self):
        """Unwrap all the modules from QDQ (Quantize-Dequantize) operations.

        Returns:
            None
        """
        module_names = self.to_unwrap_module_names
        for name in module_names:
            module = get_module(self.model, name)
            if not hasattr(module, "orig_layer"):  # skip module if it's not used in calibration
                continue
            set_module(self.model, name, module.orig_layer)

    def _cal_scales(self, absorb_to_layer, input_maxes, alpha=0.5):
        """Calculate the adjustment scales.

        Args:
            absorb_to_layer (dict): A dictionary mapping absorb layers to smooth quantized layers.
            input_maxes (dict): The channel-wise input max information for layers.
            alpha (float or dict): Alpha value to balance the quantization difficulty of activation and weight.

        Returns:
            dict: A dictionary containing the calculated adjustment scales.
        """
        absorb_to_input_maxes = {}
        for key in absorb_to_layer.keys():
            layer_name = absorb_to_layer[key][0]
            absorb_to_input_maxes[key] = input_maxes[layer_name]

        weight_scales_info = {}
        absorb_scales_info = {}
        for index, key in enumerate(absorb_to_layer.keys()):
            alpha_tmp = alpha[key] if isinstance(alpha, dict) else alpha
            if alpha_tmp < 0:
                scale = torch.ones((1), device=self.device)
            else:
                input_max = absorb_to_input_maxes[key]
                layer_names = absorb_to_layer[key]
                weights = []
                for layer_name in layer_names:
                    weight = reshape_in_channel_to_last(layer_name, self.model)
                    weights.append(weight)

                weight_max_per_channel = torch.max(torch.abs(torch.cat(weights, dim=0)), dim=0)[0]
                if self.weight_clip:
                    weight_max_per_channel = weight_max_per_channel.clamp(min=1e-5)

                if self._save_scale:
                    if key in self.weight_scale_dict and alpha_tmp in self.weight_scale_dict[key]:
                        scale = self.weight_scale_dict[key][alpha_tmp]
                    else:
                        scale = cal_scale(input_max, weights, alpha_tmp)
                else:
                    scale = cal_scale(input_max, weights, alpha_tmp)

            absorb_scales_info[key] = 1.0 / scale
            absorb_scales_info[key][scale == 0] = 0
            layer_names = absorb_to_layer[key]
            for layer_name in layer_names:
                ##self._scale_layer_weight(layer_name, scale)
                weight_scales_info[layer_name] = scale
                if self._save_scale:
                    if layer_name not in self.weight_scale_dict:
                        self.weight_scale_dict[layer_name] = {}
                    self.weight_scale_dict[layer_name][alpha_tmp] = scale
        return absorb_scales_info, weight_scales_info

    def _get_auto_loss(self, output, output_q, loss_type="abs", loss_alpha=1.0):
        """Get the loss for auto-tuning.

        Args:
            output (Tensor): FP32 output for one layer.
            output_q (Tensor): Quantized output for one layer.
            loss_type (str): The type of loss.
            loss_alpha (float): Loss alpha value for mean scale error.

        Returns:
            Tensor: A tensor containing the calculated loss.
        """
        if len(output.shape) <= 2:
            max_value = torch.max(torch.abs(output))
        else:
            output = output.reshape(output.shape[0], -1)
            output_q = output_q.reshape(output_q.shape[0], -1)
            max_value = torch.max(torch.abs(output), dim=-1).values.unsqueeze(-1)
            max_value = torch.clip(max_value, 1e-5)
        output = output / max_value  ##FIXME need copy not replace
        output_q = output_q / max_value
        if loss_type == "abs":
            return torch.sum(torch.pow(torch.abs(output - output_q), 0.5))
        else:
            return torch.sum((output - output_q) ** 2)

    def _get_sq_layer_names(self):
        """Get all the layers that could be smooth quantized.

        Returns:
            list: All the smooth quantization layer names.
        """
        ##TODO this may not fit for folding=False
        module_names = []
        for key in self.absorb_to_layer:
            module_names += self.absorb_to_layer[key]
        return module_names

    def _get_best_alpha(self, absorb_to_layer, loss_alphas, shared_criterion):
        """Obtain the optimal alpha values based on shared criteria and loss values recorded in the auto-tuning step.

        Returns:
            dict: A dictionary of layerwise alpha values.
        """

        def dict_to_list(dic):
            res = []
            for key in dic.keys():
                res.append((key, dic[key]))
            return res

        best_alpha = {}
        for ln_name in absorb_to_layer.keys():
            layer_names = absorb_to_layer[ln_name]
            cur_shared_criterion = shared_criterion
            if len(layer_names) == 1:
                cur_shared_criterion = "min"
            if cur_shared_criterion == "mean":
                loss_tmp = {}
                for alpha in loss_alphas[layer_names[0]].keys():
                    if alpha not in loss_tmp.keys():
                        loss_tmp[alpha] = 0
                    for layer_name in layer_names:
                        loss_tmp[alpha] += loss_alphas[layer_name][alpha]
                res = dict_to_list(loss_tmp)
                res.sort(key=lambda x: x[1])

                best_alpha[ln_name] = float(res[0][0])

            elif cur_shared_criterion == "min" or cur_shared_criterion == "max":
                tmp_best_alpha = []
                for layer_name in layer_names:
                    res = dict_to_list(loss_alphas[layer_name])
                    res.sort(key=lambda x: x[1])
                    tmp_best_alpha.append(float(res[0][0]))
                if cur_shared_criterion == "min":
                    best_alpha[ln_name] = min(tmp_best_alpha)
                else:
                    best_alpha[ln_name] = max(tmp_best_alpha)

            else:
                raise NotImplementedError
        return best_alpha

    def _get_one_batch_auto_loss(self, input, alpha_space, orig_best_alpha, input_maxes):
        """Calculate the losses for all alpha values given an input.

        Returns:
            dict: A dictionary of operation-wise loss values with respect to alpha values.
        """
        self._change_qdq_for_auto(enable=False)
        module_names = self._get_sq_layer_names()
        forward_wrapper(self.model, input, self.device)  ##disable quant and get fp32 output

        fp32_output = {}
        for name in module_names:
            module = get_module(self.model, name)
            fp32_output[name] = module.output
            module.output = None
        self._change_qdq_for_auto(enable=True)
        absorb_input_scales, weight_scales = self._cal_scales(self.absorb_to_layer, input_maxes, orig_best_alpha)
        self._update_scales_for_auto(absorb_input_scales, weight_scales)
        forward_wrapper(self.model, input, self.device)  ##save quant_input
        for mod_name in module_names:  # save fp32 values
            mod = get_module(self.model, mod_name)
            if mod_name in self.fp32_output_val:
                self.fp32_output_val[mod_name].append(torch.norm(mod.output))
            else:
                self.fp32_output_val[mod_name] = [torch.norm(mod.output)]
            del mod

        loss_alphas = {}
        for name in module_names:
            module = get_module(self.model, name)
            loss = self._get_auto_loss(fp32_output[name], module.output)
            cur_alpha = orig_best_alpha
            if isinstance(orig_best_alpha, dict):
                cur_alpha = orig_best_alpha[name]
            key_name = str(cur_alpha)
            loss_alphas[name] = {key_name: loss}
        # for name in module_names:
        #     loss_alphas[name]={}
        for alpha in alpha_space:
            absorb_input_scales, weight_scales = self._cal_scales(self.absorb_to_layer, input_maxes, alpha)
            self._update_scales_for_auto(absorb_input_scales, weight_scales)
            for name in module_names:
                losses = loss_alphas[name]
                if str(alpha) in losses.keys():
                    continue
                module = get_module(self.model, name)
                output = module.q_dq_forward(module.q_input, module.input_scale, module.weight_scale)
                loss = self._get_auto_loss(fp32_output[name], output)
                loss_alphas[name][str(alpha)] = loss
        return loss_alphas

    def _get_one_batch_auto_loss_blockwise(self, input, alpha_space, orig_best_alpha, input_maxes):
        """Calculate the losses for all alpha values given an input in blockwise tuning mode.

        Returns:
            dict: A dictionary of blockwise loss values with respect to alpha values.
        """
        self._change_qdq_for_auto(enable=False)
        module_names = self._get_sq_layer_names()

        block_modules = {}
        for key in self.block_names:
            block_modules[key] = get_module(self.model, key)
        self._add_blockwise_observer(block_modules)

        forward_wrapper(self.model, input, self.device)  ##disable quant and get fp32 output

        fp32_output = {}
        for block_name in self.block_names:
            fp32_output[block_name] = self.block_outputs[block_name]
        self._change_qdq_for_auto(enable=True)
        absorb_input_scales, weight_scales = self._cal_scales(self.absorb_to_layer, input_maxes, orig_best_alpha)
        self._update_scales_for_auto(absorb_input_scales, weight_scales)
        forward_wrapper(self.model, input, self.device)  ##save quant_input
        for mod_name in module_names:  # save fp32 values
            mod = get_module(self.model, mod_name)
            if mod_name in self.fp32_output_val:
                self.fp32_output_val[mod_name].append(torch.norm(mod.output))
            else:
                self.fp32_output_val[mod_name] = [torch.norm(mod.output)]
            del mod

        loss_alphas = {}

        for block_name in self.block_names:
            block = get_module(self.model, block_name)
            loss = self._get_auto_loss(fp32_output[block_name], self.block_outputs[block_name])
            cur_alpha = orig_best_alpha
            if isinstance(orig_best_alpha, dict):
                cur_alpha = orig_best_alpha[self.block_to_module[block_name][0]]
            key_name = str(cur_alpha)
            loss_alphas[block_name] = {key_name: loss}
        # for name in module_names:
        #     loss_alphas[name]={}
        for alpha in alpha_space:
            absorb_input_scales, weight_scales = self._cal_scales(self.absorb_to_layer, input_maxes, alpha)
            self._update_scales_for_auto(absorb_input_scales, weight_scales)

            for block_name in self.block_names:
                losses = loss_alphas[block_name]
                if str(alpha) in losses.keys():
                    continue
                block = get_module(self.model, block_name)
                block_copy = copy.deepcopy(block)
                for name in self.block_to_module[block_name]:
                    if name == block_name and len(self.block_to_module[block_name]) == 1:
                        module, module_copy = block, block_copy
                    else:
                        module = get_module(block, name)
                        module_copy = copy.deepcopy(module)
                    if module.weight_scale is not None:
                        module_copy.orig_layer.weight *= module.weight_scale
                    q_dq_weight = quant_dequant_w_v1(module_copy.orig_layer)
                    module_copy.orig_layer.weight.data.copy_(q_dq_weight)
                    module_copy.do_blockwise = True
                    if not (name == block_name and len(self.block_to_module[block_name]) == 1):
                        set_module(block_copy, name, module_copy)
                try:
                    output = block_copy(self.block_inputs[block_name])[0]
                except:  # Llama model decoder_layer forward requires position_id
                    position_ids = torch.arange(self.block_inputs[block_name].size()[1])
                    position_ids = position_ids.view(self.block_inputs[block_name].size()[0], -1)
                    output = block_copy(self.block_inputs[block_name], position_ids=position_ids)[0]
                loss = self._get_auto_loss(fp32_output[block_name], output)
                loss_alphas[block_name][str(alpha)] = loss
                del block_copy  # release memory
        return loss_alphas

    def opwise_rank(self, loss_alphas, best_alphas):
        """Rank the final losses of operations based on their ratio with respect to operation output norm.

        Returns:
            dict: A dictionary of ranked operations with their loss ratios.
        """
        max_op, max_ratio, max_key = "", 0, ""
        ratio_info = {}
        for key in self.absorb_to_layer:
            for op_name in self.absorb_to_layer[key]:
                fp32_norm, loss_ = (
                    torch.sum(torch.stack(self.fp32_output_val[op_name])),
                    loss_alphas[op_name][str(best_alphas[key])],
                )
                ratio = loss_ / fp32_norm
                max_op = op_name if ratio > max_ratio else max_op
                max_key = key if ratio > max_ratio else max_key
                max_ratio = max(ratio, max_ratio)
                ratio_info[op_name] = ratio
                logger.debug(
                    f"final loss: {op_name}: {loss_}; @alpha {best_alphas[key]}; \
                    fp32_output norm: {fp32_norm}; ratio: {ratio}"
                )
        import operator

        ratio_info = dict(sorted(ratio_info.items(), key=operator.itemgetter(1), reverse=True))
        for key in list(ratio_info.keys()):
            logger.debug(f"sorted opname-ratio: {key}:  {ratio_info[key]}")
        if max_op != "":
            logger.debug(
                f"max loss: {max_op}: {loss_alphas[max_op][str(best_alphas[max_key])]} @alpha {best_alphas[max_key]}\
                fp32_output norm: {torch.sum(torch.stack(self.fp32_output_val[max_op]))}; ratio: {max_ratio}"
            )
        return None

    def default_tune_setup(self):
        """Setup default auto-tune settings.

        Returns:
            dict: A dictionary of operation-wise loss values with respect to alpha values.
        """
        round_num = max(  # Initialize the alpha search space
            len(str(self.alpha_min).split(".")[1]),
            len(str(self.alpha_max).split(".")[1]),
            len(str(self.alpha_step).split(".")[1]),
        )
        self.alpha_space = numpy.round(
            numpy.arange(self.alpha_min, self.alpha_max + self.alpha_step, self.alpha_step), round_num
        ).tolist()
        ##wrapper new module
        self._qdq_model_wrapper_for_auto(save_q_input=True)

        absorb_input_scales, weight_scales = self._cal_scales(
            self.absorb_to_layer, self.input_maxes_abs, self.init_alpha
        )
        self._update_scales_for_auto(absorb_input_scales, weight_scales)
        return absorb_input_scales, weight_scales

    def _auto_tune_alpha(self):
        """Perform alpha-tuning to obtain layer-wise optimal alpha values and adjust parameters accordingly."""
        logger.info("Start alpha tuning")

        absorb_input_scales, weight_scales = self.default_tune_setup()

        total_cnt, tmp_cnt = 0, 0
        alpha_update_iter, tune_cnt = 0, 4
        # multiply_factor is used to combine samples to calib_sample_num // 4 before summarizing the best alpha
        multiply_factor = (
            self.calib_sample_num // tune_cnt if self.calib_sample_num >= tune_cnt else self.calib_sample_num
        )
        self.fp32_output_val = {}
        best_alphas = self.init_alpha

        if not self.dataloader:
            logger.info("No dataloader, performing auto-tuning with calibration function instead.")
            self.model, self.dataloader = build_captured_dataloader(self.model, self.q_func, self.calib_sample_num)

        bar = tqdm(self.dataloader, total=self.calib_sample_num, desc="auto tune alpha")  # pylint: disable=E1102
        for input in bar:
            if isinstance(input, tuple) or isinstance(input, list):
                if len(input) == 2:
                    input, _ = input  # Extract input when both input and label are yielded by dataloader.

            loss_alphas = {}
            best_alphas_per_module = best_alphas
            if isinstance(best_alphas, dict):
                for key in self.absorb_to_layer.keys():
                    layer_names = self.absorb_to_layer[key]
                    for layer_name in layer_names:
                        best_alphas_per_module[layer_name] = best_alphas_per_module[key]
            loss_tmp = self._get_one_batch_auto_loss(
                input, self.alpha_space, best_alphas_per_module, self.input_maxes_abs
            )
            if loss_alphas == {}:
                loss_alphas = loss_tmp
            else:
                for key in loss_alphas.keys():
                    cur_loss = loss_alphas[key]
                    for alpha_key in cur_loss.keys():
                        cur_loss[alpha_key] += loss_tmp[key][alpha_key]

            total_cnt += 1
            tmp_cnt += 1
            if tmp_cnt // multiply_factor >= 1:
                alpha_update_iter += 1
                tmp_cnt = 0
                best_alphas = self._get_best_alpha(self.absorb_to_layer, loss_alphas, self.shared_criterion)
                for key in best_alphas.keys():
                    logger.info(f"Auto alpha update iter: {alpha_update_iter}, {key}: {best_alphas[key]}")
                absorb_input_scales, weight_scales = self._cal_scales(
                    self.absorb_to_layer, self.input_maxes_abs, best_alphas
                )
                self._update_scales_for_auto(absorb_input_scales, weight_scales)
                # does not need to reset the weight_scale_dict, because use the weight of ori_layer, no change
                # self.weight_scale_dict = {}
            if total_cnt >= self.calib_sample_num:
                break

        best_alphas = self._get_best_alpha(self.absorb_to_layer, loss_alphas, self.shared_criterion)
        for key in best_alphas.keys():
            logger.info(f"Final alpha {key}:{best_alphas[key]}")

        self.opwise_rank(loss_alphas, best_alphas)
        self._qdq_model_unwrapper_for_auto()
        logger.info("auto tuning done")

        return best_alphas

    def _auto_tune_alpha_blockwise(self):
        """Perform blockwise-alpha-tuning to obtain optimal alpha values and adjust parameters accordingly."""
        logger.info("Start block-wise alpha tuning")
        self.block_inputs, self.block_outputs = {}, {}

        absorb_input_scales, weight_scales = self.default_tune_setup()

        total_cnt, tmp_cnt = 0, 0
        alpha_update_iter, tune_cnt = 0, 4
        # multiply_factor is used to combine samples to calib_sample_num // 4 before summarizing the best alpha
        multiply_factor = (
            self.calib_sample_num // tune_cnt if self.calib_sample_num >= tune_cnt else self.calib_sample_num
        )
        self.fp32_output_val = {}
        best_alphas = self.init_alpha

        if not self.dataloader:
            logger.info("No dataloader, performing auto-tuning with calibration function instead.")
            self.model, self.dataloader = build_captured_dataloader(self.model, self.q_func, self.calib_sample_num)

        bar = tqdm(self.dataloader, total=self.calib_sample_num, desc="auto tune alpha")  # pylint: disable=E1102
        for input in bar:
            if isinstance(input, tuple):  # Extract input when both input and label are yielded by dataloader.
                input = input[0]

            loss_alphas = {}
            best_alphas_per_module = best_alphas
            if isinstance(best_alphas, dict):
                for key in self.absorb_to_layer.keys():
                    layer_names = self.absorb_to_layer[key]
                    for layer_name in layer_names:
                        best_alphas_per_module[layer_name] = best_alphas_per_module[key]
            loss_tmp = self._get_one_batch_auto_loss_blockwise(
                input, self.alpha_space, best_alphas_per_module, self.input_maxes_abs
            )
            if loss_alphas == {}:
                for block_name in self.block_names:
                    for key in self.block_to_module[block_name]:
                        loss_alphas[key] = loss_tmp[block_name]
            else:
                for block_name in self.block_names:
                    for key in self.block_to_module[block_name]:
                        cur_loss = loss_alphas[key]
                        for alpha_key in cur_loss.keys():
                            cur_loss[alpha_key] += loss_tmp[block_name][alpha_key]

            total_cnt += 1
            tmp_cnt += 1
            if tmp_cnt // multiply_factor >= 1:
                alpha_update_iter += 1
                tmp_cnt = 0
                best_alphas = self._get_best_alpha(self.absorb_to_layer, loss_alphas, self.shared_criterion)
                for key in best_alphas.keys():
                    logger.info(f"Auto alpha update iter: {alpha_update_iter}, {key}: {best_alphas[key]}")
                absorb_input_scales, weight_scales = self._cal_scales(
                    self.absorb_to_layer, self.input_maxes_abs, best_alphas
                )
                self._update_scales_for_auto(absorb_input_scales, weight_scales)
                # does not need to reset the weight_scale_dict, because use the weight of ori_layer, no change
                # self.weight_scale_dict = {}
            if total_cnt >= self.calib_sample_num:
                break

        best_alphas = self._get_best_alpha(self.absorb_to_layer, loss_alphas, self.shared_criterion)
        for key in best_alphas.keys():
            logger.info(f"Final alpha {key}:{best_alphas[key]}")

        self.opwise_rank(loss_alphas, best_alphas)
        self._qdq_model_unwrapper_for_auto()
        logger.info("block-wise auto tuning done")

        return best_alphas


class TorchSmoothQuant:  # pragma: no cover
    """Fake input channel quantization.

    For more details please refer to:
    [1] SmoothQuant: Accurate and Efficient
    Post-Training Quantization for Large Language Models
    [2] SPIQ: Data-Free Per-Channel Static Input Quantization
    Currently, we only handle the layers whose smooth scale could be absorbed, we will support other layers later.
    We only support inplace mode which means the model weights will be changed, you can call recover function
    to recover the weights if needed
    """

    def __init__(
        self,
        model,
        dataloader=None,
        example_inputs=None,
        q_func=None,
        traced_model=None,
        scale_sharing=True,
        record_max_info=False,
    ):
        """Init TorchSmoothQuant Class.

        Args:
            model (torch.nn.Module): Torch model.
            dataloader (DataLoader): Calibration dataloader.
            traced_model (Optional[torch.jit.ScriptModule]): A specific model that shares the same architecture
                as the model and could be traced by torch.jit. If not supplied, the model will be used instead.

        Returns:
            None
        """
        self.model = model
        if not isinstance(self.model, torch.nn.Module):
            return
        device, dtype = self._get_device()
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self.dtype = dtype
        self.dataloader = dataloader
        self.example_inputs = example_inputs
        self.q_func = q_func
        self.input_maxes = {}
        self.input_mins = {}
        self.input_maxes_abs = {}
        self.traced_model = traced_model
        if self.traced_model is None:
            self.traced_model = self.model
        self.weight_scale_info = {}
        self.absorb_scales_info = {}
        self.scale_sharing = scale_sharing
        self.insert_mul = False
        self.allow_absorb = True
        self.record_max_info = record_max_info
        self.max_value_info = {}  # to record max values for alpha tune
        self.absorb_to_layer = {}
        self.weight_max_lb = 1e-5  ##weight max low bound
        self.weight_scale_dict = {}
        self.sq_scale_info = {}
        self.max_value_info = {}
        self.need_calibration = False

    def _get_device(self):
        """Get the model device.

        Returns:
            torch.device: The device on which the model is located.
        """
        for _, p in self.model.named_parameters():
            return p.data.device, p.data.dtype

    def _scale_layer_weight(self, layer_name, scale, alpha=0.5, input_minmax=None):  ##input channel
        """Scale the layer weights at input channel and depthwise convolution output channel.

        Args:
            layer_name (str): The layer name.
            scale (Tensor): The scale to be multiplied.
            alpha (float): Alpha value for SQLinearWrapper.
            input_minmax (tuple): Input min and max values for SQLinearWrapper.

        Returns:
            None
        """
        layer = get_module(self.model, layer_name)
        if self.insert_mul:
            layer = get_module(self.model, layer_name)
            if isinstance(layer, SQLinearWrapper):
                layer._recover_sq_linear()
                set_module(self.model, layer_name, layer.sq_linear)  ##recover
            else:
                new_module = SQLinearWrapper(layer, 1.0 / scale, input_minmax, alpha)
                set_module(self.model, layer_name, new_module)
        elif self.allow_absorb:
            scale = reshape_scale_as_weight(layer, scale)
            layer.weight = torch.nn.Parameter(layer.weight * scale)
        return scale

    def _absorb_scales(self, layer_name, scale):  ##output channel
        """Absorb the scale to the layer at the output channel.

        Args:
            layer_name (str): The module name.
            scale (Tensor): The scale to be absorbed.
            alpha_key (str): The alpha value passed to SQLinearWrapper.

        Returns:
            None
        """
        if self.insert_mul or not self.allow_absorb:
            return  # absorb is updated in SQLinearWrapper in def _scale_layer_weight

        ##if self.allow absorb
        layer = get_module(self.model, layer_name)
        if layer.__class__.__name__ == "WrapperLayer":
            layer = layer.orig_layer
        if (
            isinstance(layer, torch.nn.BatchNorm2d)
            or isinstance(layer, torch.nn.GroupNorm)
            or isinstance(layer, torch.nn.InstanceNorm2d)
        ):
            if layer.affine:
                layer.weight *= scale
                layer.bias *= scale
            else:
                layer.affine = True
                weight = torch.ones(layer.num_features, device=self.device, dtype=self.dtype) * scale
                layer.weight = torch.nn.Parameter(weight, requires_grad=False)
                bias = torch.zeros(layer.num_features, device=self.device, dtype=self.dtype)
                layer.bias = torch.nn.Parameter(bias, requires_grad=False)
        elif isinstance(layer, torch.nn.LayerNorm):
            if layer.elementwise_affine:
                layer.weight *= scale
                layer.bias *= scale
            else:
                layer.elementwise_affine = True
                weight = torch.ones(layer.num_features, device=self.device, dtype=self.dtype) * scale
                layer.weight = torch.nn.Parameter(torch.ones(weight, requires_grad=False))
                bias = torch.zeros(layer.num_features, device=self.device, dtype=self.dtype)
                layer.bias = torch.nn.Parameter(bias, requires_grad=False)

        elif isinstance(layer, torch.nn.Conv2d):
            ##the order could not be changed
            if hasattr(layer, "bias") and (layer.bias is not None):
                layer.bias *= scale
            scale = scale.view(scale.shape[0], 1, 1, 1)
            layer.weight *= scale

        elif isinstance(layer, torch.nn.Linear):
            if hasattr(layer, "bias") and (layer.bias is not None):
                layer.bias *= scale
            scale = scale.view(scale.shape[0], 1)
            layer.weight *= scale

        elif layer.__class__.__name__ == "LlamaRMSNorm" or layer.__class__.__name__ == "T5LayerNorm":  ##quite tricky
            layer.weight *= scale

        else:
            logger.warning(
                f"found unsupported layer {type(layer)}, try to multiply scale to "
                f"weight and bias directly, this may introduce accuracy issue, please have a check "
            )
            if hasattr(layer, "weight") and layer.weight is not None:
                layer.weight *= scale
            if hasattr(layer, "bias") and layer.bias is not None:
                layer.bias *= scale

    def _export_sq_info(self, absorb_to_layer, input_maxes, alpha=0.5):
        """Export information required for SmoothQuant including scales and min/max values.

        Args:
            absorb_to_layer (dict): A dictionary mapping absorbable layer names to layers they can absorb.
            input_maxes (dict): A dictionary mapping layer names to their channel-wise maximum values.
            alpha (float or dict, optional): Alpha value(s) to balance the quantization difficulty
                of activation and weight. Defaults to 0.5.

        Returns:
            None
        """
        absorb_to_input_maxes = {}
        for key in absorb_to_layer.keys():
            layer_name = absorb_to_layer[key][0]
            absorb_to_input_maxes[key] = input_maxes[layer_name]

        for index, key in enumerate(absorb_to_layer.keys()):
            alpha_tmp = alpha[key] if isinstance(alpha, dict) else alpha
            layer_names = absorb_to_layer[key]
            weights = []
            for layer_name in layer_names:
                weight = reshape_in_channel_to_last(layer_name, self.model)
                weights.append(weight)
            weight_max_per_channel = torch.max(torch.abs(torch.cat(weights, dim=0)), dim=0)[0]

            weight_max_per_channel = weight_max_per_channel.clamp(min=self.weight_max_lb)

            input_max = absorb_to_input_maxes[key]
            layer_names = absorb_to_layer[key]
            # weight_scale = cal_scale(input_max, weights, alpha_tmp)
            input_minmax = [self.input_mins[layer_names[0]].to("cpu"), self.input_maxes[layer_names[0]].to("cpu")]
            abs_input_max = torch.max(torch.abs(input_minmax[0]), torch.abs(input_minmax[1]))
            input_power = torch.pow(abs_input_max, alpha_tmp)
            weight_power = torch.pow(weight_max_per_channel, 1 - alpha_tmp)
            weight_scale = torch.clip(input_power / weight_power, min=1e-5)

            input_scale = 1.0 / weight_scale

            self.max_value_info[key] = {
                "alpha": alpha_tmp,
                "input_minmax": input_minmax,
                "weight_max": weight_max_per_channel,
                "absorbed_layer": layer_names,
            }  # max_value_info is used for pytorch backend and sq_scale_info is used for ipex backend.
            # the input of layers with same absorb layer is the same.
            for op_name in layer_names:
                module = copy.deepcopy(get_module(self.model, op_name))
                new_module = SQLinearWrapper(module, 1.0 / weight_scale, input_minmax, alpha_tmp)
                self.sq_scale_info[op_name] = {}
                self.sq_scale_info[op_name] = {
                    "alpha": alpha_tmp,
                    "input_scale_for_mul": input_scale.to("cpu"),
                    "input_scale_after_mul": new_module.scale,
                    "input_zero_point_after_mul": new_module.zero_point,
                    "input_dtype": new_module.dtype,
                    "weight_scale_after_mul": new_module._get_weight_scale(),
                }

    def _cal_scales(self, absorb_to_layer, input_maxes, alpha=0.5):
        """Calculate the adjustment scales.

        Args:
            absorb_to_layer (dict): A dictionary mapping absorb layers to smooth quantized layers.
            input_maxes (dict): The channel-wise input max information for layers.
            alpha (float or dict): Alpha value to balance the quantization difficulty of activation and weight.

        Returns:
            dict: A dictionary containing the calculated adjustment scales.
        """
        absorb_to_input_maxes = {}
        for key in absorb_to_layer.keys():
            layer_name = absorb_to_layer[key][0]
            absorb_to_input_maxes[key] = input_maxes[layer_name]

        weight_scales_info = {}
        absorb_scales_info = {}
        for index, key in enumerate(absorb_to_layer.keys()):
            alpha_tmp = alpha[key] if isinstance(alpha, dict) else alpha

            input_max = absorb_to_input_maxes[key]
            layer_names = absorb_to_layer[key]
            weights = []
            for layer_name in layer_names:
                weight = reshape_in_channel_to_last(layer_name, self.model)
                weights.append(weight)
            scale = cal_scale(input_max, weights, alpha_tmp)
            absorb_scales_info[key] = 1.0 / scale
            absorb_scales_info[key][scale == 0] = 0
            layer_names = absorb_to_layer[key]
            for layer_name in layer_names:
                ##self._scale_layer_weight(layer_name, scale)
                weight_scales_info[layer_name] = scale
        return absorb_scales_info, weight_scales_info

    def _adjust_parameters(self, absorb_to_layer, input_maxes, alpha=0.5):
        """Adjust the weights and biases.

        Args:
            absorb_to_layer (dict): A dictionary mapping absorb layers to smooth quantized layers.
            input_maxes (dict): The channel-wise input max information for layers.
            alpha (float or dict): Alpha value to balance the quantization difficulty of activation and weight.

        Returns:
            None
        """
        absorb_scales_info, weight_scales_info = self._cal_scales(absorb_to_layer, input_maxes, alpha)
        if not absorb_scales_info or not weight_scales_info:
            return weight_scales_info, absorb_scales_info
        for index, key in enumerate(absorb_to_layer.keys()):
            if isinstance(alpha, float):
                alpha_tmp = alpha
            elif isinstance(alpha, dict):
                alpha_tmp = alpha[key]
            absorb_scale = absorb_scales_info[key]
            self._absorb_scales(key, absorb_scale)
            layer_names = absorb_to_layer[key]
            for layer_name in layer_names:
                input_minmax = [self.input_mins[layer_names[0]], self.input_maxes[layer_names[0]]]
                self._scale_layer_weight(layer_name, weight_scales_info[layer_name], alpha_tmp, input_minmax)
        return weight_scales_info, absorb_scales_info

    def _check_need_calibration(self, alpha, percentile, op_types, scales_per_op, calib_iter):
        """Check if calibration is needed.

        Args:
            alpha (float or dict): Current alpha values.
            percentile (float): Current percentile.
            op_types (list): Current operation types.
            scales_per_op (dict): Current scales per operation.
            calib_iter (int): Current calibration iterations.

        Returns:
            bool: True if calibration is needed, False otherwise.
        """
        need_calib = True
        from peft import PeftModel  # pylint: disable=E0401

        is_peft, is_auto = isinstance(self.model, PeftModel), alpha == "auto"
        if len(self.input_maxes) == 0:  ## the first time
            need_calib = True
            self.alpha = alpha
            self.percentile = percentile
            self.op_types = op_types
            self.scales_per_op = scales_per_op
            self.calib_iter = calib_iter
            return False if (is_auto and not is_peft) else need_calib

        if (
            self.percentile == percentile
            and self.op_types == op_types
            and self.scales_per_op == scales_per_op
            and self.calib_iter == calib_iter
        ):
            if isinstance(alpha, float) or self.alpha == "auto":
                need_calib = False

        self.alpha, self.percentile, self.calib_iter = alpha, percentile, calib_iter
        self.op_types, self.scales_per_op = op_types, scales_per_op
        return need_calib

    @torch.no_grad()
    def _parse_absorb_to_layers(self, op_types, folding):
        """Parse and map layers in the model for smooth quantization based on specified operation types.

        Args:
            op_types (list): List of operation types (e.g., ["Linear"]) to consider for quantization.
            folding (bool): Flag indicating whether to insert multiplication operations (False) or
                just handle foldable layers (True) for quantization.

        Returns:
            dict or None: Dictionary mapping absorb layer names to lists of layers that can be quantized.
                If tracing fails or no layers can be quantized, returns None.
        """
        str_op_types = [i.__name__ for i in op_types]
        self_absorb_layers = {}
        if self.insert_mul:
            self_absorb_layers = self._get_all_layer_names(op_types)  # TODO: only support linear now.
            # fetch modules with the same input
            group_modules = self._trace(str_op_types, skip_unsupported_layers=False)
            if group_modules is not None:
                # use one input for qkv
                for k, v in group_modules.items():
                    for i in v:
                        if i in self_absorb_layers:
                            self_absorb_layers.pop(i)
                    self_absorb_layers[v[0]] = v
                logger.debug(f"self_absorb_layers:{self_absorb_layers}")
        if self.allow_absorb:
            self.absorb_to_layer, no_absorb_layers = self._trace(str_op_types)
            if self.absorb_to_layer is None and no_absorb_layers is None:
                return None

        # remove self.self_absorb_layers if it exists in self.absorb_to_layer
        for k, v in self.absorb_to_layer.items():
            for i in v:
                if i in self_absorb_layers:
                    self_absorb_layers.pop(i)
        self.absorb_to_layer.update(self_absorb_layers)

        if self.absorb_to_layer is None and no_absorb_layers is None:
            logger.warning(
                "sorry, could not trace the model, smooth quant is ignored."
                "If you are using huggingface model,"
                "you could set torchscript to True "
            )
            return None

        # Check if input_maxes match self.absorb_to_layer
        # (due to self._get_all_layer_names use layer tree instead of forward_path)
        if not folding and self.need_calibration:
            if len(self.input_mins) == 0:  ##there are some modules not used in forward
                calib = Calibration(self.model, self.dataloader, self.q_func, self.device)  ##
                input_mins, input_maxes = calib.calibrate(
                    1, op_types
                )  ##TODO if using qfunc for calibration, it will calibrate twice
            # use qfunc to calibrate, the input min could be used for fixed alpha transformation
            self.input_mins = input_mins
            self.input_maxes = input_maxes
            diff_modules = set(self.absorb_to_layer.keys()).difference(input_mins.keys())
            for d in diff_modules:
                del self.absorb_to_layer[d]
        return self.absorb_to_layer

    @torch.no_grad()
    def transform(
        self,
        alpha=0.5,
        folding=False,
        percentile=100,
        op_types=[torch.nn.Linear, torch.nn.Conv2d],
        scales_per_op=False,
        calib_iter=100,
        weight_clip=True,
        scale_sharing=True,
        auto_alpha_args={
            "init_alpha": 0.5,
            "alpha_min": 0.0,
            "alpha_max": 1.0,
            "alpha_step": 0.1,
            "shared_criterion": "mean",
            "n_samples": 32,  ##512 for cuda, 128 for cpu?
        },
    ):
        """The main entry of SmoothQuant.

        Args:
            alpha (float or dict): Alpha value to balance the quantization difficulty of activation and weight.
                Please refer to the paper for more details.
            folding (bool): Whether to insert multiplication (False) or just allow foldable layers (True)
                for SmoothQuant.
            percentile (float): Not supported currently.
            op_types (list): The operation types to be smooth quantized.
            scales_per_op (dict): Not supported currently.
            calib_iter (int): Data size for calibration.
            weight_clip (bool): Whether to clip weight_max when calculating scales.
            auto_alpha_args (dict): Hyperparameters used to set the alpha search space in SQ auto-tuning.
                By default, the search space is 0.0-1.0 with step_size 0.1.
            do_blockwise (bool): Whether to perform blockwise auto-tuning.
            init_alpha (float): A hyperparameter used in SQ auto-tuning; by default, it is 0.5.

        Returns:
            torch.nn.Module: An FP32 model with the same architecture as the original model
                but with modified weights, which will benefit quantization.
        """
        if not isinstance(self.model, torch.nn.Module):
            logger.warning("smoothquant is ignored since the model is not a torch module")
            return self.model

        if isinstance(alpha, float) and (alpha < 0):
            logger.warning("reset alpha to >=0")
            alpha = numpy.clip(alpha, 0.0)

        if folding:
            self.insert_mul, self.allow_absorb = False, True
        else:
            self.insert_mul, self.allow_absorb = True, False
        self.weight_clip = weight_clip

        self.revert()
        self.need_calibration = self._check_need_calibration(alpha, percentile, op_types, scales_per_op, calib_iter)
        with torch.no_grad():
            str_op_types = [i.__name__ for i in op_types]
            input_maxes_abs = self.input_maxes_abs
            if self.need_calibration:  ##avoid multiple calibaration during tuning if the only difference is alpha
                if self.insert_mul:
                    self.self_absorb_layers = self._get_all_layer_names(op_types)  # TODO: only support linear now.
                    if self.scale_sharing:
                        # fetch modules with the same input
                        group_modules = self._trace(str_op_types, skip_unsupported_layers=False)
                        if group_modules is not None:
                            # use one input for qkv
                            for k, v in group_modules.items():
                                for i in v:
                                    if i in self.self_absorb_layers:
                                        self.self_absorb_layers.pop(i)
                                self.self_absorb_layers[v[0]] = v
                            logger.debug(f"self_absorb_layers:{self.self_absorb_layers}")

        self.absorb_to_layer = self._parse_absorb_to_layers(
            op_types, folding
        )  ##need to forward to check modules not used in forward
        if len(self.input_mins) != 0:  ##this is from _parse_absorb_to_layers, ugly code to support q_func
            input_maxes_abs = {}
            for key in self.input_mins.keys():
                input_maxes_abs[key] = torch.max(torch.abs(self.input_mins[key]), torch.abs(self.input_maxes[key]))
            if self.q_func:
                self.need_calibration = False  # Avoid double-calibration in fixed-value alpha SQ.

        if self.absorb_to_layer is None:
            logger.warning("empty absorb_to_layer, smoothquant is ignored ")
            return self.model
        example_inputs = self._get_example_input()
        if alpha == "auto":  ##TODO need to polish later
            auto_alpha_version = "version1"
            auto_alpha_tuner = TUNERS[auto_alpha_version](
                self.model,
                self.dataloader,
                self.absorb_to_layer,
                op_types=op_types,
                device=self.device,
                q_func=self.q_func,
                folding=folding,
                example_inputs=self.example_inputs,
                **auto_alpha_args,
            )
            self.alpha = auto_alpha_tuner.tune()
            input_maxes_abs = auto_alpha_tuner.input_maxes_abs
            self.input_mins, self.input_maxes = auto_alpha_tuner.input_mins, auto_alpha_tuner.input_maxes
            if auto_alpha_tuner.loss_type == "blockwise":
                self.block_names = auto_alpha_tuner.block_names

        elif self.need_calibration:
            calib = Calibration(self.model, self.dataloader, self.q_func, self.device)
            self.input_mins, self.input_maxes = calib.calibrate(calib_iter, op_types)
            input_maxes_abs = {}
            for key in self.input_mins.keys():
                input_maxes_abs[key] = torch.max(torch.abs(self.input_mins[key]), torch.abs(self.input_maxes[key]))

        if example_inputs is not None:
            out_pre_sq = model_forward_per_sample(self.model, example_inputs, self.device)

        if folding:
            self._save_scale = False  ##TODO remove it later

        if self.record_max_info:
            self._export_sq_info(self.absorb_to_layer, input_maxes_abs, self.alpha)
            # # max_info is recorded in self.max_value_info
            # self._adjust_parameters(self.absorb_to_layer, input_maxes_abs, alpha)
            self.model._smoothquant_optimized = False
            return self.model

        self.weight_scale_info, self.absorb_scales_info = self._adjust_parameters(
            self.absorb_to_layer, input_maxes_abs, self.alpha
        )
        self.model._smoothquant_optimized = True

        if example_inputs is not None:
            # Check mathematical equivalency
            out_post_sq = model_forward_per_sample(self.model, example_inputs, self.device)
            if not self.output_is_equal(out_post_sq, out_pre_sq):
                logger.warning(
                    "Mathematical equivelancy of Smoothquant is not preserved. "
                    "Please kindly report this issue to https://github.com/intel/neural-compressor."
                )
        else:
            logger.warning(" Could not get example input, equivelancy check is skipped")

        return self.model

    def output_is_equal(self, out1, out2, atol=1e-04):
        """Compare two outputs to determine if they are approximately equal within a specified tolerance.

        Args:
            out1 (Union[tuple, dict, torch.Tensor]): The first output to compare.
            out2 (Union[tuple, dict, torch.Tensor]): The second output to compare.
            atol (float, optional): The absolute tolerance for the comparison. Default is 1e-04.

        Returns:
            bool: True if the outputs are approximately equal within the tolerance, False otherwise.

        Raises:
            Exception: If any unexpected error occurs during comparison, a warning is logged,
                and True is returned to indicate that automatic checking failed.
        """
        try:
            if isinstance(out1, tuple):
                return all(torch.all(torch.isclose(out1[i], out2[i], atol=atol)) for i in range(len(out1)))
            elif isinstance(out1, dict):
                return all(torch.all(torch.isclose(out1[k], out2[k], atol=atol)) for k in out1.keys())
            elif isinstance(out1, torch.Tensor):
                return torch.all(torch.isclose(out1, out2, atol=atol))
            return False
        except:
            logger.warning(
                "Automatically check failed, Please check equivelancy manually "
                "between out_pre_sq and out_post_sq if necessary."
            )
            return True

    @torch.no_grad()
    def revert(self):
        """Revert the model weights to their original state.

        Returns:
            None
        """
        for key in self.weight_scale_info:
            self._scale_layer_weight(key, 1.0 / self.weight_scale_info[key])
        for key in self.absorb_scales_info:
            self._absorb_scales(key, 1.0 / self.absorb_scales_info[key])
        self.weight_scale_info = {}  ##clear the data
        self.absorb_scales_info = {}

    def _get_all_layer_names(self, op_types=[torch.nn.Linear]):
        """Identify the layers which can be smooth quantized.

        Args:
            op_types (list): The operation types to be smooth quantized.

        Returns:
            dict: A dictionary where the keys are absorb layer names (themselves)
                and the values are lists of layers to be smooth quantized.
        """
        self_absorb_layer = {}
        op_types = [torch.nn.Linear]  # TODO： only support SQLinearWrapper
        for name, module in self.model.named_modules():
            if isinstance(module, tuple(op_types)):
                self_absorb_layer[name] = [name]
        return self_absorb_layer

    def _get_example_input(self):
        """Retrieve an example input from the dataloader or return the pre-stored example inputs.

        Returns:
            Union[torch.Tensor, None]: The example input if available, otherwise None.

        Raises:
            RuntimeError: If an error occurs while fetching inputs from the dataloader.
        """
        if self.dataloader is None and self.example_inputs is None:
            return None
        if self.example_inputs is None:
            try:
                for idx, (input, label) in enumerate(self.dataloader):
                    self.example_inputs = input
                    break
            except:
                for idx, input in enumerate(self.dataloader):
                    self.example_inputs = input
                    break

        return self.example_inputs

    def _trace(self, op_types, skip_unsupported_layers=True):
        """Identify the layers which can be smooth quantized.

        Args:
            op_types (list): The operation types to be smooth quantized.

        Returns:
            dict: A dictionary where keys are absorb layer names and values are lists of
                layers to be smooth quantized.
            list: A list of layers for which no absorb layer was found.
        """
        tg = GraphTrace()
        self._get_example_input()
        absorb_to_layer, no_absorb_layers = tg.get_absorb_to_layer(
            self.traced_model,
            self.example_inputs,
            op_types,
            skip_unsupported_layers=skip_unsupported_layers,
        )
        if not skip_unsupported_layers:
            return absorb_to_layer
        if absorb_to_layer is None and no_absorb_layers is None:
            logger.warning(
                "sorry, could not trace the model, smooth quant is skipped."
                "If you are using huggingface model,"
                "you could set torchscript to True "
                "when loading the model or set the return_dict to False"
            )
        elif absorb_to_layer == {}:
            logger.warning("could not find any layer to be absorbed")
        else:
            to_absorb_cnt = 0
            for key, item in absorb_to_layer.items():
                to_absorb_cnt += len(item)
            logger.info(
                f" {to_absorb_cnt} out of {to_absorb_cnt + len(no_absorb_layers)} "
                f"layers could be absorbed in smooth quant"
            )
        return absorb_to_layer, no_absorb_layers


class SQLinearWrapper(torch.nn.Module):  # pragma: no cover
    """SQLinearWrapper Class."""

    def __init__(self, module, input_scale, input_minmax, alpha=0.5, dtype=torch.quint8):
        """Initialize the class.

        Args:
            module (torch.nn.Module): The module to be wrapped.
            input_scale (Tensor): The scale for input features.
            input_minmax (Tuple[Tensor, Tensor]): The min and max values for input features.
            alpha (float, optional): A parameter for scaling. Defaults to 0.5.
            dtype (torch.dtype, optional): The data type for quantization. Defaults to torch.quint8.
        """
        super().__init__()
        self.register_buffer("input_scale", input_scale)
        self.alpha = alpha
        self.dtype = dtype
        # calculate and only save scale, zero_point to avoid memory usage
        self.scale, self.zero_point = self._calculate_qparams(input_scale, input_minmax, dtype)
        self.add_module("sq_linear", module)
        self._update_sq_linear()
        self.ipex = False  # a flag used for ipex inference

    @property
    def weight(self):
        """Get the weight of the sq_linear module.

        Returns:
            Tensor: The weight of the sq_linear module.
        """
        return self.sq_linear.weight

    def forward(self, X):
        """Forward pass of the module.

        Args:
            X (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the sq_linear module.
        """
        if self.ipex:
            X = self.sq_linear(X)
        else:
            X = torch.mul(X, self.input_scale)
            X = self.sq_linear(X)
        return X

    def _calculate_qparams(self, input_scale, input_minmax, dtype=torch.quint8):
        """Calculate scale and zero-point for quantization.

        Args:
            input_scale (Tensor): The scale for input features.
            input_minmax (Tuple[Tensor, Tensor]): The min and max values for input features.
            dtype (torch.dtype, optional): The data type for quantization. Defaults to torch.quint8.

        Returns:
            Tuple[Tensor, Tensor]: The calculated scale and zero-point.
        """
        if dtype == torch.quint8:
            quant_min, quant_max = 0, 255
        min_val = torch.min(input_minmax[0] * input_scale)
        max_val = torch.max(input_minmax[1] * input_scale)
        # work when min_val bigger than zero.
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.max(scale, torch.tensor([torch.finfo(torch.float32).eps], device=scale.device))
        zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point

    def _get_weight_scale(self):
        """Get the weight scale and zero-point.

        Returns:
            Tensor: The scale of the weight.
        """
        from torch.ao.quantization.observer import default_per_channel_weight_observer

        obs = default_per_channel_weight_observer()
        obs(self.sq_linear.weight)
        scale, _ = obs.calculate_qparams()
        return scale

    def _update_sq_linear(self):
        """Update the sq_linear module by removing the multiplication of scale.

        This method adjusts the weight of sq_linear for ipex inference.
        """
        scale = self.input_scale.view(1, self.input_scale.shape[0])
        with torch.no_grad():
            self.sq_linear.weight /= scale

    def _recover_sq_linear(self):
        """Recover the original sq_linear module by restoring the multiplication of scale.

        This method adjusts the weight of sq_linear for ipex inference.
        """
        scale = self.input_scale.view(1, self.input_scale.shape[0])
        with torch.no_grad():
            self.sq_linear.weight *= scale


class WrapperLayer(torch.nn.Module):  # pragma: no cover
    """WrapperLayer Class."""

    def __init__(self, layer, input_min, input_max, save_q_input=False):
        """Initialize the WrapperLayer.

        Args:
            layer (torch.nn.Module): The original layer to be wrapped.
            input_min (Tensor): Minimum value of the input.
            input_max (Tensor): Maximum value of the input.
            save_q_input (bool, optional): Whether to save the quantized input. Defaults to False.
        """
        super(WrapperLayer, self).__init__()
        self.add_module("orig_layer", layer)  # set orig_layer in get/set_module
        self.quant = False
        self.q_input = None
        self.fp32_output = None
        self.input_max = input_max
        self.input_min = input_min
        self.weight_scale = None
        self.input_scale = None
        self.save_q_input = save_q_input
        self.do_blockwise = False

    def enable_quant(self):
        """Enable quantization for the layer."""
        self.quant = True

    def disable_quant(self):
        """Disable quantization for the layer."""
        self.quant = False

    def update_scale(self, input_scale, weight_scale):
        """Update the input and weight scales.

        Args:
            input_scale (Tensor): The scale for the input.
            weight_scale (Tensor): The scale for the weight.
        """
        self.input_scale = input_scale
        self.weight_scale = weight_scale

    def q_dq_forward(self, x, input_scale, weight_scale):
        """Perform quantization and dequantization forward pass.

        Args:
            x (Tensor): The input tensor.
            input_scale (Tensor): The scale for the input.
            weight_scale (Tensor): The scale for the weight.

        Returns:
            Tensor: The output tensor after quantization and dequantization.
        """
        layer_copy = copy.deepcopy(self.orig_layer)
        if weight_scale is not None:
            layer_copy.weight *= weight_scale
        q_dq_weight = quant_dequant_w_v1(layer_copy)
        layer_copy.weight.data.copy_(q_dq_weight)
        if input_scale is None:
            x = quant_dequant_x_v1(x, self.input_min, self.input_max)
        else:
            x = input_scale * x
            x = quant_dequant_x_v1(x, self.input_min * input_scale, self.input_max * input_scale)  ##FIXME
        output = layer_copy(x)
        return output

    def q_dq_forward_blockwise(self, x, input_scale):
        """Perform blockwise quantization and dequantization forward pass.

        Args:
            x (Tensor): The input tensor.
            input_scale (Tensor): The scale for the input.

        Returns:
            Tensor: The output tensor after blockwise quantization and dequantization.
        """
        layer_copy = copy.deepcopy(self.orig_layer)
        if input_scale is None:
            x = quant_dequant_x_v1(x, self.input_min, self.input_max)
        else:
            x = input_scale * x
            x = quant_dequant_x_v1(x, self.input_min * input_scale, self.input_max * input_scale)  ##FIXME
        output = layer_copy(x)
        return output

    def forward(self, x):
        """Perform the forward pass of the module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        if self.quant:
            # self.q_input = x * scale ##save the q_input
            if self.save_q_input:
                self.q_input = x
            if not self.do_blockwise:
                output = self.q_dq_forward(x, self.input_scale, self.weight_scale)
            else:
                output = self.q_dq_forward_blockwise(x, self.input_scale)
        else:
            output = self.orig_layer(x)
        self.output = output
        return output
