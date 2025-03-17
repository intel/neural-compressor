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
"""Utility functions for Static quantization."""


import copy
import json
import os
import re
from collections import OrderedDict
from typing import Dict, List, Union

import torch
from packaging.version import Version

try:
    import intel_extension_for_pytorch as ipex
except:  # pragma: no cover
    pass

from neural_compressor.common.utils import DEFAULT_WORKSPACE, CpuInfo
from neural_compressor.torch.utils import Statistics, get_ipex_version, get_torch_version, logger

version = get_torch_version()
ipex_ver = get_ipex_version()
ipex_config_path = os.path.join(DEFAULT_WORKSPACE, "ipex_config_tmp.json")

unify_op_type_mapping_ipex = {
    "Convolution_Relu": "Conv2d",
    "Convolution_Sum_Relu": "Conv2d",
    "Convolution_BatchNorm": "Conv2d",
    "<class 'torch.nn.modules.conv.Conv1d'>": "Conv1d",
    "<class 'torch.nn.modules.conv.Conv2d'>": "Conv2d",
    "<class 'torch.nn.modules.conv.Conv3d'>": "Conv3d",
    "<class 'torch.nn.modules.activation.ReLU'>": "ReLU",
    "<class 'torch.nn.modules.sparse.EmbeddingBag'>": "EmbeddingBag",
    "<method 'add' of 'torch._C._TensorBase' objects>": "add",  # for IPEX < 2.2
    "<method 'add' of 'torch._C.TensorBase' objects>": "add",  # for IPEX >= 2.2
    "<class 'torch.nn.modules.pooling.AdaptiveAvgPool2d'>": "AdaptiveAvgPool2d",
    "Linear_Relu": "Linear",
    "Linear_add": "Linear",
    "<class 'torch.nn.modules.linear.Linear'>": "Linear",
    "<class 'torch.nn.modules.pooling.MaxPool2d'>": "MaxPool2d",
    "re": {
        "<built-in method matmul of type object at": "matmul",
        "<built-in method add of type object at": "add",
        "<built-in method bmm of type object at": "bmm",
    },
}

BLOCK_PATTERNS = [
    # [['OP_TYPE1', NUM_OPS], ['OP_TYPE2', NUM_OPS], ...]
    [["Linear", 4], ["Linear", 4]],  # TODO add model name
    [["Linear", 2], ["Linear", 2]],  # TODO add model name
    [["Conv1D", 2], ["Conv1D", 2]],  # GPT-2
    [["Linear", 4], ["Linear", 3]],  # Llama
    [["Linear", 4], ["Linear", 2]],  # T5-Encoder, OPT
    [["Linear", 4], ["Linear", 1], ["Linear", 1]],  # Bert
    [["Linear", 4], ["Linear", 4], ["Linear", 2]],  # T5-Decoder
]


def cfg_to_qconfig(tune_cfg, cfgs, op_infos_from_cfgs, output_tensor_id_op_name):  # pragma: no cover
    """Updates json file in ipex_config_path.

    Args:
        tune_cfg (dict): dictionary of quantization configuration.
        cfgs (dict): configs loaded from ipex config path.
        op_infos_from_cfgs (dict): dict containing configs that have been parsed for each op.
        output_tensor_ids_op_name (dict): dict containing op names corresponding to 'op_infos_from_cfgs'.

    Returns:
        user_cfg (dict): quantization configuration for ops.
    """
    assert cfgs is not None, "No configure for IPEX int8 model..."
    op_infos = copy.deepcopy(op_infos_from_cfgs)
    cfgs, user_cfg = check_cfg_and_qconfig(tune_cfg["op"], cfgs, op_infos, output_tensor_id_op_name)
    with open(ipex_config_path, "w") as write_f:
        json.dump(cfgs, write_f, indent=4)
    return user_cfg


def check_cfg_and_qconfig(user_cfg, cfgs, op_infos_from_cfgs, output_tensor_ids_op_name):  # pragma: no cover
    """Check configs and quantization configs.

    Args:
        user_cfg (dict): quantization configuration for ops.
        cfgs (dict): configs loaded from ipex config path.
        op_infos_from_cfgs (dict): dict containing configs that have been parsed for each op.
        output_tensor_ids_op_name (dict): dict containing op names corresponding to 'op_infos_from_cfgs'.

    Returns:
        cfgs (dict): updated configs.
    """
    ori_user_cfg = copy.deepcopy(user_cfg)
    tmp_user_cfg = OrderedDict()
    for op in user_cfg:  # map ipex op_name to pt op_name
        for i, op_name in enumerate(op):
            for ops, _ in op_infos_from_cfgs.items():
                if "fqn" in op_infos_from_cfgs[ops].keys() and op_infos_from_cfgs[ops]["fqn"] == op_name:
                    if op_infos_from_cfgs[ops]["op_type"] in unify_op_type_mapping_ipex:
                        ori_op = (tuple(ops), unify_op_type_mapping_ipex[op_infos_from_cfgs[ops]["op_type"]])
                        tmp_user_cfg[((ori_op[0],), ori_op[1])] = user_cfg[op]
                        break

    for op_name in tmp_user_cfg:
        inc_op_cfg = tmp_user_cfg[op_name]
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
                            inc_scheme, inc_algorithm, smooth_quant=False, smooth_quant_enable=smooth_quant_enable
                        )
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
    return cfgs, ori_user_cfg


def generate_xpu_qconfig(tune_cfg):  # pragma: no cover
    """Generates qconfig for quantiztaion on xpu device.

    Args:
        tune_cfg (dict): dictionary of quantization configuration.

    Returns:
        qconfig (dict): quantization configuration for ops.
    """
    # qconfig observer & config constants for ipex-xpu
    from torch.ao.quantization import HistogramObserver, MinMaxObserver, QConfig

    act_observer_minmax_asym = MinMaxObserver.with_args(quant_min=0, quant_max=127)
    act_observer_minmax_sym = MinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, quant_min=-128, quant_max=127
    )
    act_observer_kl_asym = HistogramObserver.with_args(quant_min=0, quant_max=127)
    act_observer_kl_sym = HistogramObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, quant_min=-128, quant_max=127
    )
    # no tuning for granularity due to tuning space
    weight_observer_minmax_sym = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)

    qconfig = {}
    user_cfg = copy.deepcopy(tune_cfg["op"])
    for _, cfg in user_cfg.items():
        act_algo = cfg["activation"]["algorithm"]
        act_sym = cfg["activation"]["scheme"]
        break

    if act_algo == "minmax":
        if act_sym == "sym":
            activation = act_observer_minmax_sym
        else:
            activation = act_observer_minmax_asym
    else:
        if act_sym == "sym":
            activation = act_observer_kl_sym
        else:
            activation = act_observer_kl_asym

    qconfig[""] = QConfig(activation=activation, weight=weight_observer_minmax_sym)

    for (op_name, op_type), cfg in user_cfg.items():
        if cfg["weight"]["dtype"] == "fp32":
            qconfig[op_name] = None
    return qconfig


def generate_activation_observer(
    scheme, algorithm, smooth_quant=False, smooth_quant_enable=False, alpha=0.5
):  # pragma: no cover
    """This is a helper method to generate an activation observer.

    Args:
        scheme (str): Quantization scheme to be used.
        algorithm (str): What algorithm for computing the quantization parameters based on.

    Returns:
        An observer.
    """
    kl_activation_observer = {
        "name": "HistogramObserver",
        "bins": 2048,
        "dtype": "torch.quint8",
        "qscheme": "torch.per_tensor_affine",
        "reduce_range": False,
        "quant_min": 0,
        "quant_max": 255,
    }
    minmax_activation_observer = {
        "name": "MinMaxObserver",
        "dtype": "torch.quint8",
        "qscheme": "torch.per_tensor_affine",
        "reduce_range": False,
        "quant_min": 0,
        "quant_max": 255,
    }
    smoothquant_kl_activation_observer = {
        "name": "SmoothQuantActivationObserver",
        "smooth_quant_enabled": smooth_quant_enable,
        "dtype": "torch.quint8",
        "qscheme": "torch.per_tensor_affine",
        "reduce_range": False,
        "quant_min": 0,
        "quant_max": 255,
        "alpha": 0.5 if alpha == "auto" else alpha,
        "act_observer": kl_activation_observer,
        "act_ic_observer": {
            "name": "PerChannelMinMaxObserver",
            "ch_axis": -1,
            "dtype": "torch.quint8",
            "qscheme": "torch.per_channel_affine",
            "reduce_range": False,
            "quant_min": 0,
            "quant_max": 255,
        },
    }
    smoothquant_minmax_activation_observer = {
        "name": "SmoothQuantActivationObserver",
        "smooth_quant_enabled": smooth_quant_enable,
        "dtype": "torch.quint8",
        "qscheme": "torch.per_tensor_affine",
        "reduce_range": False,
        "quant_min": 0,
        "quant_max": 255,
        "alpha": 0.5 if alpha == "auto" else alpha,
        "act_observer": minmax_activation_observer,
        "act_ic_observer": {
            "name": "PerChannelMinMaxObserver",
            "ch_axis": -1,
            "dtype": "torch.quint8",
            "qscheme": "torch.per_channel_affine",
            "reduce_range": False,
            "quant_min": 0,
            "quant_max": 255,
        },
    }
    REDUCE_RANGE = False if CpuInfo().vnni else True
    if REDUCE_RANGE:
        minmax_activation_observer["reduce_range"] = REDUCE_RANGE
        kl_activation_observer["reduce_range"] = REDUCE_RANGE
    if scheme == "sym":
        minmax_activation_observer["qscheme"] = "torch.per_tensor_symmetric"
        minmax_activation_observer["dtype"] = "torch.qint8"
        minmax_activation_observer["quant_min"] = -128
        minmax_activation_observer["quant_max"] = 127
        kl_activation_observer["qscheme"] = "torch.per_tensor_symmetric"
        kl_activation_observer["dtype"] = "torch.qint8"
        kl_activation_observer["quant_min"] = -128
        kl_activation_observer["quant_max"] = 127
    if smooth_quant and smooth_quant_enable:
        if algorithm == "kl":
            return smoothquant_kl_activation_observer
        if algorithm == "minmax":
            return smoothquant_minmax_activation_observer
    else:
        if algorithm == "kl":
            return kl_activation_observer
        if algorithm == "minmax":
            return minmax_activation_observer


def get_quantizable_ops_recursively(model, example_inputs):  # pragma: no cover
    """Get all quantizable ops from model.

    Args:
        model (object): input model
        example_inputs (dict|list|tuple|torch.Tensor): used to trace torch model.

    Returns:
        quantizable_ops (list): list of tuples of op_name and op_type.
        cfgs (dict): dict of configuration.
        op_infos_from_cfgs (dict): dict containing configs that have been parsed for each op.
        output_tensor_ids_op_name (dict): dict containing op names corresponding to 'op_infos_from_cfgs'.
    """
    quantizable_ops = []
    op_name_info = []
    # group ops by position for transform-based model
    detector = TransformerBasedModelBlockPatternDetector(model)
    detect_result = detector.detect_block()
    attention_block = detect_result.get("attention_blocks", None)
    ffn_blocks = detect_result.get("ffn_blocks", None)
    logger.info(f"Attention Blocks: {len(attention_block)}")
    logger.info(f"FFN Blocks: {len(ffn_blocks)}")
    if not os.path.exists(ipex_config_path):
        assert isinstance(model, torch.nn.Module), "The model passed in is not the instance of torch.nn.Module"

    if hasattr(model, "save_qconf_summary"):  # pragma: no cover
        os.makedirs(os.path.dirname(ipex_config_path), exist_ok=True)
        model.save_qconf_summary(qconf_summary=ipex_config_path)
    else:
        model.eval()

        # create a quantization config file for intel pytorch extension model
        os.makedirs(os.path.dirname(ipex_config_path), exist_ok=True)
        assert example_inputs is not None, "IPEX need q_dataloader or example_inputs to prepare the model"
        from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig

        if ipex_ver.release >= Version("2.1").release:
            # HistogramObserver will cause a performance issue.
            # static_qconfig = ipex.quantization.default_static_qconfig_mapping
            qconfig = QConfig(
                activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
                weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
            )
            from torch.ao.quantization import QConfigMapping

            static_qconfig = QConfigMapping().set_global(qconfig)
        else:
            static_qconfig = QConfig(
                activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
                weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
            )

        if isinstance(example_inputs, dict):
            model = ipex.quantization.prepare(model, static_qconfig, example_kwarg_inputs=example_inputs, inplace=True)
        else:
            model = ipex.quantization.prepare(model, static_qconfig, example_inputs=example_inputs, inplace=True)
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
                    if "class" in ipex_op_type:  # "<class 'torch.nn.modules.activation.ReLU'>"
                        op_type = ipex_op_type.split("'")[1]
                        op_name_info.append((module_fqn, eval(op_type).__name__))
                    elif "method" in ipex_op_type:  # "<method 'add' of 'torch._C._TensorBase' objects>"
                        method = ipex_op_type.split("'")[1]
                        op_name_info.append((module_fqn, method))
                    elif "_" in ipex_op_type:  # "Convolution_Relu", "Linear_Relu"
                        op_name_info.append((module_fqn, ipex_op_type.split("_")[0]))
                else:
                    re_flag = False
                    for pattern, unify_op_type in unify_op_type_mapping_ipex["re"].items():
                        if re.match(pattern, ipex_op_type):
                            re_flag = True
                            quantizable_ops.append((tuple(name), unify_op_type))
                            map_op_name_to_fqn[(tuple(name), unify_op_type)] = module_fqn
                            op_name_info.append((module_fqn, ipex_op_type))
                            break
                    if not re_flag:
                        quantizable_ops.append((tuple(name), ipex_op_type))
                        map_op_name_to_fqn[(tuple(name), ipex_op_type)] = module_fqn
                        op_name_info.append((module_fqn, ipex_op_type))
            else:
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
                op_name_info.append((module_fqn, op_type))

    logger.debug("Map op name to fqn: ")
    logger.debug(map_op_name_to_fqn)
    logger.info("Attention Blocks : ")
    logger.info(attention_block)
    logger.info("FFN Blocks : ")
    logger.info(ffn_blocks)
    return quantizable_ops, cfgs, op_infos_from_cfgs, output_tensor_id_op_name, op_name_info


def simple_inference(q_model, example_inputs, iterations=1):
    """The function is used for ipex warm-up inference."""
    for _ in range(iterations):
        if isinstance(example_inputs, tuple) or isinstance(example_inputs, list):
            q_model(*example_inputs)
        elif isinstance(example_inputs, dict):
            q_model(**example_inputs)
        else:
            q_model(example_inputs)


def dump_model_op_stats(user_cfg):
    """This is a function to dump quantizable ops of model to user.

    Args:
        user_cfg (dict): quantization config

    Returns:
        None
    """
    res = dict()
    for k, v in user_cfg.items():
        op_type = k[1]
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


def parse_cfgs(cfgs):  # pragma: no cover
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


class TransformerBasedModelBlockPatternDetector:  # pragma: no cover
    """Detect the attention block and FFN block in transformer-based model."""

    def __init__(self, model: torch.nn.Module, pattern_lst: List[List[Union[str, int]]] = BLOCK_PATTERNS) -> None:
        """Init the block detector.

        Args:
            model: the model to be detected.
            pattern_lst: block pattern list.
        """
        self.model = model
        self.pattern_lst = pattern_lst
        self.pos_info = None

    def detect_block(self) -> Dict[str, List[List[str]]]:
        """Traverse the model definition and return the attention blocks and ffn blocks.

        Returns:
            blocks: A dict include the detected attention blocks and ffn blocks.
        """
        # Step 1: Traverse model definition and record the op position
        if not self.pos_info:
            pos_info = {0: {}}
            self.traverse_model(self.model, result=pos_info)
            self.pos_info = pos_info
        # Step 2: Traverse all blocks in different depths and record the blocks that matched the pattern
        detect_result = []
        for pattern in self.pattern_lst:
            _, result = self._search_pattern(pos_info, pattern)
            if result:
                detect_result.append((result, pattern))
        # Step 3: Get the attention blocks and ffn blocks
        blocks = {"attention_blocks": None, "ffn_blocks": None}
        blocks["attention_blocks"], blocks["ffn_blocks"] = self._group_block(detect_result)
        logger.debug(f'FFN BLOCKS: {blocks["ffn_blocks"]}')
        logger.debug(f'Attention BLOCKS: {blocks["attention_blocks"]}')
        return blocks

    @staticmethod
    def traverse_model(model, prefix="", depth=1, result=None, key=0):
        """Traverse the pytorch model according to its hierarchical structure.

        Args:
            model: input model to be traversed.
            prefix: prefix of module. Defaults to "".
            depth: current traverse depth. Defaults to 1.
            result: depth and its included ops. Defaults to {0: {}}.
            key: current root key. Defaults to 0.
        """
        module_lst = list(model.named_children())
        if len(module_lst) == 0:
            # layer name: 'encoder.layer.7.attention.self.query'
            # model repr: Linear(in_features=768, out_features=768, bias=True)
            # class name: 'Linear'
            result[key] = (prefix, model, model.__class__.__name__)
        for i, (name, sub_module) in enumerate(module_lst, 1):
            indent = "    " * depth
            new_name = prefix + "." + name if prefix != "" else name
            model_type = sub_module.__class__.__name__
            logger.debug(f"Depth: [{depth}]" + indent + f"[{model_type}]{ new_name}")
            sub_key = (depth, i, model_type)
            if sub_key not in result[key]:
                result[key][sub_key] = dict()
            TransformerBasedModelBlockPatternDetector.traverse_model(
                sub_module, prefix=new_name, depth=depth + 1, result=result[key], key=sub_key
            )

    @staticmethod
    def _search_pattern(pos_info: Dict, pattern: List[List[Union[str, int]]]) -> List[List[str]]:
        """Search all blocks that matched the pattern.

        Args:
            pos_info: the position information of ops.
            pattern: block pattern.

        Returns:
            The number of matched blocks and the matched blocks.
        """
        max_depth = get_depth(pos_info)
        matched_cnt = 0
        result = []
        for depth in range(max_depth, -1, -1):
            attention_depth = depth
            depth_block_lst = []
            get_dict_at_depth(pos_info, attention_depth, depth_block_lst, 0)
            target_op_types = set(pair[0] for pair in pattern)
            for i, block in enumerate(depth_block_lst):
                sub_block_lst = []
                get_dict_at_depth(block, 1, sub_block_lst, 0)
                block_pattern = []
                block_result = []
                for sub_block in sub_block_lst:
                    ops_lst = []
                    get_element_under_depth(sub_block, ops_lst)
                    filter_ops = [op for op in ops_lst if op[2] in target_op_types]
                    if len(filter_ops) > 0:
                        sub_block_pattern = [filter_ops[0][2], len(filter_ops)]
                        block_pattern.append(sub_block_pattern)
                        ops_name = [op[0] for op in filter_ops]
                        block_result.append(ops_name)
                if block_pattern == pattern:
                    matched_cnt += 1
                    logger.debug(f"[DEPTH] {depth} [BLOCK] {i},  Found block match pattern {pattern}!!")
                    logger.debug(f"[Block keys] {block.keys()}")
                    logger.debug(f"[Block Ops] { [pair[0] for pair in ops_lst if pair[2] in target_op_types]}")
                    result.append(block_result)
        if matched_cnt > 0:
            logger.info(f" Found {matched_cnt} blocks")
        return matched_cnt, result

    @staticmethod
    def _group_block(detect_result):
        """Collect attention and ffn blocks from detect result."""
        import itertools

        ffn_block_lst = []
        attention_block_lst = []
        for block_lst, pattern in detect_result:
            for block in block_lst:
                # Group the first block as attention blocks and
                # the remaining blocks belong to ffn block.
                if block:
                    attention_block_lst.append(block[0])
                    ffn_block = list(itertools.chain(*block[1:]))
                    if ffn_block:
                        ffn_block_lst.append(ffn_block)
        return attention_block_lst, ffn_block_lst
