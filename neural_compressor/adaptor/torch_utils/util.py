#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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
"""Util Class and Functions."""
import copy
import json
import re
from collections import UserDict
from functools import partial

import numpy as np
from packaging.version import Version

from ...utils import logger
from ...utils.utility import CpuInfo, LazyImport

tqdm = LazyImport("tqdm")
torch = LazyImport("torch")


def move_input_device(input, device="cpu"):
    """Auto mapping input to device for all kinds of format.

    Args:
        input (torch.tensor): input data
        device (str, optional): target device. Defaults to "cpu".

    Returns:
        input (torch.tensor): input data on target device
    """
    if device == "cpu":
        return input
    if isinstance(input, dict) or isinstance(input, UserDict):
        tmp_input = {}
        for k, inp in input.items():
            tmp_input[k] = move_input_device(inp, device)
        input = tmp_input
    elif isinstance(input, list) or isinstance(input, tuple):
        tmp_input = []
        for inp in input:
            tmp_input.append(move_input_device(inp, device))
        input = tmp_input
    elif isinstance(input, torch.Tensor):
        input = input.to(device)  # pylint: disable=no-member
    return input


def forward_wrapper(model, input):
    """Model forward with device auto mapping.

    Args:
        model (torch.nn.Module): input model
        input (torch.tensor): input data

    Returns:
        output: output data
    """
    try:
        device = next(model.parameters()).device
    except:
        # for RecursiveScriptModule
        device = "cpu"
    input = move_input_device(input, device)
    if isinstance(input, dict) or isinstance(input, UserDict):
        output = model(**input)
    elif isinstance(input, list) or isinstance(input, tuple):
        try:
            output = model(*input)
        except:
            output = model(input)
    else:
        output = model(input)
    return output


def get_embedding_contiguous(model):
    """This is a helper function for nn.Embedding, and it will get input contiguous.

    Args:
        model (object): the input model

    Returns:
        None
    """

    def contiguous_hook(module, input):
        embeddings = input[0].contiguous()
        modified_input = (embeddings, *input[1:])
        return modified_input

    for child in model.modules():
        child_type = child.__class__.__name__
        if child_type == "Embedding":
            child.register_forward_pre_hook(contiguous_hook)


def is_fused_module(module):
    """This is a helper function for `_propagate_qconfig_helper` to detect if this module is fused.

    Args:
        module (object): the input module

    Returns:
        (bool): is fused or not
    """
    op_type = str(type(module))
    if "fused" in op_type:
        return True
    else:
        return False


def collate_torch_preds(results):
    """Fetch collated results.

    Args:
        result (list): input result

    Returns:
        collate_results (list): collated results
    """
    batch = results[0]
    if isinstance(batch, list):
        results = zip(*results)
        collate_results = []
        for output in results:
            output = [batch.numpy() if isinstance(batch, torch.Tensor) else batch for batch in output]
            collate_results.append(np.concatenate(output))
    elif isinstance(batch, torch.Tensor):
        results = [batch.numpy() if isinstance(batch, torch.Tensor) else batch for batch in results]
        collate_results = np.concatenate(results)
    return collate_results


def input2tuple(input):
    """This is a helper function to converting a inputting dict values or a list to a tuple.

    Args:
        input (list or dict).

    Returns:
        A tuple.
    """
    if isinstance(input, dict) or isinstance(input, UserDict):
        output = tuple(input.values())
    elif isinstance(input, list) or isinstance(input, tuple):
        output = tuple(input)
    else:
        output = input
    return output


def append_attr(fx_model, model, fx_white_list=[]):
    """This is a helper method to append attributes for the symbolic traced model.

    Args:
        fx_model (torch.fx.GraphModule): The symbolic traced model.
        model (torch.nn.Module): The original model.

    Returns:
        fx_model (dir): The symbolic traced model with additional attributes.
    """
    fx_attr = dir(fx_model)
    org_attr = dir(model)
    ignore_match_patterns = [r"_", r"quant", r"dequant", r"weight", r"bias", r"activation_post_process"]
    ignore_search_patterns = [r"_scale_", r"_zero_point_", r"_activation_post_process_"]
    add_special_patterns = [r"_forward_hooks", r"_forward_pre_hooks", r"_backward_hooks"]
    attr_names = []
    if hasattr(fx_model, "module") and hasattr(fx_model.module, "weight"):
        if not isinstance(fx_model.module.weight, torch.Tensor):
            fx_model.weight = fx_model.module.weight()
        else:
            fx_model.weight = fx_model.module.weight
        if hasattr(fx_model.module, "bias"):
            if not isinstance(fx_model.module.bias, torch.Tensor) and fx_model.module.bias is not None:
                fx_model.bias = fx_model.module.bias()
            else:
                fx_model.bias = fx_model.module.bias
    for i in org_attr:
        if (
            type(model) in fx_white_list
            and type(model) != torch.nn.Sequential
            and any([re.search(p, i) for p in add_special_patterns])
        ):
            continue
        if any([re.search(p, i) for p in add_special_patterns]) or (
            i not in fx_attr
            and not any([re.match(p, i) for p in ignore_match_patterns])
            and not any([re.search(p, i) for p in ignore_search_patterns])
        ):
            attr_names.append(i)
    for name in attr_names:
        attr = getattr(model, name, None)

        if isinstance(attr, torch.nn.Module) or isinstance(attr, torch.quantization.qconfig.QConfig):
            continue
        setattr(fx_model, name, attr)
    return fx_model


def generate_activation_observer(scheme, algorithm, smooth_quant=False, smooth_quant_enable=False):  # pragma: no cover
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
        "upsample_rate": 128,
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
        "alpha": 0.5,
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
        "alpha": 0.5,
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


def check_cfg_and_qconfig(
    tune_cfg, cfgs, op_infos_from_cfgs, output_tensor_ids_op_name, smooth_quant=False
):  # pragma: no cover
    """Check configs and quantization configs.

    Args:
        tune_cfg (dict): dictionary of quantization configuration.
        cfgs (dict): the input configs.
        op_infos_from_cfgs (dict): op infos from configs.
        output_tensor_ids_op_name (dict): dictionary of output tensor op names.

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
                            inc_scheme, inc_algorithm, smooth_quant, smooth_quant_enable
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


def update_sq_scale(ipex_config_path, smoothquant_scale_info):
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


def auto_copy(module):  # pragma: no cover
    """Get an IPEX prepared model and return a fp32 model.

    Args:
        module (object): IPEX prepared model.

    Returns:
        fp32 model.
    """
    from intel_extension_for_pytorch.quantization._quantization_state import AutoQuantizationStateModuleDict

    def _nn_sequential_patched_forward(cls, x):
        for module in cls:
            if not isinstance(module, AutoQuantizationStateModuleDict):
                x = module(x)
        return x

    new_module = copy.deepcopy(module)
    if hasattr(new_module, "_qconf_summary"):
        del new_module._qconf_summary
    if hasattr(new_module, "_fqn_to_auto_quant_state_map"):
        del new_module._fqn_to_auto_quant_state_map
    if hasattr(new_module, "q_config"):
        del new_module.q_config

    def convert_to_dispatch_proxy(x):
        if isinstance(x, torch.Tensor):
            return x.as_subclass(CopyTensorProxy)  # type: ignore[arg-type]
        else:
            return x

    global_disable_torch_function_override = False

    class CopyTensorProxy(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            nonlocal global_disable_torch_function_override
            if (
                # global override means disable the override here
                global_disable_torch_function_override
                or
                # to prevent printing things from going into an infinite loop
                func == torch.Tensor.__repr__
                or
                # we don't need to override getters in this framework
                func.__name__ == "__get__"
            ):
                return super().__torch_function__(func, types, args, kwargs)
            kwargs = kwargs if kwargs else {}
            output = super().__torch_function__(func, types, args, kwargs)
            if output is NotImplemented:
                with torch._C.DisableTorchFunction():
                    output = func(*args, **kwargs).as_subclass(
                        CopyConvertTensorProxy  # pylint: disable=E0602 # noqa: F821
                    )
                assert output is not NotImplemented
            return output

        def __repr__(self):
            return f"CopyTensorProxy({super().__repr__()})"

    cur_module = None
    module_stack: List[torch.nn.Module] = []  # pylint: disable=E0602 # noqa: F821
    assert len(module.__class__.__bases__) == 1

    class CopyDispatchModule(module.__class__.__bases__[0]):
        def __call__(self, *args, **kwargs):
            new_args = torch.fx.node.map_aggregate(args, convert_to_dispatch_proxy)
            new_kwargs = torch.fx.node.map_aggregate(kwargs, convert_to_dispatch_proxy)
            orig_module_call = torch.nn.Module.__call__
            orig_nn_sequential_forward = torch.nn.Sequential.forward

            def _patched_module_call(self, *args, **kwargs):
                nonlocal cur_module
                old_module = cur_module
                cur_module = self
                nonlocal global_disable_torch_function_override
                try:
                    parent_module = module_stack[-1] if len(module_stack) else None
                    module_stack.append(self)
                    output = orig_module_call(self, *args, **kwargs)
                    return output
                finally:
                    module_stack.pop()
                    cur_module = old_module

            torch.nn.Module.__call__ = _patched_module_call
            torch.nn.Sequential.forward = _nn_sequential_patched_forward  # type: ignore[assignment]
            try:
                output = super().__call__(*new_args, **new_kwargs)

                def unwrap_proxy(a):
                    if isinstance(a, CopyTensorProxy):
                        a.__class__ = torch.Tensor  # type: ignore[assignment]
                    return a

                output = torch.fx.node.map_aggregate(output, unwrap_proxy)
                return output
            finally:
                torch.nn.Module.__call__ = orig_module_call
                torch.nn.Sequential.forward = orig_nn_sequential_forward  # type: ignore[assignment]

    new_module.__class__ = CopyDispatchModule
    return new_module


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
            module = module
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
    module = model
    name_list = op_name.split(".")
    for name in name_list[:-1]:
        if hasattr(module, name):
            module = getattr(module, name)
        else:
            module = module
    setattr(module, name_list[-1], new_module)


def simple_inference(model, input):
    """Record model output tensor.

    Args:
        model (object): the input model.
        input (object).

    Returns:
        output (object).
    """
    with torch.no_grad():
        if isinstance(input, (dict, UserDict)):
            output = model(**input)
        elif isinstance(input, (list, tuple)):
            try:
                output = model(*input)
            except:
                output = model(input)
        else:
            output = model(input)
    return output


def get_example_input(dataloader, i=1):
    """Get the example input.

    Args:
        dataloader (object): calibration dataset.

    Returns:
        example_inp (object).
    """
    iter = 0
    try:
        for example_inp, label in dataloader:
            if iter == i:
                break
            else:
                iter += 1
    except:
        for example_inp in dataloader:
            if iter == i:
                break
            else:
                iter += 1
    return example_inp


def get_fallback_order(
    adaptor, fp32_model, dataloader, tune_cfg, confidence_batches, fallback=False, requantize_cfgs=None
):
    """Get the fall back order for strategy.

    Args:
        fp32_model (object): the input model.
        dataloader(torch.utils.data.DataLoader): The calibration dataloader.
        tune_cfg (dict): dictionary of quantization configuration.
        confidence_batches (int): number of confidence batches.
        fallback (bool): if the order is fallback.

    Returns:
        ordered_ops (dict/list): The fallback order for strategy.
    """
    fp32_model.eval()
    order_dict = {}
    for i in range(0, confidence_batches):
        example_input = get_example_input(dataloader, i)
        if fallback:
            ordered_ops = get_mse_order_per_fp32(adaptor, fp32_model, example_input, tune_cfg)
            for i, name in enumerate(ordered_ops):
                order_dict[name] = order_dict.get(name, 0) + len(order_dict) - i
            ordered_ops = sorted(order_dict, key=lambda k: order_dict[k], reverse=True)
        else:
            ordered_ops = get_mse_order_per_int8(adaptor, fp32_model, example_input, tune_cfg)
            for i, name in enumerate(ordered_ops):
                order_dict[name] = order_dict.get(name, 0) + len(order_dict) - i
    return ordered_ops


op_cfg_mapping = {}


def get_mse_order_per_fp32(adaptor, model, example_inp, tune_cfg):
    """This is a helper method to check the mse influence to last module after QDQ(quant/dequant).

    Args:
        model (torch.fx.GraphModule/torch.nn.Module): A torch model.
        example_inp (object): example inputs.
        tune_cfg (dict): dictionary of quantization configuration.

    Returns:
        fallback_order (dict/list): The fallback order for strategy.
    """
    inner_output = None

    def output_hook(self, input, output):
        nonlocal inner_output
        inner_output = output
        return output

    op_type_dict = {}
    for k, v in tune_cfg["op"].keys():
        op_type_dict[k] = v

    from ..pytorch import PyTorch_FXAdaptor, _cfg_to_qconfig, _cfgs_to_fx_cfgs

    op_cfgs = _cfg_to_qconfig(tune_cfg, tune_cfg["approach"])
    # insert hook to get output tesnor from last module
    last_module_name = list(op_cfgs.keys())[-1]
    module = fetch_module(model, last_module_name)  # get last module
    module.register_forward_hook(output_hook)
    # record fp32 model output tensor at first
    output_fp32 = simple_inference(model, example_inp)
    inner_output_fp32 = inner_output

    fx_op_cfgs = {}
    fallback_order = {}
    logger.info("Evaluate the sensitivity for each int8 operation")
    for op_name, qconfig in tqdm(op_cfgs.items()):
        if op_name == "bf16_ops_list":
            continue
        global op_cfg_mapping
        if op_name not in op_cfg_mapping:
            op_cfg_mapping[op_name] = qconfig
        tmp_model = copy.deepcopy(model)
        if not qconfig:
            logger.debug(f"No qconfig for {op_name}, next op.")
            continue
        op_cfgs[op_name] = None
        fx_op_cfgs = _cfgs_to_fx_cfgs(op_cfgs, tune_cfg["approach"])
        op_cfgs[op_name] = qconfig
        from torch.quantization.quantize_fx import convert_fx, prepare_fx

        # do quantization
        if adaptor.sub_module_list is None:
            if adaptor.version.release >= Version("1.13.0").release:  # pragma: no cover
                tmp_model = prepare_fx(tmp_model, fx_op_cfgs, example_inp)
            else:
                tmp_model = prepare_fx(  # pylint: disable=E1120
                    tmp_model,
                    fx_op_cfgs,
                )
        else:
            PyTorch_FXAdaptor.prepare_sub_graph(adaptor.sub_module_list, fx_op_cfgs, tmp_model, prefix="")
        simple_inference(tmp_model, example_inp)
        if adaptor.sub_module_list is None:
            tmp_model = convert_fx(tmp_model)
        else:
            PyTorch_FXAdaptor.convert_sub_graph(adaptor.sub_module_list, tmp_model, prefix="")

        # insert hook to get output tesnor from last module
        module = fetch_module(tmp_model, list(op_cfgs.keys())[-1])  # get last module
        module.register_forward_hook(output_hook)
        output_qdq = simple_inference(tmp_model, example_inp)
        inner_output_int8 = inner_output.dequantize() if inner_output.dtype == torch.quint8 else inner_output
        mse_val = (inner_output_fp32 - inner_output_int8).pow(2).sum()
        fallback_order[(op_name, op_type_dict[op_name])] = mse_val

    logger.debug(f"fallback order: {fallback_order}")
    ordered_ops = sorted(fallback_order.keys(), key=lambda key: fallback_order[key], reverse=False)
    if not ordered_ops:
        return ordered_ops
    min_mse, max_mse = fallback_order[ordered_ops[0]], fallback_order[ordered_ops[-1]]

    if min_mse < 0.8 * max_mse:
        logger.debug("Return the sorted ops early.")
        return ordered_ops

    double_check_list = []
    for op_name in ordered_ops:
        if min_mse <= fallback_order[op_name] <= (max_mse - min_mse) * 0.1 + min_mse:
            double_check_list.append(op_name)

    check_num = min(len(ordered_ops) // 10 + 1, 5)
    double_check_list = ordered_ops[:check_num]
    logger.debug(f"double check list: {double_check_list}")
    worst_op_name = ordered_ops[-1]
    op_cfgs[worst_op_name[0]] = None  # fallback worst module first
    new_fallback_order = {}

    logger.info("Evaluate the sensitivity gradient for selected operations")
    for op_name, op_type in tqdm(double_check_list):
        tmp_model = copy.deepcopy(model)
        qconfig = op_cfgs[op_name]
        op_cfgs[op_name] = None
        fx_op_cfgs = _cfgs_to_fx_cfgs(op_cfgs, tune_cfg["approach"])
        op_cfgs[op_name] = qconfig
        from torch.quantization.quantize_fx import convert_fx, prepare_fx

        # do quantization
        if adaptor.sub_module_list is None:
            if adaptor.version.release >= Version("1.13.0").release:  # pragma: no cover
                tmp_model = prepare_fx(tmp_model, fx_op_cfgs, example_inp)
            else:
                tmp_model = prepare_fx(  # pylint: disable=E1120
                    tmp_model,
                    fx_op_cfgs,
                )
        else:
            PyTorch_FXAdaptor.prepare_sub_graph(adaptor.sub_module_list, fx_op_cfgs, tmp_model, prefix="")
        simple_inference(tmp_model, example_inp)
        if adaptor.sub_module_list is None:
            tmp_model = convert_fx(tmp_model)
        else:
            PyTorch_FXAdaptor.convert_sub_graph(adaptor.sub_module_list, tmp_model, prefix="")

        # insert hook to get output tesnor from last module
        module = fetch_module(tmp_model, last_module_name)  # get last module
        module.register_forward_hook(output_hook)
        output_qdq = simple_inference(tmp_model, example_inp)
        inner_output_int8 = inner_output.dequantize() if inner_output.dtype == torch.quint8 else inner_output
        mse_val = (inner_output_fp32 - inner_output_int8).pow(2).sum()
        new_fallback_order[(op_name, op_type_dict[op_name])] = mse_val

    ordered_ops = sorted(new_fallback_order.keys(), key=lambda key: new_fallback_order[key], reverse=False)

    return ordered_ops


def get_mse_order_per_int8(adaptor, fp32_model, example_input, tune_cfg):
    """This is a helper method to check the mse influence to last module after QDQ(quant/dequant).

    Args:
        model (torch.fx.GraphModule/torch.nn.Module): A torch model.
        example_inp (object): example inputs.
        tune_cfg (dict): dictionary of quantization configuration.

    Returns:
        fallback_order (dict/list): The fallback order for strategy.
    """
    inner_output = None

    def output_hook(self, input, output):
        nonlocal inner_output
        inner_output = output
        return output

    op_type_dict = {}
    for k, v in tune_cfg["op"].keys():
        op_type_dict[k] = v

    example_inp = example_input

    from ..pytorch import _cfg_to_qconfig

    op_cfgs = _cfg_to_qconfig(tune_cfg, tune_cfg["approach"])
    module = fetch_module(fp32_model, list(op_cfgs.keys())[-1])  # get last module
    # insert hook to get output tesnor from last module
    module.register_forward_hook(output_hook)
    # record fp32 model output tensor at first
    output_fp32 = simple_inference(fp32_model, example_inp)
    inner_output_fp32 = inner_output

    quant_list = []
    for k, v in tune_cfg["op"].items():
        if k[1] in ["LayerNorm", "Dropout", "InstanceNorm3d"]:
            continue
        if v["weight"]["dtype"] == "fp32":
            quant_list.append(k)
    fallback_order = {}
    logger.info("Evaluate the sensitivity for each fp32 operation")
    for op_name, op_type in tqdm(quant_list):
        if op_name in op_cfg_mapping:
            tmp_model = copy.deepcopy(fp32_model)
            from ..pytorch import PyTorch_FXAdaptor, _cfg_to_qconfig, _cfgs_to_fx_cfgs

            op_cfgs[op_name] = op_cfg_mapping[op_name]
            fx_op_cfgs = _cfgs_to_fx_cfgs(op_cfgs, tune_cfg["approach"])
            from torch.quantization.quantize_fx import convert_fx, prepare_fx

            # do quantization
            if adaptor.sub_module_list is None:
                if adaptor.version.release >= Version("1.13.0").release:  # pragma: no cover
                    tmp_model = prepare_fx(tmp_model, fx_op_cfgs, example_inp)
                else:
                    tmp_model = prepare_fx(  # pylint: disable=E1120
                        tmp_model,
                        fx_op_cfgs,
                    )
            else:
                PyTorch_FXAdaptor.prepare_sub_graph(adaptor.sub_module_list, fx_op_cfgs, tmp_model, prefix="")
            simple_inference(tmp_model, example_inp)
            if adaptor.sub_module_list is None:
                tmp_model = convert_fx(tmp_model)
            else:
                PyTorch_FXAdaptor.convert_sub_graph(adaptor.sub_module_list, tmp_model, prefix="")

            # record int8 model output tensor
            module = fetch_module(tmp_model, list(op_cfgs.keys())[-1])  # get last module
            module.register_forward_hook(output_hook)
            output_qdq = simple_inference(tmp_model, example_inp)
            inner_output_int8 = inner_output
            if inner_output_fp32.dtype == torch.quint8:
                inner_output_fp32 = inner_output_fp32.dequantize()
            if inner_output_int8.dtype == torch.quint8:
                inner_output_int8 = inner_output_int8.dequantize()

            mse_val = (inner_output_fp32 - inner_output_int8).pow(2).sum()
            fallback_order[(op_name, op_type_dict[op_name])] = mse_val
            # re-insert fp32 module into model
    ordered_ops = sorted(fallback_order.keys(), key=lambda key: fallback_order[key], reverse=False)
    return ordered_ops


def get_torch_version():
    """Get torch version."""
    from packaging.version import Version

    try:
        torch_version = torch.__version__.split("+")[0]
    except ValueError as e:  # pragma: no cover
        assert False, "Got an unknown version of torch: {}".format(e)
    version = Version(torch_version)
    return version


def match_datatype_pattern(datatype, pattern=None):
    """Check the datatype pattern."""
    import re

    if not pattern:
        pattern = r"(uint|int)([1-8])"
    match = re.match(pattern, datatype)
    return match


def _get_signed_and_bits(datatype):
    """Parse sign and bits from datatype."""
    unsigned = datatype[0] == "u"
    if unsigned:
        num_bits = int(datatype[4:])
    else:
        num_bits = int(datatype[3:])
    return unsigned, num_bits


def calculate_quant_min_max(unsigned, num_bits):
    """Calculate the qmin and qmax according to the datatype."""
    # TODO handle reduce range
    quant_min, quant_max = None, None
    if unsigned:
        quant_min, quant_max = 0.0, 2.0 ** (num_bits) - 1.0
    else:
        quant_min, quant_max = -1 * 2.0 ** (num_bits - 1), 2.0 ** (num_bits - 1) - 1
    return quant_min, quant_max


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


def get_op_type_by_name(op_name, quantizable_ops):
    """Get op type by op name."""
    for pair in quantizable_ops:
        if pair[0] == op_name:
            return pair[1]
    return None


def collect_weight_info(model, q_config):
    """Collect weight info from q_config for dumping into qconfig.json.

    qconfig.json example:
    ```
    {
        'fc': {
            'bits': 4,
            'group_size': 128,
            'scheme': 'asym',
            'algorithm': 'RTN'
        }
        ...
    }
    ```

    Args:
        q_config (_type_): quantization configure
    """
    weight_info = {}
    from neural_compressor.utils.logger import DEBUG, level

    for op, config in q_config["op"].items():
        op_name, op_type = op
        if config["weight"]["dtype"] == "fp32":
            weight_info[op_name] = {"dtype": "fp32"}
        else:
            # fetch module type for MulLinear
            module = fetch_module(model, op_name)
            if level == DEBUG:
                weight_info[op_name] = {
                    "dtype": config["weight"]["dtype"],
                    "bits": config["weight"]["bits"],
                    "group_size": config["weight"]["group_size"],
                    "scheme": config["weight"]["scheme"],
                    "module_type": str(type(module)).split("'")[1],
                    "algorithm": config["weight"]["algorithm"],
                }
            else:
                weight_info[op_name] = {
                    "dtype": config["weight"]["dtype"],
                    "bits": config["weight"]["bits"],
                    "group_size": config["weight"]["group_size"],
                    "scheme": config["weight"]["scheme"],
                    "module_type": str(type(module)).split("'")[1],
                }
    return weight_info


def get_module_input_output(
    model, module_hook_config={}, dataloader=None, iters=-1, calib_func=None, input_func=None, output_func=None
):
    """A help function to get input and output tensor of modules in module_name_list.

    Args:
        model: torch model.
        module_hook_config (dict, optional): required module name for input/output. Defaults to {}.
            For example:
                module_hook_config = {
                    'fc1': ['output'],
                    'fc2': ['input', 'output']
                }
        dataloader: dataloader for model input.
        iters: iterations for inference.
        calib_func: a custom inference function to replace dataloader and iters.
        input_func: preprocess input for less memory usage
        output_func: preprocess output for less memory usage

    Returns:
        total_values: recorded input_values, output_values.
            for example:
                {'fc1':
                    {'input': [], 'output': []},
                }
    """
    from collections import defaultdict

    total_values = defaultdict(defaultdict)

    def _save_input_output_hook(name, record_input=False, record_output=False):
        """
        A forward hook to save input and output values of a module
            param name: the module name
            return: A hook function
        """

        def _hook(module, inputs, outputs):
            if record_input:
                input = inputs[0]
                if input_func is not None:
                    input = input_func(input)
                if name in total_values and "input" in total_values[name]:
                    total_values[name]["input"].append(input)
                else:
                    total_values[name]["input"] = [input]
            if record_output:
                output = outputs[0] if isinstance(outputs, tuple) else outputs
                if output_func is not None:
                    output = output_func(output)
                if input_func is not None:
                    input = input_func(input)
                if name in total_values and "output" in total_values[name]:
                    total_values[name]["output"].append(output)
                else:
                    total_values[name]["output"] = [output]

        return _hook

    hook_list = []
    for name, module in model.named_modules():
        if name in module_hook_config:
            require_list = module_hook_config[name]
            logger.debug(f"required hooks {name}: {require_list}")
            _hook = _save_input_output_hook(
                name,
                record_input="input" in require_list,
                record_output="output" in require_list,
            )
            require_list = module_hook_config[name]
            hook_list.append(module.register_forward_hook(_hook))
    if calib_func:
        calib_func(model)
    else:
        from neural_compressor.adaptor.torch_utils.waq import model_forward

        model_forward(model, dataloader, iters, device=next(model.parameters()).device)
    for h in hook_list:
        h.remove()
    return total_values


def get_absorb_layers(model, example_inputs, supported_layers=["Linear"], folding=False):
    """Get absorb_to_layer and no_absorb_layer.

    Args:
        model (torch.nn.Module): input model
        example_inputs: example_inputs
        supported_layers (list, optional): supported_layers. Defaults to ['Linear'].
        folding (bool, optional): whether allow self-absorption. Defaults to False.

    Returns:
        absorb_to_layer: dict of absorb_to_layer. eg. {absorb, [absorbed_1, xx]}
        no_absorb_layers: list of no_absorb_layers
    """
    # get modules that can be absorbed.
    from neural_compressor.adaptor.torch_utils.waq import GraphTrace

    tg = GraphTrace()
    absorb_to_layer, no_absorb_layers = tg.get_absorb_to_layer(model, example_inputs, supported_layers)
    if absorb_to_layer is None or absorb_to_layer == {}:
        absorb_to_layer = {}
        logger.warning("No absorb layer is detected.")
        # if no_absorb_layers is None, jit trace failed.
        # collect all linears for next step
        if no_absorb_layers is None:
            no_absorb_layers = []
            op_types = ["Linear"]
            for name, module in model.named_modules():
                for op_type in op_types:
                    if op_type == str(module.__class__.__name__):
                        no_absorb_layers.append(name)
    return absorb_to_layer, no_absorb_layers


def get_block_prefix(model):
    """Get prefix and number of blocks.

    Args:
        model (torch.nn.Module): input model

    Returns:
        block_prefix(str): block_list name in model
        block_num(int): number of block in block_list
    """
    module_types = [torch.nn.ModuleList]
    for n, m in model.named_modules():
        if type(m) in module_types:
            block_prefix = n
            block_num = len(m)
            logger.debug(f"block_prefix: {block_prefix}, block_num: {block_num} ")
            break
    assert block_num > 0, "block num shouldn't be zero!"
    return block_prefix, block_num


def calibration(model, dataloader=None, n_samples=128, calib_func=None):
    """Calibration with dataloader or calib_func.

    Args:
        model (torch.nn.Module): input model
        dataloader: dataloader. Defaults to None.
        n_samples (int, optional): n_samples. Defaults to 128.
        calib_func: calib_func. Defaults to None.
    """
    # calibration with dataloader or calib_func
    if calib_func is not None:
        calib_func(model)
    else:
        import math

        from neural_compressor.adaptor.torch_utils.waq import model_forward

        batch_size = dataloader.batch_size
        iters = int(math.ceil(n_samples / batch_size))
        if n_samples % batch_size != 0:
            logger.info(
                "calibration samples increase from {} to {} due to batch_size is {}".format(
                    n_samples,
                    iters * batch_size,
                    batch_size,
                )
            )
        model_forward(model, dataloader, iters, next(model.parameters()).device)


def get_hidden_states(model, dataloader=None, n_samples=128, calib_func=None):
    """Get the input args and kwargs of first block.

    Args:
        model (torch.nn.Module): input model
        dataloader (dataloader, optional): input dataloader. Defaults to None.
        n_samples (int, optional): number samples from dataloader. Defaults to 128.
        calib_func (func, optional): a calib func to replace dataloader. Defaults to None.

    Raises:
        ValueError: to avoid inference of rest parts in model

    Returns:
        total_block_args(list): a list of input args of each batch
        total_block_kwargs(list):  a list of input kwargs of each batch
    """
    # Step 1: replace block_forward to collect block inputs and avoid entire inference
    total_block_args = []
    total_block_kwargs = []

    def forward(layer, *args, **kwargs):
        # update total_hidden_states, total_block_kwargs, per batch
        total_block_args.append(list(args))
        total_block_kwargs.append(kwargs)
        raise ValueError

    block_prefix, block_num = get_block_prefix(model)
    block_list = fetch_module(model, block_prefix)
    first_block = block_list[0]
    block_forward_cache = first_block.forward
    first_block.forward = partial(forward, first_block)

    # Step 2: replace model_forward to avoid ValueError
    model_forward_cache = model.forward

    def model_forward(model, *args, **kwargs):
        nonlocal model_forward_cache
        try:
            model_forward_cache(*args, **kwargs)
        except ValueError:
            pass

    model.forward = partial(model_forward, model)

    # Step 3: execute calibration
    calibration(model, dataloader=dataloader, n_samples=n_samples, calib_func=calib_func)
    logger.info("The hidden_states collection is done.")

    # Step 4: recover model and block forward
    model.forward = model_forward_cache
    first_block.forward = block_forward_cache
    return total_block_args, total_block_kwargs
