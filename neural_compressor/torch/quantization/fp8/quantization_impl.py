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
# pylint:disable=import-error

import copy
import os

import habana_frameworks.torch.core as htcore
import torch
from deepspeed.module_inject import LinearAllreduce, LinearLayer
from deepspeed.module_inject.layers import LmHeadLinearAllreduce

from neural_compressor.common.utility import FP8_QUANT
from neural_compressor.torch.utils import fetch_module, logger, register_algo, set_module

from ..layers import Autocast, BatchMatmul, Matmul
from .modules import (
    FP8BatchMatmul,
    FP8Cast,
    FP8DynamicBatchMatmul,
    FP8DynamicLinear,
    FP8DynamicMatmul,
    FP8Linear,
    FP8LinearAllreduce,
    FP8LinearLayer,
    FP8LmHeadLinearAllreduce,
    FP8Matmul,
)

quantization_mapping = {
    LinearAllreduce: FP8LinearAllreduce,
    LinearLayer: FP8LinearLayer,
    LmHeadLinearAllreduce: FP8LmHeadLinearAllreduce,
    torch.nn.Linear: FP8Linear,
    BatchMatmul: FP8BatchMatmul,
    Matmul: FP8Matmul,
    Autocast: FP8Cast,
    # torch.matmul: fp8_matmul
}
white_list = tuple(quantization_mapping.keys())


# without scale factor 0.9, the output will be abnormal.
E4M3_AMAX = torch.tensor(240 * 0.9, dtype=torch.float).to("hpu")
E5M2_AMAX = torch.tensor(57344 * 0.9, dtype=torch.float).to("hpu")
FP8_DTYPE = [torch.float8_e5m2, torch.float8_e4m3fn]


def _replace_module(module, qconfig):
    if qconfig.approach == "static":
        if isinstance(module, white_list):
            QModule = quantization_mapping[type(module)]
            assert qconfig.weight_dtype == qconfig.act_dtype, "weight and activation should be the same dtype."
            module = QModule(module, qconfig.act_dtype)
    elif qconfig.approach == "dynamic":
        dtype = qconfig.act_dtype
        if isinstance(module, torch.nn.Linear):
            # need module for initialization
            module = FP8DynamicLinear(module, dtype)
        elif isinstance(module, Matmul):
            module = FP8DynamicMatmul(dtype)
        elif isinstance(module, BatchMatmul):
            module = FP8DynamicBatchMatmul(dtype)
        elif isinstance(module, Autocast):
            module = FP8Cast(dtype=dtype)
    htcore.mark_step()
    return module


def quantize_dynamic(model, dtype=torch.float8_e4m3fn, inplace=True):
    q_model = model if inplace else copy.deepcopy(model)
    for n, m in q_model.named_modules():
        if isinstance(m, torch.nn.Linear):
            new_m = FP8DynamicLinear(m, dtype)  # need m for init
            set_module(q_model, n, new_m)
        elif isinstance(m, Matmul):
            new_m = FP8DynamicMatmul(dtype)
            set_module(q_model, n, new_m)
        elif isinstance(m, BatchMatmul):
            new_m = FP8DynamicBatchMatmul(dtype)
            set_module(q_model, n, new_m)
        elif isinstance(m, Autocast):
            new_m = FP8Cast(dtype=dtype)
            set_module(q_model, n, new_m)
        htcore.mark_step()
    return q_model


def _add_observer(module, qconfig):
    algorithm = qconfig.act_algo

    def input_observer_forward_pre_hook(self, input):
        try:
            if isinstance(input[0], torch.Tensor):
                self.input_activation_post_process(input[0])
            if hasattr(self, "input_activation_post_process1") and isinstance(input[1], torch.Tensor):
                self.input_activation_post_process1(input[1])
            return input
        except Exception as e:
            # The KL algorithm may encounter a overflow error on EltwiseAdd.
            pass

    ### Insert input observer into model, only for fp8_e4m3 static quantization ###
    from .observer import FP8HistogramObserver, MinMaxObserver

    if isinstance(module, white_list):
        module.add_module(
            "input_activation_post_process", FP8HistogramObserver() if algorithm == "kl" else MinMaxObserver()
        )
    if isinstance(module, (BatchMatmul, Matmul)):
        module.add_module(
            "input_activation_post_process1", FP8HistogramObserver() if algorithm == "kl" else MinMaxObserver()
        )
    module.register_forward_pre_hook(input_observer_forward_pre_hook)


def _remove_observer(module, qconfig):
    import deepspeed.comm as dist
    from torch.distributed import ReduceOp

    HF_max = E4M3_AMAX if qconfig.act_dtype == torch.float8_e4m3fn else E5M2_AMAX
    if hasattr(module, "input_activation_post_process"):
        if hasattr(module.input_activation_post_process, "_non_linear_param_search"):  # kl
            min_val, max_val = module.input_activation_post_process._non_linear_param_search()
        else:
            min_val = module.input_activation_post_process.min_val
            max_val = module.input_activation_post_process.max_val
        amax = torch.max(torch.abs(max_val), torch.abs(min_val))
        if dist.is_initialized():
            amax = amax.to("hpu")
            dist.all_reduce(amax, op=ReduceOp.MAX)
        scale = HF_max / amax
        module.register_parameter("scale", torch.nn.Parameter(scale))
        delattr(module, "input_activation_post_process")
    if hasattr(module, "input_activation_post_process1"):
        if hasattr(module.input_activation_post_process1, "_non_linear_param_search"):
            min_val, max_val = module.input_activation_post_process1._non_linear_param_search()
        else:
            min_val = module.input_activation_post_process1.min_val
            max_val = module.input_activation_post_process1.max_val
        amax = torch.max(torch.abs(max_val), torch.abs(min_val))
        if dist.is_initialized():
            amax = amax.to("hpu")
            dist.all_reduce(amax, op=ReduceOp.MAX)
        scale = HF_max / amax
        module.register_parameter("scale1", torch.nn.Parameter(scale))
        delattr(module, "input_activation_post_process1")

    # remove observer hooks
    hook_map = module._forward_pre_hooks
    handle_ids_to_remove = set()
    for handle_id, hook_fn in hook_map.items():
        if hasattr(hook_fn, "__name__") and hook_fn.__name__ == "input_observer_forward_pre_hook":
            handle_ids_to_remove.add(handle_id)
    for handle_id in handle_ids_to_remove:
        hook_map.pop(handle_id)


def prepare(model, qconfig_mapping):
    for (op_name, op_type), qconfig in qconfig_mapping.items():
        if qconfig.approach == "dynamic":
            continue
        if qconfig.weight_dtype not in FP8_DTYPE:
            continue
        module = fetch_module(model, op_name)
        if module is None:
            logger.info(f"{op_name} is not found in model.")
            continue
        _add_observer(module, qconfig)
        set_module(model, op_name, module)
    return model


def convert(model, qconfig_mapping):
    for (op_name, op_type), qconfig in qconfig_mapping.items():
        if qconfig.weight_dtype not in FP8_DTYPE:
            continue
        module = fetch_module(model, op_name)
        if module is None:
            logger.info(f"{op_name} is not found in model.")
            continue
        if qconfig.approach != "dynamic":
            _remove_observer(module, qconfig)
        module = _replace_module(module, qconfig)
        set_module(model, op_name, module)
        htcore.mark_step()
    return model


@register_algo(name=FP8_QUANT)
def quantize(model, qconfig_mapping, run_fn=None, run_args=None, inplace=True):
    q_model = model if inplace else copy.deepcopy(model)
    q_model = prepare(q_model, qconfig_mapping)
    if run_fn is not None:
        if run_args is not None:
            run_fn(q_model, *run_args)
        else:
            run_fn(q_model)
    q_model = convert(q_model, qconfig_mapping)
    return q_model


# def autotune(fp32_model, quant_config, tune_config, eval_func, ...):
#     pass
