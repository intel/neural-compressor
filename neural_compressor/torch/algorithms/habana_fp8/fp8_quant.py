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

# pylint:disable=import-error

import copy
import os

import habana_frameworks.torch.core as htcore
import torch
from deepspeed.module_inject import LinearAllreduce, LinearLayer
from deepspeed.module_inject.layers import LmHeadLinearAllreduce
from habana_frameworks.torch.core.quantization import _check_params_as_const, _mark_params_as_const

from neural_compressor.torch.utils import fetch_module, logger, set_module

from .modules import (  # fp32; dynamic modules; static modules; dtype amax
    Autocast,
    BatchMatmul,
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
    Matmul,
)
from .observer import observer_mapping

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


FP8_DTYPE = [torch.float8_e5m2, torch.float8_e4m3fn, "fp8_e5m2", "fp8_e4m3"]
dtype_mapping = {"fp8_e5m2": torch.float8_e5m2, "fp8_e4m3": torch.float8_e4m3fn}
# enable inference optimizations
htcore.hpu_initialize()


def _replace_module(module, qconfig):
    assert qconfig.w_dtype == qconfig.act_dtype, "weight and activation should be the same dtype."
    dtype = dtype_mapping[qconfig.w_dtype]
    # only modules that have weight should use this observer
    if hasattr(module, "weight"):
        observer_cls = observer_mapping[qconfig.w_observer]
        observer_obj = observer_cls(dtype=dtype)
    if qconfig.approach == "static":
        if isinstance(module, white_list):
            QModule = quantization_mapping[type(module)]
            qmodule = QModule(module, dtype)
    elif qconfig.approach == "dynamic":
        if isinstance(module, torch.nn.Linear):
            # need module for initialization
            qmodule = FP8DynamicLinear(module, dtype)
        elif isinstance(module, Matmul):
            qmodule = FP8DynamicMatmul(dtype)
        elif isinstance(module, BatchMatmul):
            qmodule = FP8DynamicBatchMatmul(dtype)
        elif isinstance(module, Autocast):
            qmodule = FP8Cast(dtype=dtype)
    # only modules that have weight should use this API
    if hasattr(qmodule, "from_float"):
        qmodule.from_float(module, observer_obj)
    return qmodule


def quantize_dynamic(model, dtype=torch.float8_e4m3fn, inplace=True):
    torch.set_grad_enabled(False)
    q_model = model if inplace else copy.deepcopy(model)
    if isinstance(dtype, str):
        dtype = dtype_mapping[dtype]
    for n, m in q_model.named_modules():
        if isinstance(m, torch.nn.Linear):
            observer_cls = observer_mapping["minmax_per_channel"]
            observer_obj = observer_cls(dtype=dtype)
            new_m = FP8DynamicLinear(m, dtype)  # need m for init
            new_m.from_float(m, observer_obj)
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
    _mark_params_as_const(q_model)
    _check_params_as_const(q_model)
    return q_model


def _add_observer(module, qconfig):
    act_observer = qconfig.act_observer

    def input_observer_forward_pre_hook(self, input):
        try:
            if isinstance(input[0], torch.Tensor):
                self.input_activation_post_process(input[0])
            if hasattr(self, "input_activation_post_process1") and isinstance(input[1], torch.Tensor):
                self.input_activation_post_process1(input[1])
            return input
        except Exception as e:
            # The KL act_observer may encounter a overflow error on EltwiseAdd.
            pass

    ### Insert input observer into model, only for fp8_e4m3 static quantization ###
    observer_cls = observer_mapping[act_observer]
    # import pdb;pdb.set_trace()

    if isinstance(module, white_list):
        observer_obj = observer_cls(dtype=dtype_mapping[qconfig.act_dtype])
        module.add_module("input_activation_post_process", observer_obj)
    if isinstance(module, (BatchMatmul, Matmul)):
        observer_obj = observer_cls(dtype=dtype_mapping[qconfig.act_dtype])
        module.add_module("input_activation_post_process1", observer_obj)
    module.register_forward_pre_hook(input_observer_forward_pre_hook)


def _remove_observer(module):
    import deepspeed.comm as dist
    from torch.distributed import ReduceOp

    if hasattr(module, "input_activation_post_process"):
        scale = module.input_activation_post_process.calculate_qparams()
        if dist.is_initialized():
            scale = scale.to("hpu")
            dist.all_reduce(scale, op=ReduceOp.MAX)
        if hasattr(module, "input_activation_post_process1"):
            module.register_parameter("scale1", torch.nn.Parameter(scale))
        else:
            module.register_parameter("scale", torch.nn.Parameter(scale))
        delattr(module, "input_activation_post_process")
    if hasattr(module, "input_activation_post_process1"):
        scale = module.input_activation_post_process1.calculate_qparams()
        if dist.is_initialized():
            scale = scale.to("hpu")
            dist.all_reduce(scale, op=ReduceOp.MAX)
        module.register_parameter("scale2", torch.nn.Parameter(scale))
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
    model.qconfig = qconfig_mapping
    for (op_name, op_type), qconfig in qconfig_mapping.items():
        if qconfig.approach == "dynamic":
            continue
        if qconfig.w_dtype not in FP8_DTYPE:
            continue
        module = fetch_module(model, op_name)
        if module is None:
            logger.info(f"{op_name} is not found in model.")
            continue
        _add_observer(module, qconfig)
        set_module(model, op_name, module)
    return model


def convert(model):
    for (op_name, op_type), qconfig in model.qconfig.items():
        if qconfig.w_dtype not in FP8_DTYPE:
            continue
        module = fetch_module(model, op_name)
        if module is None:
            logger.info(f"{op_name} is not found in model.")
            continue
        if qconfig.approach != "dynamic":
            _remove_observer(module)
        module = _replace_module(module, qconfig)
        set_module(model, op_name, module)
        htcore.mark_step()
    return model


def quantize(model, qconfig_mapping, run_fn=None, run_args=None, inplace=True):
    torch.set_grad_enabled(False)
    q_model = model if inplace else copy.deepcopy(model)
    q_model = prepare(q_model, qconfig_mapping)
    if run_fn is not None:
        if run_args is not None:
            run_fn(q_model, *run_args)
        else:
            run_fn(q_model)
    q_model = convert(q_model)
    _mark_params_as_const(q_model)
    _check_params_as_const(q_model)
    return q_model
