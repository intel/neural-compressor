"""Wanda utils."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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
import transformers

from neural_compressor.utils import logger
from neural_compressor.utils.utility import LazyImport

torch = LazyImport("torch")
nn = LazyImport("torch.nn")
F = LazyImport("torch.nn.functional")


def find_layers(module, op_types=["Linear", "Conv1D"], name=""):
    """Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    for layer_type in op_types:
        if layer_type in module.__class__.__name__:
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, op_types=op_types, name=name + "." + name1 if name != "" else name1))
    return res


def get_module_list(model):
    module_list = None
    for _, module in model.named_modules():
        if hasattr(type(module), "__name__") and "ModuleList" in type(module).__name__:
            module_list = module
            break
    assert module_list is not None, "cannot find any transformers layers, please check the model."
    return module_list


def get_tensor_sparsity_ratio(target_tensor):
    zero_cnt = float(torch.sum(target_tensor == 0.0).data.item())
    total_cnt = float(target_tensor.numel())
    return round(zero_cnt / total_cnt, 6)
