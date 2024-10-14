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
"""Bf16 Convert for Torch Utils."""
import torch
import torch.nn as nn

from ...utils import logger
from .util import append_attr


class BF16ModuleWrapper(nn.Module):
    """BF16Module Wrapper Class."""

    def __init__(self, module):
        """Init a BF16ModuleWrapper object."""
        super(BF16ModuleWrapper, self).__init__()
        module = module.bfloat16()
        self.add_module("module", module)
        self.train(module.training)
        # WA for TransformerEncoder to access its Linear's weights and bias
        if isinstance(module, nn.Linear):
            self.weight = self.module.weight if hasattr(self.module, "weight") else None
            self.bias = self.module.bias if hasattr(self.module, "bias") else None

    def forward(self, X):
        """Convert dtype."""
        X = X.to(torch.bfloat16)
        X = self.module(X)
        return X.float()


def Convert(model, tune_cfg):
    """Convert to bf16 model.

    Args:
        model (object): the input model.
        tune_cfg (dict): dictionary of quantization configuration.

    Returns:
        mixed_precision_model (object): model with mixed precision.
    """
    bf16_ops_list = tune_cfg["bf16_ops_list"]
    if len(bf16_ops_list) > 0:
        logger.info("Convert operators to bfloat16")
    mixed_precision_model = _bf16_wrapper_model(model, bf16_ops_list)
    return mixed_precision_model


def _bf16_wrapper_model(model, bf16_ops_list, prefix=""):
    for name, child in model.named_children():
        op_name = prefix + "." + name if prefix != "" else name
        _bf16_wrapper_model(child, bf16_ops_list, op_name)
        for bf16_op_name in bf16_ops_list:
            if op_name == bf16_op_name[0] or op_name == bf16_op_name[0].split(".module")[0]:
                child_bf16 = BF16ModuleWrapper(child)
                append_attr(child_bf16, child)
                setattr(model, name, child_bf16)
    return model
