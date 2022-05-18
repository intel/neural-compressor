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

import torch
import torch.nn as nn
from torch.fx import symbolic_trace

class BF16ModuleWrapper(nn.Module):
    def __init__(self, module):
        super(BF16ModuleWrapper, self).__init__()
        self.add_module('module', module)
        self.train(module.training)

    def forward(self, X):
        X = X.to(torch.bfloat16)
        self.module.bfloat16()
        X = self.module(X)
        return X.float()

def Convert(model, tune_cfg):
        bf16_ops_list = tune_cfg['bf16_ops_list']
        fx_sub_module_list = tune_cfg['fx_sub_module_list'] \
                             if 'fx_sub_module_list' in tune_cfg.keys() else []
        mixed_precision_model = bf16_wrapper_model(model, bf16_ops_list)
        if len(fx_sub_module_list) > 0:
            mixed_precision_model = bf16_symbolic_trace(mixed_precision_model, fx_sub_module_list)
        return mixed_precision_model

def bf16_wrapper_model(model, bf16_ops_list, prefix=''):
    for name, child in model.named_children():
        op_name = prefix + '.' + name if prefix != '' else name
        for bf16_op_name in bf16_ops_list:
            if op_name == bf16_op_name[0]:
                child = BF16ModuleWrapper(child)
        else:
            bf16_wrapper_model(child, bf16_ops_list, op_name)
            setattr(model, name, child)
    return model


def bf16_symbolic_trace(model, fx_sub_module_list, prefix=''):
    for name, child in model.named_children():
        op_name = prefix + '.' + name if prefix != '' else name
        for fx_sub_module_name in fx_sub_module_list:
            if op_name == fx_sub_module_name:
                child = symbolic_trace(child)
        else:
            bf16_symbolic_trace(child, fx_sub_module_list, op_name)
            setattr(model, name, child)
    return model