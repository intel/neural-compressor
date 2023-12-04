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
"""Symbolic Trace for Torch Utils."""
import torch
from packaging.version import Version

from neural_compressor.adaptor.pytorch import PyTorch_FXAdaptor, get_torch_version

version = get_torch_version()


def trace_and_fuse_sub_graph(model, prefix, is_qat):
    """Trace and fuse subgraph of the model.

    Args:
        model (object): the input model.
        prefix (str): prefix of op name.
        is_qat (bool): if is qat.

    Returns:
        model (object).
    """
    import torch.quantization.quantization_mappings as tqqm
    from torch.quantization.quantize_fx import _fuse_fx

    fx_white_list = tqqm.get_default_qconfig_propagation_list()
    for name, module in model.named_children():
        # FX QAT cannot fallback nn.Dropout from train mode to eval
        if type(module) == torch.nn.Dropout:  # pragma: no cover
            continue
        op_name = prefix + "." + name if prefix != "" else name
        if type(module) in fx_white_list:
            module = torch.quantization.QuantWrapper(module)
        if PyTorch_FXAdaptor._check_dynamic_control(module):
            trace_and_fuse_sub_graph(module, op_name, is_qat=is_qat)
        else:
            try:
                graph_module = torch.fx.symbolic_trace(module)
                if version >= Version("1.11.0-rc1"):  # pragma: no cover
                    fused_model = _fuse_fx(graph_module, is_qat)
                else:
                    fused_model = _fuse_fx(graph_module)  # pylint: disable=E1120
                setattr(model, name, fused_model)
            except:
                trace_and_fuse_sub_graph(module, op_name, is_qat)
    return model


def symbolic_trace(model, is_qat=False):
    """Trace the model.

    Args:
        model (object): the input model.
        is_qat (bool): if is qat.

    Returns:
        traced_model (object).
    """
    try:
        traced_model = torch.fx.symbolic_trace(model)
    except:
        traced_model = trace_and_fuse_sub_graph(model, prefix="", is_qat=is_qat)
    return traced_model
