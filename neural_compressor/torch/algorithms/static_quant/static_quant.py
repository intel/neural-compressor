#!/usr/bin/env python
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

import json
from copy import deepcopy
from types import MethodType

import torch

try:
    import intel_extension_for_pytorch as ipex
except:
    assert False, "Please install IPEX for static quantization."

from collections import OrderedDict

from packaging.version import Version

from neural_compressor.torch.algorithms import Quantizer
from neural_compressor.torch.utils import logger

from .utility import (
    cfg_to_qconfig,
    dump_model_op_stats,
    get_ipex_version,
    get_quantizable_ops_recursively,
    ipex_config_path,
    simple_inference,
)

ipex_ver = get_ipex_version()


class StaticQuantQuantizer(Quantizer):
    def __init__(self, quant_config: OrderedDict = {}):
        """Init a StaticQuantQuantizer object.

        Args:
            quant_config (OrderedDict, optional): quantization config for ops. Defaults to {}.
        """
        super().__init__(quant_config)

    def prepare(self, model, example_inputs, inplace=True, *args, **kwargs):
        """Prepares a given model for quantization.

        Args:
            model: A float model to be quantized.
            example_inputs: Used to trace torch model.
            inplace: Whether to carry out model transformations in-place. Defaults to True.

        Returns:
            A prepared model.
        """
        assert example_inputs is not None, "Please provide example_inputs for static quantization."

        _, cfgs, op_infos_from_cfgs, output_tensor_id_op_name, _ = get_quantizable_ops_recursively(
            model, example_inputs
        )
        # update json file in ipex_config_path; map ipex op_name to pt op_name
        user_cfg = cfg_to_qconfig(self.tune_cfg, cfgs, op_infos_from_cfgs, output_tensor_id_op_name)
        model.eval()

        # Check save_qconf_summary part is a workaround for IPEX bug.
        # Sometimes the prepared model from get_op_capablitiy loss this attribute
        if not hasattr(model, "save_qconf_summary") or not hasattr(model, "load_qconf_summary"):
            from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig

            if ipex_ver.release >= Version("2.1").release:
                static_qconfig = ipex.quantization.default_static_qconfig_mapping
            else:
                static_qconfig = QConfig(
                    activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
                    weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
                )
            if isinstance(example_inputs, dict):
                model = ipex.quantization.prepare(
                    model, static_qconfig, example_kwarg_inputs=example_inputs, inplace=inplace
                )
            else:
                model = ipex.quantization.prepare(model, static_qconfig, example_inputs=example_inputs, inplace=inplace)

        model.load_qconf_summary(qconf_summary=ipex_config_path)
        setattr(model, "user_cfg", user_cfg)
        return model

    def convert(self, model, example_inputs, inplace=True, *args, **kwargs):
        """Converts a prepared model to a quantized model.

        Args:
            model: The prepared model to be converted.
            example_inputs: Used to trace torch model.
            inplace: Whether to carry out model transformations in-place. Defaults to True.

        Returns:
            A quantized model.
        """
        from neural_compressor.torch.algorithms.static_quant import save

        user_cfg = getattr(model, "user_cfg", OrderedDict())

        model.save_qconf_summary(qconf_summary=ipex_config_path)
        model = _ipex_post_quant_process(model, example_inputs, inplace=inplace)

        with open(ipex_config_path, "r") as f:
            model.tune_cfg = json.load(f)
        model.ipex_config_path = ipex_config_path

        dump_model_op_stats(user_cfg)

        logger.info("Static quantization done.")
        model.ori_save = model.save
        model.save = MethodType(save, model)
        return model


def _ipex_post_quant_process(model, example_inputs, inplace=False):
    """Convert to a jit model.

    Args:
        model: a prepared model.
        example_inputs: used to trace torch model.
        inplace: whether to carry out model transformations in-place.

    Returns:
        A converted jit model.
    """
    model = ipex.quantization.convert(model, inplace=inplace)
    with torch.no_grad():
        try:
            if isinstance(example_inputs, dict):
                model = torch.jit.trace(model, example_kwarg_inputs=example_inputs)
            else:
                model = torch.jit.trace(model, example_inputs)
            model = torch.jit.freeze(model.eval())
        except:
            if isinstance(example_inputs, dict):
                model = torch.jit.trace(model, example_kwarg_inputs=example_inputs, strict=False, check_trace=False)
            else:
                model = torch.jit.trace(model, example_inputs, strict=False)
            model = torch.jit.freeze(model.eval())
    # After freezing, run 1 time to warm up the profiling graph executor to insert prim::profile
    # At the 2nd run, the llga pass will be triggered and the model is turned into
    # an int8 model: prim::profile will be removed and will have LlgaFusionGroup in the graph
    simple_inference(model, example_inputs, iterations=2)
    return model
