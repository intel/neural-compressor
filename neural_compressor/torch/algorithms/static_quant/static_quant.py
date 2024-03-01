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

import torch

try:
    import intel_extension_for_pytorch as ipex
except:
    assert False, "Please install IPEX for static quantization."

from packaging.version import Version

from .utility import (
    cfg_to_qconfig,
    dump_model_op_stats,
    get_ipex_version,
    get_quantizable_ops_recursively,
    ipex_config_path,
    simple_inference,
)

ipex_ver = get_ipex_version()


def static_quantize(model, tune_cfg, run_fn, example_inputs, inplace=True):
    """Execute the quantize process on the specified model.

    Args:
        model: a float model to be quantized.
        tune_cfg: quantization config for ops.
        run_fn: a calibration function for calibrating the model.
        example_inputs: used to trace torch model.
        inplace: whether to carry out model transformations in-place.

    Returns:
        A quantized model.
    """
    model.eval()

    if ipex_ver.release >= Version("1.12.0").release:
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
        run_fn(model)
        model.save_qconf_summary(qconf_summary=ipex_config_path)
        model = _ipex_post_quant_process(model, example_inputs, inplace=inplace)

    else:  # pragma: no cover
        # for IPEX version < 1.12
        _, cfgs, default_cfgs, fuse_ops = get_quantizable_ops_recursively(model, example_inputs)
        qscheme = cfg_to_qconfig(tune_cfg, cfgs, default_cfgs, fuse_ops)
        ipex_conf = ipex.quantization.QuantConf(
            configure_file=ipex_config_path, qscheme=qscheme
        )  # pylint: disable=E1101
        run_fn(model)
        ipex_conf.save(ipex_config_path)
        ipex_conf = ipex.quantization.QuantConf(ipex_config_path)  # pylint: disable=E1101
        model = ipex.quantization.convert(model, ipex_conf, example_inputs, inplace=True)  # pylint: disable=E1121

    with open(ipex_config_path, "r") as f:
        model.tune_cfg = json.load(f)
    model.ipex_config_path = ipex_config_path
    if ipex_ver.release >= Version("1.12.0").release:
        dump_model_op_stats(tune_cfg)
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
