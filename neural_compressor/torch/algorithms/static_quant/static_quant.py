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
"""The quantizer using StaticQuant path."""


import json
import os
from copy import deepcopy
from types import MethodType

import torch

try:
    import intel_extension_for_pytorch as ipex
except:  # pragma: no cover
    assert False, "Please install IPEX for static quantization."

from collections import OrderedDict

from packaging.version import Version

from neural_compressor.torch.algorithms import Quantizer
from neural_compressor.torch.utils import logger
from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator

from .utility import (
    CpuInfo,
    cfg_to_qconfig,
    dump_model_op_stats,
    generate_xpu_qconfig,
    get_ipex_version,
    get_quantizable_ops_recursively,
    ipex_config_path,
    simple_inference,
)

ipex_ver = get_ipex_version()


class StaticQuantQuantizer(Quantizer):
    """The quantizer using Static Quant."""

    def __init__(self, quant_config: OrderedDict = {}):
        """Init a StaticQuantQuantizer object.

        Args:
            quant_config (OrderedDict, optional): quantization config for ops. Defaults to {}.
        """
        super().__init__(quant_config)
        self.user_cfg = OrderedDict()
        self.device = auto_detect_accelerator().current_device()

    def prepare(self, model, example_inputs, inplace=True, *args, **kwargs):
        """Prepares a given model for quantization.

        Args:
            model (torch.nn.Module): raw fp32 model or prepared model.
            example_inputs (tensor/tuple/dict): used to trace torch model.
            inplace (bool, optional): whether to carry out model transformations in-place. Defaults to True.

        Returns:
            A prepared model.
        """
        assert example_inputs is not None, "Please provide example_inputs for static quantization."

        if self.device == "cpu":
            _, cfgs, op_infos_from_cfgs, output_tensor_id_op_name, _ = get_quantizable_ops_recursively(
                model, example_inputs
            )
            # update json file in ipex_config_path; map ipex op_name to pt op_name
            self.user_cfg = cfg_to_qconfig(self.quant_config, cfgs, op_infos_from_cfgs, output_tensor_id_op_name)
        else:  # pragma: no cover
            model = model.to("xpu")

        model.eval()

        # Check save_qconf_summary part is a workaround for IPEX bug.
        # Sometimes the prepared model from get_op_capablitiy loss this attributes
        if not hasattr(model, "save_qconf_summary") or not hasattr(model, "load_qconf_summary"):  # pragma: no cover
            from torch.ao.quantization import HistogramObserver, MinMaxObserver, PerChannelMinMaxObserver, QConfig

            if self.device != "cpu":  # pragma: no cover
                from torch.quantization.quantize_jit import prepare_jit

                with torch.no_grad():
                    modelJit = torch.jit.trace(model, example_inputs)
                qconfig = generate_xpu_qconfig(self.quant_config)
                model = prepare_jit(modelJit, qconfig, inplace)
            else:
                if ipex_ver.release >= Version("2.1").release:
                    # HistogramObserver will cause a performance issue.
                    # static_qconfig = ipex.quantization.default_static_qconfig_mapping
                    qconfig = QConfig(
                        activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
                        weight=PerChannelMinMaxObserver.with_args(
                            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
                        ),
                    )
                    from torch.ao.quantization import QConfigMapping

                    static_qconfig = QConfigMapping().set_global(qconfig)
                else:  # pragma: no cover
                    static_qconfig = QConfig(
                        activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
                        weight=PerChannelMinMaxObserver.with_args(
                            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
                        ),
                    )
                if isinstance(example_inputs, dict):
                    model = ipex.quantization.prepare(
                        model, static_qconfig, example_kwarg_inputs=example_inputs, inplace=inplace
                    )
                else:
                    model = ipex.quantization.prepare(
                        model, static_qconfig, example_inputs=example_inputs, inplace=inplace
                    )

        if self.device == "cpu":
            model.load_qconf_summary(qconf_summary=ipex_config_path)

        return model

    def convert(self, model, example_inputs, inplace=True, *args, **kwargs):
        """Converts a prepared model to a quantized model.

        Args:
            model (QuantizationInterceptionModule): the prepared model to be converted.
            example_inputs (tensor/tuple/dict): used to trace torch model.
            inplace (bool, optional): whether to carry out model transformations in-place. Defaults to True.

        Returns:
            A quantized model.
        """
        use_bf16 = self.quant_config.get("use_bf16", None)

        from neural_compressor.torch.algorithms.static_quant import save

        if self.device != "cpu":  # pragma: no cover
            from torch.quantization.quantize_jit import convert_jit

            model = convert_jit(model, inplace)
            simple_inference(model, example_inputs, iterations=2)
            model.qconfig = self.quant_config["op"]
            dump_model_op_stats(model.qconfig)
        else:
            model.save_qconf_summary(qconf_summary=ipex_config_path)
            model = _ipex_post_quant_process(model, example_inputs, use_bf16, inplace=inplace)

            with open(ipex_config_path, "r") as f:
                model.tune_cfg = json.load(f)
            model.ipex_config_path = ipex_config_path

            dump_model_op_stats(self.user_cfg)

        model.ori_save = model.save
        model.save = MethodType(save, model)

        logger.info("Static quantization done.")
        return model


def _ipex_post_quant_process(model, example_inputs, use_bf16, inplace=False):
    """Converts to a jit model.

    Args:
        model (QuantizationInterceptionModule): a prepared model.
        example_inputs (tensor/tuple/dict): used to trace torch model.
        use_bf16 (bool): whether to use bf16 for mixed precision.
        inplace (bool, optional): whether to carry out model transformations in-place. Defaults to True.

    Returns:
        A converted jit model.
    """
    if use_bf16 and (CpuInfo().bf16 or os.getenv("FORCE_BF16") == "1"):  # pragma: no cover
        with torch.no_grad():
            with torch.cpu.amp.autocast():
                model = ipex.quantization.convert(model, inplace=inplace)
                try:
                    if isinstance(example_inputs, dict):
                        model = torch.jit.trace(model, example_kwarg_inputs=example_inputs)
                    else:
                        model = torch.jit.trace(model, example_inputs)
                    model = torch.jit.freeze(model.eval())
                except:
                    if isinstance(example_inputs, dict):
                        model = torch.jit.trace(
                            model, example_kwarg_inputs=example_inputs, strict=False, check_trace=False
                        )
                    else:
                        model = torch.jit.trace(model, example_inputs, strict=False)
                    model = torch.jit.freeze(model.eval())
    else:  # pragma: no cover
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
