#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 MIT HAN Lab
# This source code is licensed under the MIT license
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
"""RTN quantization."""

import copy
from collections import OrderedDict

import torch

from neural_compressor.torch.algorithms import Quantizer
from neural_compressor.torch.utils import (
    get_accelerator,
    get_attr,
    get_model_device,
    is_transformers_imported,
    logger,
    set_attr,
    set_module,
)

from .modules import INCWeightOnlyLinear
from .utility import cast_fp8, quant_tensor, search_clip

if is_transformers_imported():
    import transformers


class RTNQuantizer(Quantizer):
    """RTN Quantizer."""

    def __init__(self, quant_config: OrderedDict = {}):
        """Init a RTNQuantizer object.

        Args:
            quant_config (OrderedDict, optional): quantization config for ops. Defaults to {}.
        """
        super().__init__(quant_config)

    @torch.no_grad()
    def prepare(self, model, *args, **kwargs):
        """Prepares a given model for quantization.

        Will return model directly in RTN algorithm.

        Args:
            model (torch.nn.Module): The model to be prepared.
        """
        return model

    @torch.no_grad()
    def convert(
        self,
        model,
        dtype="int",
        bits=4,
        scheme="sym",
        group_size=32,
        group_dim=1,
        quantile=1.0,
        use_full_range=False,
        use_mse_search=False,
        use_layer_wise=False,
        model_path="",
        quant_lm_head=False,
        *args,
        **kwargs,
    ):
        """Quant the model with round to nearest method and inplace is True.

        Args:
            model: torch module
            dtype (str, optional): select from int, nf4, fp4. Defaults to int.
            bits: num bits. Defaults to 4.
            scheme (str, optional): sym or asym. Defaults to "sym".
            group_size (int, optional): how many elements share one scale/zp. Defaults to 32.
            group_dim (int, optional):  0 means splitting output channel,
                                        1 means splitting input channel. Defaults to 1.
            quantile (float, optional): percentile of clip. Defaults to 1.0.
            use_full_range (bool, optional): Choose sym range whether use -2**(bits-1).
                                        Defaults to False.
            use_mse_search (bool, optional):  Whether to search clip range.
                                        Defaults to True.
            quant_lm_head (bool, optional):  Whether to quantize the lm_head layer.
                                        Defaults to False.

        Returns:
            model: fake quantized torch module
        """
        weight_config = self.quant_config
        device = get_accelerator(kwargs.pop("device", "auto")).current_device_name()
        model_device = get_model_device(model)  # return model on the same device

        # for transformers model. If lm_head is tied from embedding, we deepcopy it.
        if quant_lm_head and getattr(getattr(model, "config", None), "tie_word_embeddings", False):
            for key in model._tied_weights_keys:
                weight = get_attr(model, key)
                set_attr(model, key, copy.deepcopy(weight))

        assert isinstance(model, torch.nn.Module), "only support torch module"
        if is_transformers_imported():
            supported_layers = (torch.nn.Linear, transformers.Conv1D)
        else:
            supported_layers = (torch.nn.Linear,)
        # initialize global configuration
        double_quant_config = {
            "double_quant": kwargs.get("use_double_quant", False),
            "double_quant_dtype": kwargs.get("double_quant_dtype", "int"),
            "double_quant_bits": kwargs.get("double_quant_bits", 8),
            "double_quant_scheme": kwargs.get("double_quant_scheme", "sym"),
            "double_quant_group_size": kwargs.get("double_quant_group_size", 256),
        }
        use_optimum_format = kwargs.get("use_optimum_format", True)

        if use_layer_wise:
            from neural_compressor.common.utils import DEFAULT_WORKSPACE
            from neural_compressor.torch.algorithms.layer_wise.utils import get_path, load_module

            if model_path == "":
                model_path = model.path
            assert model_path, "model_path should not be None."
            model_path = get_path(model_path)

        for name, m in model.named_modules():
            if use_layer_wise and len(list(m.named_children())) == 0:
                load_module(model, name, model_path, device=device)
            if not isinstance(m, supported_layers):
                continue
            if name in weight_config:  # pragma: no cover
                # initialize op configuration
                dtype = weight_config[name].get("dtype", "int")
                if dtype == "fp32":
                    continue
                # Move modules to the accelerator device layer-by-layer
                if not use_layer_wise:
                    m.to(device)
                ### FP8 cast part
                if dtype in ["fp8_e5m2", "fp8_e5m2fnuz", "fp8_e4m3fn", "fp8_e4m3fnuz"]:
                    logger.debug("Cast module {} to FP8 using qdq mode, no scaling".format(name))
                    m.weight = cast_fp8(m.weight, dtype, use_qdq=True)
                    continue
                ####
                logger.debug("Apply RTN on module %s.", name)
                bits = weight_config[name].get("bits", 4)
                group_size = weight_config[name]["group_size"]
                scheme = weight_config[name]["scheme"]
                quantile = weight_config[name].get("quantile", 1.0)
                group_dim = weight_config[name]["group_dim"]
                use_full_range = weight_config[name]["use_full_range"]
                use_mse_search = weight_config[name]["use_mse_search"]
                use_optimum_format = kwargs.get("use_optimum_format", True)
                # double quant config
                double_quant_config = {
                    "double_quant": weight_config[name]["use_double_quant"],
                    "double_quant_dtype": weight_config[name]["double_quant_dtype"],
                    "double_quant_bits": weight_config[name]["double_quant_bits"],
                    "double_quant_scheme": weight_config[name]["double_quant_scheme"],
                    "double_quant_group_size": weight_config[name]["double_quant_group_size"],
                }
                if dtype != "int" and "int" in dtype:
                    bits = int(dtype.lstrip("int"))
                    dtype = "int"
            else:
                continue
            log_msg = (
                f"RTN quantization config: bits={bits}, group_size={group_size}, "
                + f"scheme={scheme}, quantile={quantile}"
            )
            if dtype != "int":
                log_msg += f", dtype={dtype}"
            elif scheme == "sym":  # nf4/fp4 is always [-7,7]
                log_msg += f", use_full_range={use_full_range}"
            if dtype == "fp32":
                continue
            logger.debug(f"RTN quantized module:{name, m}")
            logger.debug(log_msg)

            # for only group_dim is 0 or only `transformers.Conv1D`, we need transpose weight.
            if is_transformers_imported():
                transpose = (group_dim == 0) ^ (isinstance(m, transformers.Conv1D))
            else:
                transpose = group_dim == 0
            if transpose:
                weight = m.weight.detach().T.contiguous()
            else:
                weight = m.weight.detach()
            if use_mse_search:
                quantile = search_clip(m, bits, group_size, scheme, dtype, use_full_range)
            int_weight, scale, zp = quant_tensor(
                weight,
                dtype=dtype,
                bits=bits,
                group_size=group_size,
                scheme=scheme,
                quantile=quantile,
                return_int=True,
                full_range=use_full_range,
                **double_quant_config,
            )
            int_weight = int_weight.t_().contiguous() if transpose else int_weight
            scale = scale.t_().contiguous() if transpose else scale
            zp = zp.t_().contiguous() if transpose and zp is not None else zp
            if isinstance(m, torch.nn.Linear):
                in_features = m.in_features
                out_features = m.out_features
            elif is_transformers_imported() and isinstance(m, transformers.Conv1D):
                in_features = m.weight.shape[0]
                out_features = m.weight.shape[1]
                int_weight = int_weight.t_().contiguous()
                scale = scale.t_().contiguous()
                zp = zp.t_().contiguous() if zp is not None else zp

            new_module = INCWeightOnlyLinear(
                in_features,
                out_features,
                dtype=dtype,
                bits=bits,
                group_size=group_size,
                zp=zp is not None,
                bias=m.bias is not None,
                use_optimum_format=use_optimum_format,
                device=device,
            )
            new_module.pack(int_weight, scale, zp, m.bias)

            if use_layer_wise:
                m = m.to_empty(device=torch.device("meta"))
            if name == "":
                return new_module
            else:
                set_module(model, name, new_module)
            # Move modules back to the model device layer-by-layer
            if not use_layer_wise:
                m.to(model_device)
                new_module.to(model_device)
        if not use_layer_wise:
            model.to(model_device)
        return model
