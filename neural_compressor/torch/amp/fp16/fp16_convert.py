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
"""FP16 Convert for Torch Utils."""

import torch

from torch.fx import symbolic_trace
from typing import Dict, Tuple

from neural_compressor.common import logger
from neural_compressor.quantization import MixPrecisionConfig

class FP16ModuleWrapper(torch.nn.Module):
    """FP16Module Wrapper Class."""

    def __init__(self, module, device="cpu"):
        """Init a FP16ModuleWrapper object."""
        super(FP16ModuleWrapper, self).__init__()
        self.add_module("module", module)
        self.train(module.training)
        self.device = device
        # WA for TransformerEncoder to access its Linear's weights and bias
        if isinstance(module, torch.nn.Linear):
            self.weight = self.module.weight if hasattr(self.module, "weight") else None
            self.bias = self.module.bias if hasattr(self.module, "bias") else None

    def forward(self, X):
        """Convert dtype."""
        with torch.autocast(device_type=self.device, dtype=torch.float16):
            X = self.module(X)
        return X.float()


class FP16Converter():
    """FP16 Converter Class."""

    def __init__(self, configs_mapping: Dict[Tuple[str], MixPrecisionConfig], *args, **kwargs):
        """Initialize the FP16 Converter with config.
 
        Args:
            config (MixPrecisionConfig): config class for mix-precision.
        """
        self.configs_mapping = configs_mapping
        # self.fp16_ops_list = config.tune_cfg["fp16_ops_list"]

    # def Convert(self, model: torch.nn.Module):
    #     """Convert to FP16 model.

    #     Args:
    #         model (torch.nn.Module): the input model.
    #         tune_cfg (dict): dictionary of quantization configuration.

    #     Returns:
    #         mixed_precision_model (object): model with mixed precision.
    #     """
    #     if len(self.fp16_ops_list) > 0:
    #         logger.info("Convert operators to float16")
    #     mixed_precision_model = self._fp16_wrapper_model(model, self.fp16_ops_list)
    #     # if fx_sub_module_list is not None and len(fx_sub_module_list) > 0:
    #     #     mixed_precision_model = FP16Converter.fp16_symbolic_trace(mixed_precision_model, fx_sub_module_list)
    #     return mixed_precision_model

    # def _fp16_wrapper_model(self, model: torch.nn.Module, prefix=""):
    #     for name, child in model.named_children():
    #         op_name = prefix + "." + name if prefix != "" else name
    #         for fp16_op_name in self.fp16_ops_list:
    #             if op_name == fp16_op_name[0]:
    #                 child = FP16ModuleWrapper(child, device)
    #         else:
    #             self._fp16_wrapper_model(child, op_name)
    #             setattr(model, name, child)
    #     return model

    # @staticmethod
    # def fp16_symbolic_trace(model, fx_sub_module_list, prefix=""):
    #     """Symbolic trace for fp16 models.

    #     Args:
    #         model (object): the input model.
    #         fx_sub_module_list (list): _description_
    #         prefix (str): prefix of op name.

    #     Returns:
    #         model (object)
    #     """
    #     for name, child in model.named_children():
    #         op_name = prefix + "." + name if prefix != "" else name
    #         for fx_sub_module_name in fx_sub_module_list:
    #             if op_name == fx_sub_module_name:
    #                 child = symbolic_trace(child)
    #         else:
    #             FP16Converter.fp16_symbolic_trace(child, fx_sub_module_list, op_name)
    #             setattr(model, name, child)
    #     return model
