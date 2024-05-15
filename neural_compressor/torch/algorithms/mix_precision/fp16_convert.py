#
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
"""FP16 Convert for Torch Modules."""

import torch

from typing import Dict, Tuple

from neural_compressor.common import logger
from neural_compressor.torch.algorithms.mix_precision.module_wrappers import FP16ModuleWrapper
from neural_compressor.torch.quantization import MixPrecisionConfig
from neural_compressor.torch.utils import get_device


class FP16Converter():
    """FP16 Converter Class."""

    def __init__(self, configs_mapping: Dict[Tuple[str], MixPrecisionConfig], *args, **kwargs):
        """Initialize the FP16 Converter with config.
 
        Args:
            config (MixPrecisionConfig): config class for mix-precision.
        """
        self.configs_mapping = configs_mapping
        self.device = "auto"
        for _, config in configs_mapping.items():
            self.device = config.device
            break
        if self.device == "auto":
            self.device = get_device()

    def convert(self, model: torch.nn.Module):
        """Convert to FP16 model.

        Args:
            model (torch.nn.Module): the input model.
            tune_cfg (dict): dictionary of quantization configuration.

        Returns:
            mixed_precision_model (object): model with mixed precision.
        """
        if len(self.configs_mapping) > 0:
            logger.info("Convert operators to float16")
        mixed_precision_model = self._fp16_wrapper_model(model)
        return mixed_precision_model

    def _fp16_wrapper_model(self, model: torch.nn.Module, prefix=""):
        for name, child in model.named_children():
            op_name = prefix + "." + name if prefix != "" else name
            for op_info, config in self.configs_mapping.items():
                if op_name == op_info[0] and config.dtype == "fp16":
                    child = FP16ModuleWrapper(child, self.device)
            else:
                self._fp16_wrapper_model(child, op_name)
                setattr(model, name, child)
        return model
