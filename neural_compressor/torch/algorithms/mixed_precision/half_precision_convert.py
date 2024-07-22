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
"""Half-precision Convert for Torch Modules."""

from typing import Dict, Tuple

import torch

from neural_compressor.common import logger
from neural_compressor.torch.algorithms.mixed_precision.module_wrappers import HalfPrecisionModuleWrapper
from neural_compressor.torch.utils import get_accelerator


class HalfPrecisionConverter:
    """Converter Class for FP16 and BF16."""

    dtype_mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    def __init__(self, configs_mapping: Dict[Tuple[str], object], *args, **kwargs):
        """Initialize the Half-precision Converter with config.

        Args:
            configs_mapping (Dict): config class for mixed-precision.
        """
        self.configs_mapping = configs_mapping
        self.device = get_accelerator().current_device_name()

    def convert(self, model: torch.nn.Module):
        """Convert to FP16 or BF16 model.

        Args:
            model (torch.nn.Module): the input model.

        Returns:
            mixed_precision_model (torch.nn.Module): model with mixed-precision.
        """
        if len(self.configs_mapping) > 0:
            logger.info("Convert operators to half-precision")

        if next(model.parameters()).is_cuda:
            self.device = "cuda"
        elif next(model.parameters()).is_cpu:
            self.device = "cpu"

        mixed_precision_model = self._wrap_half_precision_model(model)
        mixed_precision_model.to(self.device)

        return mixed_precision_model

    def _wrap_half_precision_model(self, model: torch.nn.Module, prefix=""):
        """Wrap and replace half-precision target modules.

        Args:
            model (torch.nn.Module): the input module.
            prefix (str): the name prefix for named children.

        Returns:
            model (torch.nn.Module): the model whose target modules have been wrapped.
        """
        for name, child in model.named_children():
            op_name = prefix + "." + name if prefix != "" else name
            for op_info, config in self.configs_mapping.items():
                if op_name == op_info[0] and config.dtype in ("fp16", "bf16"):
                    child = HalfPrecisionModuleWrapper(
                        module=child, device=self.device, dtype=self.dtype_mapping[config.dtype]
                    )
            else:
                self._wrap_half_precision_model(child, op_name)
                setattr(model, name, child)

        return model
