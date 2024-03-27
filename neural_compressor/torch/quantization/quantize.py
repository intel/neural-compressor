# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from pathlib import Path
from neural_compressor.common.base_config import BaseConfig
from neural_compressor.torch.quantization import init_backend
from typing import Union


def prepare(model: torch.nn.Module, quant_config: Union[str, Path]):
    """Prepare the model for calibration.

    Insert observers into the model so that it can monitor the input and output tensors during calibration.

    Args:
        model (torch.nn.Module): origin model
        quant_config (Union[str, Path]): path to quantization config

    Returns:
        model with observers
    """
    backend = init_backend(quant_config)
    model = backend.prepare(model)
    setattr(model, "calib", True)
    return model


def convert(model: torch.nn.Module, quant_config, calib_result):
    """Convert the prepared model to a quantized model.

    Load the calibration results and apply post-processing to generate the quantized module.
    Then, swap out the original module with the newly created quantized module.

    Args:
        model (torch.nn.Module): the prepared model
    """
    backend = init_backend(quant_config)
    q_model = backend.convert(model, calib_result)
    setattr(model, "quant", True)
    return q_model


def save(model, fname="./saved_results"):
    if getattr(model, "calib", False):
        from neural_compressor.torch.algorithms.habana_fp8 import save_calib_result
        save_calib_result(model, fname)
    if getattr(model, "quant", False):
        from neural_compressor.torch.algorithms.habana_fp8 import save_fp8_model
        save_fp8_model(model, fname)


def load(model, fname="./saved_results"):
    pass
