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
from neural_compressor.common.base_config import BaseConfig
from neural_compressor.torch.quantization import init_backend
from typing import Union


def prepare(model: torch.nn.Module, quant_config: Union[BaseConfig, str, dict]):
    """Prepare the model for calibration.

    Insert observers into the model so that it can monitor the input and output tensors during calibration.

    Args:
        model (torch.nn.Module): origin model
        quant_config (Union[BaseConfig, str, dict]): quantization config

    Returns:
        model with observers
    """
    # TODO: process quant_config according to its type

    if _need_calibration():
        backend = init_backend(model, quant_config)
        model = backend.prepare(model)

    # TODO: quant config should be assigned to model in `.qconfig` attribute.
    model.qconfig = quant_config
    return model


def convert(model: torch.nn.Module):
    """Convert the prepared model to a quantized model.

    Load the calibration results and apply post-processing to generate the quantized module.
    Then, swap out the original module with the newly created quantized module.

    Args:
        model (torch.nn.Module): the prepared model
    """
    if _need_convert():
        backend = init_backend(model)
        q_model = backend.convert(model)
        return q_model
    else:
        return model


def save_calibration_result(model: torch.nn.Module):
    """Save calibration result to local file.

    Args:
        model (torch.nn.Module): model with observers (output model of prepare() func)
    """
    if _need_calibration():
        _save_calibration_results(model)
    else:
        return


def _need_calibration():
    return True

def _need_convert():

    return True

def _save_calibration_results(model):
    return
