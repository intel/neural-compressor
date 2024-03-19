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
from neural_compressor.torch.quantization import init_quantizer

# another proposal to not pass `quant_config` as a parameter
# global quant_config


def prepare(model: torch.nn.Module, quant_config: BaseConfig):
    """Prepare the model for calibration.

    Insert observers into the model so that it can monitor the input and output tensors during calibration.

    Args:
        model (torch.nn.Module): origin model
        quant_config (BaseConfig): quantization config, including observer method

    Returns:
        model with observers
    """
    if _need_calibration():
        quantizer = init_quantizer(model, quant_config)
        prepared_model = quantizer.prepare(model)
        return prepared_model
    else:
        return model


def convert(model: torch.nn.Module, quant_config: BaseConfig):
    """Convert the origin model to a quantized model.

    Load the calibration results and apply post-processing to generate the quantized module. Then, swap out the original module with the newly created quantized module.

    Args:
        model (torch.nn.Module): the origin model
        quant_config (BaseConfig): quantization config, including scale method
    """
    quantizer = init_quantizer(model, quant_config)
    q_model = quantizer.convert(model, quant_config)
    return q_model


def save_calibration_result(model: torch.nn.Module, quant_config: BaseConfig):
    """Save calibration result to local file.

    Args:
        model (torch.nn.Module): model with observers (output model of prepare() func)
        quant_config (BaseConfig): including save path of calibration results
    """
    if _need_calibration():
        _save_calibration_results(model, quant_config)
    else:
        return


def _need_calibration():
    return True


def _save_calibration_results(model, quant_config):
    return
