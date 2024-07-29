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
"""The entry interface for algorithms."""

from typing import Callable, Dict

from neural_compressor.common.base_config import BaseConfig
from neural_compressor.common.utils import SMOOTH_QUANT, STATIC_QUANT
from neural_compressor.tensorflow.algorithms import KerasAdaptor, Tensorflow_ITEXAdaptor, TensorFlowAdaptor
from neural_compressor.tensorflow.quantization.config import SmoothQuantConfig
from neural_compressor.tensorflow.utils import BaseModel, KerasModel, TFConfig, register_algo, valid_keras_format


@register_algo(name=STATIC_QUANT)
def static_quant_entry(
    model: BaseModel,
    quant_config: BaseConfig,
    calib_dataloader: Callable = None,
    calib_iteration: int = 100,
    calib_func: Callable = None,
):
    """The main entry to apply static quantization.

    Args:
        model: a fp32 model to be quantized.
        quant_config: a quantization configuration.
        calib_dataloader: a data loader for calibration.
        calib_iteration: the iteration of calibration.
        calib_func: the function used for calibration, should be a substitution for calib_dataloader
        when the built-in calibration function of INC does not work for model inference.

    Returns:
        q_model: the quantized model.
    """
    if isinstance(model, KerasModel):
        assert valid_keras_format(model.model), "Only Sequential or Functional models are supported now."
        framework = KerasAdaptor
    elif TFConfig.global_config["backend"] == "itex":
        framework = Tensorflow_ITEXAdaptor
    else:
        framework = TensorFlowAdaptor

    quantizer = framework(TFConfig.global_config)
    q_model = quantizer.quantize(quant_config, model, calib_dataloader, calib_iteration, calib_func)
    TFConfig.reset_global_config()

    return q_model


@register_algo(name=SMOOTH_QUANT)
def smooth_quant_entry(
    model: BaseModel,
    smooth_quant_config: SmoothQuantConfig,
    calib_dataloader: Callable = None,
    calib_iteration: int = 100,
    calib_func: Callable = None,
):
    """The main entry to apply smooth quantization.

    Args:
        model: a fp32 model to be quantized.
        quant_config: a quantization configuration.
        calib_dataloader: a data loader for calibration.
        calib_iteration: the iteration of calibration.
        calib_func: the function used for calibration, should be a substitution for calib_dataloader
        when the built-in calibration function of INC does not work for model inference.

    Returns:
        q_model: the quantized model.
    """
    assert not isinstance(model, KerasModel), "INC don't support smooth quantization for Keras models now."

    from neural_compressor.tensorflow.algorithms import SmoothQuant

    converter = SmoothQuant(smooth_quant_config, calib_dataloader, calib_iteration, calib_func)
    sq_model = converter(model)

    return sq_model
