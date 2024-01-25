# Copyright (c) 2023 Intel Corporation
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


from typing import Callable, Dict

import tensorflow as tf

from neural_compressor.common.utils import STATIC_QUANT
from neural_compressor.common.base_config import BaseConfig
from neural_compressor.tensorflow.utils import register_algo
from neural_compressor.tensorflow.auto_tune import generate_tune_config
from neural_compressor.tensorflow.algorithms.static_quantize.keras import KerasAdaptor
from neural_compressor.tensorflow.algorithms.static_quantize.tensorflow import TensorFlowAdaptor
from neural_compressor.tensorflow.model import BaseModel, KerasModel, framework_specific_info


@register_algo(name=STATIC_QUANT)
def static_quantize_entry(
    model: BaseModel,
    quant_config: BaseConfig,
    calib_dataloader: Callable = None,
    calib_iteration: int = 100,
):
    """The main entry to apply static quantization.

    Args:
        model: a fp32 model to be quantized.
        quant_config: a quantization configuration.
        calib_dataloader: a data loader for calibration.
        calib_iteration: the iteration of calibration.

    Returns:
        q_model: the quantized model.
    """
    Adaptor = KerasAdaptor if isinstance(model, KerasModel) else TensorFlowAdaptor
    adaptor = Adaptor(framework_specific_info)
    capability = adaptor.query_fw_capability(model)
    tune_cfg = generate_tune_config(model, quant_config, calib_iteration, \
                                        calib_dataloader, capability)
    q_model = adaptor.quantize(tune_cfg, model, calib_dataloader)
    return q_model
