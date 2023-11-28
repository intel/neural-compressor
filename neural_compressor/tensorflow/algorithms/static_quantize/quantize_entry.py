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

from collections import OrderedDict
from typing import Callable, Dict

import tensorflow as tf

from neural_compressor.common.utility import STATIC_QUANT
from neural_compressor.tensorflow.algorithms.static_quantize.keras import KerasAdaptor
from neural_compressor.tensorflow.quantization.config import StaticQuantConfig
from neural_compressor.tensorflow.utils import register_algo

framework_specific_info = {
    "device": "cpu",
    "backend": "itex",
    "approach": "post_training_static_quant",
}

support_int8_weight = {"Dense", "Conv2d", "DepthwiseConv2D", "SeparableConv2D"}

support_int8_activation = {
    "Dense",
    "Conv2d",
    "DepthwiseConv2D",
    "SeparableConv2D",
    "AvgPool2D",
    "AveragePooling2D",
    "MaxPool2D",
    "MaxPooling2D",
}


def update_config(op_value: Dict, quant_config: StaticQuantConfig, layer_class: str):
    """Update op-wise config from global config or operator name config or operator type config."""
    op_value["activation"].update(
        {
            "dtype": quant_config.act_dtype,
            "quant_mode": "static",
            "scheme": ("sym" if quant_config.act_sym else "asym"),
            "granularity": quant_config.act_granularity,
            "algorithm": "minmax",
        }
    )
    if layer_class not in support_int8_weight:
        return
    op_value["weight"] = {
        "dtype": quant_config.weight_dtype,
        "scheme": "sym" if quant_config.weight_sym else "asym",
        "granularity": quant_config.weight_granularity,
        "algorithm": "minmax",
    }


def parse_to_keras_tune_cfg(model: tf.keras.Model, quant_config: StaticQuantConfig, calib_iteration: int) -> Dict:
    """The function that parses StaticQuantConfig to keras tuning config.

    Args:
        model: a fp32 model to be quantized.
        quant_config: a quantization configuration.
        calib_iteration: the iteration of calibration.

    Returns:
        tune_cfg: the tuning config for keras adaptor.
    """
    tune_cfg = {"op": OrderedDict()}
    for layer in model.layers:
        layer_class = layer.__class__.__name__
        if layer_class not in support_int8_activation:
            continue
        op_key = (layer.name, layer_class)
        op_value = {"activation": {}}

        local_config = None
        # priority local > global
        if quant_config.local_config and layer.name in quant_config.local_config.keys():
            local_config = quant_config.local_config[layer.name]

        if local_config:
            update_config(op_value, local_config, layer_class)
        else:
            update_config(op_value, quant_config, layer_class)

        tune_cfg["op"].update({op_key: op_value})
        tune_cfg["calib_iteration"] = calib_iteration

    return tune_cfg


@register_algo(name=STATIC_QUANT)
def static_quantize_entry(
    model: tf.keras.Model,
    quant_config: StaticQuantConfig,
    calib_dataloader: Callable = None,
    calib_iteration: int = 100,
) -> tf.keras.Model:
    """The main entry to apply static quantization.

    Args:
        model: a fp32 model to be quantized.
        quant_config: a quantization configuration.
        calib_dataloader: a data loader for calibration.
        calib_iteration: the iteration of calibration.

    Returns:
        q_model: the quantized model.
    """

    keras_adaptor = KerasAdaptor(framework_specific_info)
    keras_adaptor.query_fw_capability(model)
    tune_cfg = parse_to_keras_tune_cfg(model, quant_config, calib_iteration)
    q_model = keras_adaptor.quantize(tune_cfg, model, calib_dataloader)
    return q_model
