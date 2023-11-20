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


from typing import Dict

import tensorflow as tf

from neural_compressor.adaptor.torch_utils.util import fetch_module, set_module

from neural_compressor.utils import logger
from neural_compressor import quantization
from neural_compressor.keras.utils import register_algo
from neural_compressor.common.utility import KERAS_STATIC_QUANT
from neural_compressor.keras.quantization.config import KerasStaticQuantConfig


@register_algo(name=KERAS_STATIC_QUANT)
def static_quantize_entry(model: tf.keras.Model, quant_config: KerasStaticQuantConfig, calib_dataloader) -> tf.keras.Model:
    """The main entry to apply keras basic quantization."""
    def fake_eval(model: tf.keras.Model):
        return 1
    q_model = quantization.fit(model, conf=quant_config, calib_dataloader=calib_dataloader, eval_func=fake_eval)
    return q_model
