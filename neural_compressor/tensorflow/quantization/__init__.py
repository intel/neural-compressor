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
"""Intel Neural Compressor Tensorflow quantization API."""


from neural_compressor.tensorflow.quantization.quantize import quantize_model
from neural_compressor.tensorflow.quantization.autotune import autotune, get_all_config_set, TuningConfig
from neural_compressor.tensorflow.quantization.algorithm_entry import static_quant_entry, smooth_quant_entry
from neural_compressor.tensorflow.quantization.config import (
    StaticQuantConfig,
    SmoothQuantConfig,
    get_default_sq_config,
    get_default_static_quant_config,
)
