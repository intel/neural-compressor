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
<<<<<<<< HEAD:neural_compressor/transformers/utils/__init__.py
"""Utils for optimization."""

from .quantization_config import RtnConfig, AwqConfig, TeqConfig, GPTQConfig, AutoRoundConfig
========

# First, run measurements (based on custom_config/measure_config.json)
QUANT_CONFIG=measure_config python3 imagenet_quant.py

# Next, run the quantized model (based on custom_config/quant_config.json)
QUANT_CONFIG=quant_config python3 imagenet_quant.py
>>>>>>>> 23fe77ec31ed8ef87e5b0717d7ab41eb0b34afc8:neural_compressor/torch/algorithms/fp8_quant/internal/inference_quant_examples/run_example.sh
