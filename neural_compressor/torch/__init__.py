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

from neural_compressor.torch.utils import register_algo
from neural_compressor.torch.algorithms import rtn_quantize_entry, gptq_quantize_entry

from neural_compressor.torch.quantization import (
    quantize,
    RTNWeightQuantConfig,
    get_default_rtn_config,
    GPTQConfig,
    get_default_gptq_config,
)
