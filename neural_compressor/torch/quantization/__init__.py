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
"""Intel Neural Compressor PyTorch quantization API."""

from neural_compressor.torch.quantization.quantize import quantize, prepare, convert, finalize_calibration
from neural_compressor.torch.quantization.config import (
    RTNConfig,
    get_default_rtn_config,
    get_default_double_quant_config,
    GPTQConfig,
    get_default_gptq_config,
    AWQConfig,
    get_default_awq_config,
    INT8StaticQuantConfig,
    StaticQuantConfig,
    get_default_static_config,
    SmoothQuantConfig,
    get_default_sq_config,
    TEQConfig,
    get_default_teq_config,
    AutoRoundConfig,
    get_default_AutoRound_config,
    HQQConfig,
    get_default_hqq_config,
    FP8Config,
    get_default_fp8_config,
    get_default_fp8_config_set,
    MXQuantConfig,
    get_default_mx_config,
    MixedPrecisionConfig,
    get_default_mixed_precision_config,
    get_default_mixed_precision_config_set,
    get_woq_tuning_config,
    DynamicQuantConfig,
    get_default_dynamic_config,
)

from neural_compressor.torch.quantization.autotune import (
    autotune,
    TuningConfig,
    get_all_config_set,
    get_rtn_double_quant_config_set,
)

### Quantization Function Registration ###
import neural_compressor.torch.quantization.algorithm_entry
from neural_compressor.torch.quantization.save_load_entry import save, load
