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

from neural_compressor.torch.quantization.quantize import (
    convert,
    finalize_calibration,
    prepare,
    prepare_qat,
    quantize,
)
from neural_compressor.torch.quantization.config import (
    AutoRoundConfig,
    AWQConfig,
    FP8Config,
    GPTQConfig,
    HQQConfig,
    HybridGPTQConfig,
    INT8StaticQuantConfig,
    MixedPrecisionConfig,
    MXQuantConfig,
    RTNConfig,
    SmoothQuantConfig,
    StaticQuantConfig,
    TEQConfig,
    get_default_AutoRound_config,
    get_default_awq_config,
    get_default_double_quant_config,
    get_default_fp8_config,
    get_default_fp8_config_set,
    get_default_gptq_config,
    get_default_hqq_config,
    get_default_mixed_precision_config,
    get_default_mixed_precision_config_set,
    get_default_mx_config,
    get_default_rtn_config,
    get_default_sq_config,
    get_default_static_config,
    get_default_teq_config,
    get_woq_tuning_config,
)
from neural_compressor.torch.quantization.autotune import (
    TuningConfig,
    autotune,
    get_all_config_set,
    get_rtn_double_quant_config_set,
)
from neural_compressor.torch.algorithms.fake_quant import (
    FP8_DEFAULT_CFG,
    MXFP4_DEFAULT_CFG,
    MXFP4_FP8_HYBRID_CFG,
    MXFP8_DEFAULT_CFG,
    NVFP4_DEFAULT_CFG,
    NVFP4_FP8_HYBRID_CFG,
    FakeQuantConfig,
    ModuleQuantConfig,
    QuantConversionReport,
    QuantFormat,
    QuantizerConfig,
    QuantLinear,
    QuantLoraLinear,
    QuantSchemeRule,
    TensorQuantizer,
    build_quant_plan,
    calibrate,
    calibrate_static_amax,
    convert_peft_lora_to_quant_lora,
    convert_to_quantized_model,
    disable_fake_quant,
    get_default_fake_quant_config,
    get_format,
    list_formats,
    load_quantizer_state,
    match_quant_config,
    merge_lora_into_quant_linear,
    register_peft_quant_modules,
    replace_quant_modules,
    save_quantizer_state,
)


def __getattr__(name: str):
    if name in ("QATTrainer", "QuantizationTrainingArguments", "save_fsdp2_full_model", "save_fsdp2_peft_adapter"):
        from neural_compressor.torch.algorithms.fake_quant import trainer

        return getattr(trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


### Quantization Function Registration ###
import neural_compressor.torch.quantization.algorithm_entry
from neural_compressor.torch.quantization.save_load_entry import load, save
