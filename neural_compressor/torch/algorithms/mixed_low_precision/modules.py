# Copyright (c) 2025 Intel Corporation
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

import functools
import math
from abc import abstractmethod

import numpy as np
import torch
from torch.autograd import Function
from torch.nn import functional as F

from neural_compressor.torch.utils import accelerator, logger, set_module

from ..weight_only.modules import HPUWeightOnlyLinear

cast_to_fp8_fcn = lambda x, dtype, scale_inv=None: torch.ops.hpu.cast_to_fp8_v2(x, scale_inv, False, False, dtype)[0]


class HPUMixedPrecisionLinear(HPUWeightOnlyLinear):
    """Weight and Activations quant (W4A8 gptq) Linear for HPU device."""

    def __init__(
        self,
        in_features,
        out_features,
        bias,
        **kwargs,
    ):
        """Init the HPUMixedPrecisionLinear object."""
        super(HPUMixedPrecisionLinear, self).__init__(in_features, out_features, bias=bias)

    def forward(self, input):
        """The forward function of HPUMixedPrecisionLinear."""
        input_dtype = input.dtype
        output_shape = input.shape[:-1] + (self.out_features,)
        scales = self.scales
        qweight = self.qweight
        zeros = self.qzeros
        weight = torch.ops.hpu.convert_from_uint4(qweight, scales, zeros, torch.float8_e4m3fn)
        output = self.matmul_internal(input, weight)
        output = output.to(dtype=input_dtype).reshape(
            output_shape
        )  # A cast is needed here as for some reason the vecquant2matmul_faster_old still allocate a float32 output.
        output = output + self.bias if self.bias is not None else output
        return output

    def forward_mesaure(self, input):
        """The measure forward for w4a8 scheme.

        Measuring is done in bf16 precision matmul for
        generation of fp8 scales for the activations.
        """
        input_dtype = input.dtype
        output_shape = input.shape[:-1] + (self.out_features,)
        scales = self.scales
        scale_bf16_to_fp8 = self.scale_bf16_to_fp8
        qweight = self.qweight
        zeros = self.qzeros
        weight = torch.ops.hpu.convert_from_uint4(qweight, scales, zeros, torch.float8_e4m3fn)
        weight = weight.to(input_dtype) * scale_bf16_to_fp8
        output = self.matmul_internal(input, weight)
        output = output.to(dtype=input_dtype).reshape(output_shape)
        output = output + self.bias if self.bias is not None else output
        return output

    def forward_inference(self, input):
        """Performs the inference-only forward pass.

        The `forward_inference` method differs from `forward` in two key ways:
            1) The `scale_other` parameter in `matmul_internal` is fixed and pre-absorbed into the scales.
            2) The weight is preconverted to FP8 during initialization, whereas in `forward`, it is converted to FP8 on the fly.
        """
        input_dtype = input.dtype
        output_shape = input.shape[:-1] + (self.out_features,)
        scales = self.scales
        qweight = self.qweight
        zeros = self.qzeros

        weight = torch.ops.hpu.convert_from_uint4(
            qweight, scales, zeros, torch.float8_e4m3fn
        )  # todo: div scales in init
        output = self.matmul_internal(input, weight)
        output = output.to(dtype=input_dtype).reshape(output_shape)
        output = output + self.bias if self.bias is not None else output
        return output

    @staticmethod
    def convert_from_weight_only(obj):
        bias = obj.bias is not None
        new_self = HPUMixedPrecisionLinear(obj.in_features, obj.out_features, bias)
        for attr, value in vars(obj).items():
            setattr(new_self, attr, value)
        new_self.matmul_internal.no_input_quant = True  # flag for 8bit input, which shouldn't be quantized in matmul
        if hasattr(new_self, "_original_forward"):
            new_self.forward = new_self._original_forward
        return new_self

    def prepare_from_weight_only(obj):
        # prepare class for measurement
        bias = obj.bias is not None
        new_self = HPUMixedPrecisionLinear(obj.in_features, obj.out_features, bias)
        for attr, value in vars(obj).items():
            setattr(new_self, attr, value)
        new_self._original_forward = new_self.forward
        new_self.forward = new_self.forward_mesaure
        return new_self

    def post_process_for_inference(self):
        """Post process for inference."""
        from neural_compressor.torch.algorithms.fp8_quant._core.quant_dequant import QuantDequantNone, QuantInput
        from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import PatchedMatmul

        self = self.to("hpu")
        module = self
        patched_matmul: PatchedMatmul = module.matmul_internal
        w_bf16_to_fp8_scale = module.w_bf16_to_fp8_scale
        assert w_bf16_to_fp8_scale.numel() == 1, f"Only support per-tensor scale, but got {w_bf16_to_fp8_scale}"
        w_scales = module.scales
        w_fp8_to_int4_scale = w_scales / w_bf16_to_fp8_scale
        module.scales.data.copy_(w_fp8_to_int4_scale)
        act_scales = module.act_scales
        patched_matmul.scale_input = act_scales.item()
        patched_matmul.scale_other = w_bf16_to_fp8_scale.item()
        input_quantizer: QuantInput = patched_matmul.quant_input_0
        input_quantizer.scale_inv = 1.0 / act_scales.item()
        self.forward = self.forward_inference


def assign_scales_to_patched_matmul_(model):
    for _, module in model.named_modules():
        if isinstance(module, HPUMixedPrecisionLinear):
            logger.debug(f"Assign scales to PatchedMatmul in {module}")
            module.post_process_for_inference()


def _create_fp8_config_from_quant_config(model):
    from neural_compressor.torch.quantization import FP8Config

    blocklist = {"types": [], "names": []}
    allowlist = {"types": ["Matmul"], "names": []}
    fp8_config = FP8Config(mode="LOAD", allowlist=allowlist, blocklist=blocklist, scale_format="scalar")
    return fp8_config


@functools.lru_cache(maxsize=None)
def init_quantized_func_wrapper_factory_once():
    from neural_compressor.torch.algorithms.fp8_quant._core.quantized_func_wrappers import (
        init_quantized_func_wrapper_factory,
    )

    init_quantized_func_wrapper_factory()


def patch_model_for_inference_(model):
    init_quantized_func_wrapper_factory_once()
    from neural_compressor.torch.algorithms.fp8_quant import prep_model as fp8_quant_prep_model_

    fp8_config = _create_fp8_config_from_quant_config(model)
    fp8_config.save_temp_json_file()
    fp8_quant_prep_model_(model, fp8_config.json_file)
    assign_scales_to_patched_matmul_(model)


def replace_hpu_woq_with_hpu_mixed_precision_linear(woq_model):
    for name, module in woq_model.named_modules():
        if isinstance(module, HPUWeightOnlyLinear):
            new_module = HPUMixedPrecisionLinear.convert_from_weight_only(module)
            set_module(woq_model, name, new_module)
    # If we already have the calibration results, we can directly replace the matmul with PatchedMatmul.
    patch_model_for_inference_(woq_model)
    return woq_model
