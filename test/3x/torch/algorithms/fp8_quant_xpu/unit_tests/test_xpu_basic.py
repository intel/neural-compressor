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

import os

import pytest
import torch
import intel_perfcore_for_pytorch as ipex


from neural_compressor.torch.algorithms.fp8_quant._core.quant_dequant import QuantInput, QuantDequant, DequantOutput
from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import ScaleFormat, Fp8cfg, set_hqt_config
from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import PatchedLinear, PatchedMatmul
from neural_compressor.torch.algorithms.fp8_quant.model_configs import ModuleExtraConfig, ModuleConfig
from neural_compressor.torch.algorithms.fp8_quant._core.quantized_func_wrappers import (
    init_quantized_func_wrapper_factory,
    clear_quantized_func_wrapper_factory)

from neural_compressor.torch.quantization import (
    FP8Config,
    convert,
)

class Matmul(torch.nn.Module):

    def __init__(self, dim=16):
        super().__init__()

    def forward(self, input, other):
        return torch.matmul(input, other)


class MyModelMatmul(torch.nn.Module):

    def __init__(self, dim=16):
        super().__init__()
        self.my_matmul = Matmul()

    def forward(self, input, other):
        return self.my_matmul(input, other)


class MyModel(torch.nn.Module):

    def __init__(self, dim=16):
        super().__init__()
        self.my_linear = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, input):
        return self.my_linear(input)


@pytest.mark.xfail(reason="PYTORCHDGQ-6840 - enable once low-precision casting custom XPU ops are supported")
def test_xpu_basic_mamtul():
    # test convert flow and quantized func
    my_model = MyModelMatmul()
    my_model = my_model.to("xpu")
    print(my_model)
    qconfig = FP8Config(fp8_config="E4M3", scale_method="unit_scale", scale_format="const")
    my_model = convert(my_model, qconfig)
    print(my_model)
    verified_matmul_quantized_func_wrapper = False
    verified_quant_input_quantized_func_wrapper = False
    for _, mod in my_model.named_modules():
        # verify correct custom ops were assigned
        if isinstance(mod, PatchedMatmul):
            assert mod.matmul_fp8._quantized_func_ == torch.ops.torch_ipex.fp8_gemm
            verified_matmul_quantized_func_wrapper = True
        elif isinstance(mod, QuantInput):
            assert mod.cast_to_op._quantized_func_ == torch.ops.torch_ipex.cast_to_fp8
            verified_quant_input_quantized_func_wrapper = True
    # verify that we actually did the checks
    assert verified_matmul_quantized_func_wrapper and verified_quant_input_quantized_func_wrapper

pytest.mark.xfail(reason="PYTORCHDGQ-6840 - enable once low-precision casting custom XPU ops are supported")
def test_xpu_quantized_func_wrapper():
    # test for verifying xpu quantized wrapper logic
    my_model = MyModel()

    # set config object in the model
    config = FP8Config(fp8_config="E4M3", scale_method="unit_scale", scale_format="const", mode="QUANTIZE")
    fp8_cfg = Fp8cfg.parse(config.to_dict())
    set_hqt_config(my_model, fp8_cfg)
    set_hqt_config(my_model.my_linear, fp8_cfg)
    # set scale tensors
    scale_in = torch.Tensor([0.5])
    scale_weight = torch.Tensor([0.5])
    scale_out = torch.Tensor([4.0])
    lp_dtype = torch.float8_e4m3fn
    hp_dtype = torch.bfloat16
    scale_format = ScaleFormat.SCALAR

    init_quantized_func_wrapper_factory()

    quant_input = QuantInput(scale_in, lp_dtype, hp_dtype, scale_format)
    quant_weight = QuantInput(scale_weight, lp_dtype, hp_dtype, scale_format)
    quant_output = DequantOutput(scale_out, lp_dtype, hp_dtype, scale_format)

    assert quant_input.cast_to_op._quantized_func_ == torch.ops.torch_ipex.cast_to_fp8
    assert quant_weight.cast_to_op._quantized_func_ == torch.ops.torch_ipex.cast_to_fp8
    assert quant_output.cast_from_op._quantized_func_ == torch.ops.torch_ipex.cast_from_fp8

    # similar logic to scale calculation
    scales_mod_config = ModuleConfig([scale_in], [scale_out], {"weight": scale_weight})
    module_ex_config = ModuleExtraConfig([quant_input],
                                         [quant_output],
                                         [quant_weight],
                                         scales_mod_config,
                                         {"lp_dtype": lp_dtype,
                                          "hp_dtype": hp_dtype})

    patched_my_linear = PatchedLinear(my_model.my_linear, my_model, module_ex_config)
    assert patched_my_linear.matmul_fp8._quantized_func_ == torch.ops.torch_ipex.fp8_gemm

    clear_quantized_func_wrapper_factory()

def test_xpu_basic_mamtul_qdq():
    # test convert flow and quantized func
    my_model = MyModelMatmul()
    my_model = my_model.to("xpu")
    print(my_model)
    qconfig = FP8Config(fp8_config="E4M3", scale_method="unit_scale", scale_format="const", use_qdq=True)
    my_model = convert(my_model, qconfig)
    print(my_model)
    verified_matmul_quantized_func_wrapper = False
    verified_quant_input_quantized_func_wrapper = False
    for _, mod in my_model.named_modules():
        # verify correct custom ops were assigned
        if isinstance(mod, PatchedMatmul):
            # There is no quantized function wrapper for qdq.
            verified_matmul_quantized_func_wrapper = True
        elif isinstance(mod, QuantDequant):
            assert mod.quantize_op._quantized_func_ == torch.ops.quantized_decomposed.quantize_per_tensor
            assert mod.dequantize_op._quantized_func_ == torch.ops.quantized_decomposed.dequantize_per_tensor
            verified_quant_input_quantized_func_wrapper = True
    # verify that we actually did the checks
    assert verified_matmul_quantized_func_wrapper and verified_quant_input_quantized_func_wrapper

def test_xpu_quantized_func_wrapper_qdq():
    # test for verifying xpu quantized wrapper logic
    my_model = MyModel()

    # set config object in the model
    config = FP8Config(fp8_config="E4M3", scale_method="unit_scale", scale_format="const", mode="QUANTIZE", use_qdq=True)
    fp8_cfg = Fp8cfg.parse(config.to_dict())
    set_hqt_config(my_model, fp8_cfg)
    set_hqt_config(my_model.my_linear, fp8_cfg)
    # set scale tensors
    scale_in = torch.Tensor([0.5])
    scale_weight = torch.Tensor([0.5])
    scale_weight_inv = torch.Tensor([2.0])
    scale_out = torch.Tensor([4.0])
    lp_dtype = torch.float8_e4m3fn
    hp_dtype = torch.bfloat16
    scale_format = ScaleFormat.SCALAR

    init_quantized_func_wrapper_factory()

    quant_input = QuantInput(scale_in, lp_dtype, hp_dtype, scale_format, use_qdq=True)
    quant_output = DequantOutput(scale_out, lp_dtype, hp_dtype, scale_format, use_qdq=True)

    assert quant_input.quantize_op._quantized_func_ == torch.ops.quantized_decomposed.quantize_per_tensor
    assert quant_output.dequantize_op._quantized_func_ == torch.ops.quantized_decomposed.dequantize_per_tensor

    weight_config = [
        QuantInput(scale_weight_inv, lp_dtype, hp_dtype, scale_format=scale_format, use_qdq=True),
        DequantOutput(scale_weight, lp_dtype, hp_dtype, scale_format=scale_format, use_qdq=True),
    ]

    assert weight_config[0].quantize_op._quantized_func_ == torch.ops.quantized_decomposed.quantize_per_tensor
    assert weight_config[1].dequantize_op._quantized_func_ == torch.ops.quantized_decomposed.dequantize_per_tensor

    # similar logic to scale calculation
    scales_mod_config = ModuleConfig([scale_in], [scale_out], {"weight": scale_weight})
    module_ex_config = ModuleExtraConfig([quant_input],
                                         [quant_output],
                                         {"weight": weight_config},
                                         scales_mod_config,
                                         {"lp_dtype": lp_dtype,
                                          "hp_dtype": hp_dtype})

    patched_my_linear = PatchedLinear(my_model.my_linear, my_model, module_ex_config)
    # There is no quantized function wrapper for qdq.

    clear_quantized_func_wrapper_factory()
