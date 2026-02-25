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
import shutil

import pytest
import torch

from neural_compressor.torch.algorithms.fp8_quant._core.quant_dequant import QuantDequant
from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import PatchedEmbeddingBag, PatchedLinear
from neural_compressor.torch.quantization import (
    FP8Config,
    convert,
    prepare,
)


class MyModel(torch.nn.Module):

    def __init__(self, dim=16):
        super().__init__()
        self.embed = torch.nn.EmbeddingBag(1, dim)
        self.my_linear = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, input, offset):
        out = self.embed(input, offset)
        return self.my_linear(out)


@pytest.mark.parametrize("scale_format", ["const", "scalar"])
@pytest.mark.parametrize("scale_method", ["unit_scale", "MAXABS_ARBITRARY", "ACT_MAXABS_POW2_WEIGHTS_PCS_MAXABS_POW2"])
def test_cpu_basic_flow(scale_format, scale_method):
    # test convert flow
    model = MyModel()
    model = model.to("cpu")
    qconfig = FP8Config(fp8_config="E4M3", scale_method=scale_method, scale_format=scale_format, use_qdq=True)
    if scale_method in ["unit_scale"]:
        model = convert(model, qconfig)
    else:
        model = prepare(model, qconfig)
        model(torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long))
        model = convert(model)
    for _, mod in model.named_modules():
        if isinstance(mod, PatchedLinear):
            assert isinstance(mod.quant_input, QuantDequant), "Linear is not quantized with QDQ mode"
        if isinstance(mod, PatchedEmbeddingBag):
            assert not hasattr(mod, "quant_input"), "EmbeddingBag is not weight-only quantized"
    shutil.rmtree("hqt_output", ignore_errors=True)


def test_cpu_quant_flow():
    my_model = MyModel()
    my_model = my_model.to("cpu")
    qconfig = FP8Config(fp8_config="E4M3", scale_method="unit_scale", scale_format="const")
    with pytest.raises(ValueError):
        my_model = convert(my_model, qconfig)
