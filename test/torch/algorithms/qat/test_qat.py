# -*- coding: utf-8 -*-
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
"""QAT smoke tests on the new fake-quant mechanism (no auto_round needed)."""

import warnings

import torch
import torch.nn as nn

from neural_compressor.torch.quantization import FakeQuantConfig, QuantRule, prepare_qat
from neural_compressor.torch.quantization.fake_quant import QuantLinear, TensorQuantizer


class TinyModel(nn.Module):
    """Two linears plus an excluded lm_head."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.lm_head = nn.Linear(64, 2)

    def forward(self, x):
        return self.lm_head(self.fc1(x))


def _qat_config():
    return FakeQuantConfig(rules=[QuantRule(module=r".*fc1$", weight="mxfp8", input="mxfp8")])


def test_prepare_inserts_quantizers():
    model = TinyModel()
    prepare_qat(model, _qat_config())
    assert isinstance(model.fc1, QuantLinear)
    assert not isinstance(model.lm_head, QuantLinear)
    replaced = sum(isinstance(module, TensorQuantizer) for _, module in model.named_modules())
    assert replaced == 3  # weight/input/output on fc1
    assert model.fc1.weight_quantizer.enabled
    assert not model.fc1.output_quantizer.enabled


def test_qat_train_step():
    torch.manual_seed(20)
    model = TinyModel()
    prepare_qat(model, _qat_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    output = model(torch.randn(2, 32))
    loss = output.mean()
    loss.backward()
    for _, param in model.named_parameters():
        assert param.grad is not None
    optimizer.step()


def test_prepare_qat_default_legacy_mapping_warns():
    model = TinyModel()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prepare_qat(model)
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
    assert model.fc1.weight_quantizer.format.name == "mxfp8"
