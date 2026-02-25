import math
import types

import pytest
import torch
import torch.nn as nn

# Skip the whole module if auto_round (needed for get_quant_func inside TensorQuantizer) is not available
auto_round = pytest.importorskip("auto_round")

from neural_compressor.torch.algorithms.qat.tensor_quantizer import TensorQuantizer
from neural_compressor.torch.quantization.quantize import prepare_qat


def setup_seed(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class TinyModel(nn.Module):
    """Simple hierarchical model for recursive replacement tests."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.lm_head = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        return self.lm_head(x)


def test_replace_quant_layer():
    """Check the inserted quant linear."""
    model = TinyModel()

    prepare_qat(model)

    replaced_modules = sum(isinstance(m, TensorQuantizer) for _, m in model.named_modules())

    assert replaced_modules == 3


def test_train():
    """QAT test."""
    setup_seed(20)

    model = TinyModel()
    prepare_qat(model)

    inp = torch.randn([2, 32])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(inp)
        loss = output.mean()

    optimizer.zero_grad()
    loss.backward()

    # check the grad
    for name, param in model.named_parameters():
        assert param.grad is not None
    optimizer.step()


def test_train_mxfp4():
    """QAT test."""
    setup_seed(20)

    model = TinyModel()
    mappings = {torch.nn.Linear: "MXFP4"}
    prepare_qat(model, mappings)

    inp = torch.randn([2, 32])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(inp)
        loss = output.mean()

    optimizer.zero_grad()
    loss.backward()

    # check the grad
    for name, param in model.named_parameters():
        assert param.grad is not None
    optimizer.step()
