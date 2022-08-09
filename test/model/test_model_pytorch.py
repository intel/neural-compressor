import torch
import torch.nn as nn
import torch.nn.quantized as nnq
from torch.quantization import QuantStub, DeQuantStub
import torchvision
import unittest
import os
from neural_compressor.adaptor import FRAMEWORKS
from neural_compressor.model import MODELS
from neural_compressor.adaptor.pytorch import  PyTorchVersionMode
import neural_compressor.adaptor.pytorch as nc_torch
from neural_compressor.experimental import Quantization, common
from neural_compressor.utils.pytorch import load
from neural_compressor.utils.utility import recover
import shutil
import copy
import numpy as np
import yaml

try:
    import intel_pytorch_extension as ipex
    TEST_IPEX = True
except:
    TEST_IPEX = False

PT_VERSION = nc_torch.get_torch_version()
if PT_VERSION >=  PyTorchVersionMode.PT18.value:
    FX_MODE = True
else:
    FX_MODE = False


class TestPytorchModel(unittest.TestCase):
    framework = "pytorch"
    model = torchvision.models.quantization.resnet18()
    lpot_model = MODELS['pytorch'](model)

    def test_get_all_weight_name(self):
        assert len(list(self.lpot_model.get_all_weight_names())) == 62

    def test_get_weight(self):
        for name, param in self.model.named_parameters():
            if name == "layer4.1.conv2.weight":
                param.data.fill_(0.0)
            if name == "fc.bias":
                param.data.fill_(0.1)
        assert int(torch.sum(self.lpot_model.get_weight("layer4.1.conv2.weight"))) == 0
        assert torch.allclose(
            torch.sum(
                torch.tensor(self.lpot_model.get_weight("fc.bias"))),
            torch.tensor(100.))

    def test_get_input(self):
        model = MODELS['pytorch'](torchvision.models.quantization.resnet18())
        model.model.eval().fuse_model()
        model.register_forward_pre_hook()
        rand_input = torch.rand(100, 3, 256, 256).float()
        model.model(rand_input)
        assert torch.equal(model.get_inputs('x'), rand_input)
        model.remove_hooks()

    def test_update_weights(self):
        self.lpot_model.update_weights('fc.bias', torch.zeros([1000]))
        assert int(torch.sum(self.lpot_model.get_weight("fc.bias"))) == 0

    def test_gradient(self):
        with self.assertRaises(AssertionError):
            self.lpot_model.get_gradient('fc.bias')

        shape = None
        for name, tensor in self.lpot_model._model.named_parameters():
            if name == 'fc.bias':
                shape = tensor.shape
                tensor.grad = torch.randn(shape)
                break
        new_grad = torch.zeros(shape)
        self.lpot_model.update_gradient('fc.bias', new_grad)
        assert torch.equal(torch.tensor(self.lpot_model.get_gradient('fc.bias')), torch.zeros(shape))

        rand_input = torch.rand(100, 3, 256, 256).float()
        rand_input.grad = torch.ones_like(rand_input)
        assert torch.equal(torch.tensor(self.lpot_model.get_gradient(rand_input)),
                           torch.ones_like(rand_input))

    def test_report_sparsity(self):
        df, total_sparsity = self.lpot_model.report_sparsity()
        self.assertTrue(total_sparsity > 0)
        self.assertTrue(len(df) == 22)


if __name__ == "__main__":
    unittest.main()
