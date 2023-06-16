import numpy as np
import unittest
import os
import yaml
import importlib
import shutil
import torch
from neural_compressor.adaptor.torch_utils.weight_only import quant_model_weight_only


class TestWeightOnlyQuant(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, 2, 2)
                self.act = torch.nn.ReLU6()
                self.conv2 = torch.nn.Conv2d(4, 10, 3, 3)

            def forward(self, x):
                out = self.conv1(x)
                out = self.act(out)
                out = self.conv2(out) + x
                return out

        self.model = Model()

    @classmethod
    def tearDownClass(self):
        pass

    def test_conv(self):
        quant_model_weight_only(self.model, num_bits=3, group_size=-1)
    #
    # def test_conv_memory_format(self):
    #     import  copy
    #     model = copy.deepcopy(self.model)
    #     model = model.to(memory_format=torch.channels_last)
    #     quant_model_weight_only(self.model, num_bits=3, group_size=-1)
