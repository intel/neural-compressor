
import numpy as np
import torch
import torchvision
import unittest
import os
from ilit.adaptor import FRAMEWORKS


class TestAdaptorPytorch(unittest.TestCase):
    framework_specific_info = {'device': "cpu",
                               'approach': "post_training_static_quant",
                               'random_seed': 1234}
    framework = "pytorch"
    adaptor = FRAMEWORKS[framework](framework_specific_info)
    model = torchvision.models.resnet18()

    def test_get_all_weight_name(self):
        assert len(list(self.adaptor.get_all_weight_names(self.model))) == 62

    def test_get_weight(self):
        for name, param in self.model.named_parameters():
            if name == "layer4.1.conv2.weight":
                param.data.fill_(0.0)
            if name == "fc.bias":
                param.data.fill_(0.1)
        assert int(torch.sum(self.adaptor.get_weight(self.model, "layer4.1.conv2.weight"))) == 0
        assert torch.allclose(
            torch.sum(
                self.adaptor.get_weight(
                    self.model,
                    "fc.bias")),
            torch.tensor(100.))

    def test_update_weights(self):
        model = self.adaptor.update_weights(self.model, "fc.bias", torch.zeros([1000]))
        assert int(torch.sum(self.adaptor.get_weight(model, "fc.bias"))) == 0


if __name__ == "__main__":
    unittest.main()
