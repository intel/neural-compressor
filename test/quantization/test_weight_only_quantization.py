import unittest
import copy
import torch
from neural_compressor.adaptor.torch_utils.weight_only import rtn_quantize


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
        fp32_model = copy.deepcopy(self.model)
        model1 = rtn_quantize(fp32_model, num_bits=3, group_size=-1)
        w_layers_config = {
            # 'op_name': (bit, group_size, sheme)
            'conv1': (8, 128, 'sym'),
            'conv2': (4, 32, 'asym')
        }
        model2 = rtn_quantize(fp32_model, num_bits=3, group_size=-1, w_layers_config=w_layers_config)


if __name__ == "__main__":
    unittest.main()
