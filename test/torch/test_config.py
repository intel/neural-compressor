import unittest
from copy import deepcopy

import torch


def build_simple_torch_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(4, 4, 2)
            self.act = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv1d(4, 4, 2)
            self.linear = torch.nn.Conv2d(32, 3)

        def forward(self, x):
            out = self.conv1(x)
            out = self.act(out)
            out = self.conv2(out)
            out = out.view(1, -1)
            out = self.linear(out)
            return out

    model = Model()
    return model


class TestQuantizationConfig(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.fp32_model = build_simple_torch_model()

    @classmethod
    def tearDownClass(self):
        pass

    def test_quantize_rtn_from_dict_beginner(self):
        from neural_compressor.torch.quantization import quantize

        quant_config = {
            "rtn_weight_only_quant": {
                "weight_dtype": "nf4",
                "weight_bits": 4,
                "weight_group_size": 32,
            },
        }
        fp32_model = build_simple_torch_model()
        qmodel = quantize(fp32_model, quant_config)
        self.assertIsNotNone(qmodel)

    def test_quantize_rtn_from_class_beginner(self):
        from neural_compressor.torch.quantization import quantize
        from neural_compressor.torch.quantization.config import RTNWeightQuantConfig

        quant_config = RTNWeightQuantConfig(weight_bits=4, weight_dtype="nf4", weight_group_size=32)
        fp32_model = build_simple_torch_model()
        qmodel = quantize(fp32_model, quant_config)
        self.assertIsNotNone(qmodel)

    def test_quantize_rtn_from_dict_advance(self):
        from neural_compressor.torch.quantization import quantize

        fp32_model = build_simple_torch_model()
        quant_config = {
            "rtn_weight_only_quant": {
                "global": {
                    "weight_dtype": "nf4",
                    "weight_bits": 4,
                    "weight_group_size": 32,
                },
                "operator_type": {
                    "Conv2d": {
                        "weight_dtype": "nf4",
                        "weight_bits": 6,
                    }
                },
                "operator_name": {
                    "model.conv1": {
                        "weight_dtype": "int8",
                        "weight_bits": 4,
                    }
                },
            }
        }
        qmodel = quantize(fp32_model, quant_config)
        self.assertIsNotNone(qmodel)

    def test_quantize_rtn_from_class_advance(self):
        from neural_compressor.torch.quantization import quantize
        from neural_compressor.torch.quantization.config import RTNWeightQuantConfig

        quant_config = RTNWeightQuantConfig(weight_bits=4, weight_dtype="nf4")
        # set global config
        global_config = RTNWeightQuantConfig(weight_bits=4, weight_dtype="nf4")
        quant_config.set_global(global_config)
        # set operator type
        conv_config = RTNWeightQuantConfig(weight_bits=6, weight_dtype="nf4")
        quant_config.set_operator_type(torch.nn.Conv2d, conv_config)
        # set operator instance
        conv1_config = RTNWeightQuantConfig(weight_bits=4, weight_dtype="int8")
        quant_config.set_operator_name("model.conv1", conv1_config)

        # get model and quantize
        fp32_model = build_simple_torch_model()
        qmodel = quantize(fp32_model, quant_config)
        self.assertIsNotNone(qmodel)


if __name__ == "__main__":
    unittest.main()
