import unittest
from copy import deepcopy

from neural_compressor.utils.logger import Logger

logger = Logger().get_logger()
import torch


def build_simple_torch_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = torch.nn.Linear(30, 50)
            self.fc2 = torch.nn.Linear(50, 30)
            self.fc3 = torch.nn.Linear(30, 5)

        def forward(self, x):
            out = self.fc1(x)
            out = self.fc2(out)
            out = self.fc3(out)
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

    def setUp(self):
        # print the test name
        logger.info(f"\nRunning test: {self._testMethodName}")

    def test_quantize_rtn_from_dict_default(self):
        logger.info("test_quantize_rtn_from_dict_default")
        from neural_compressor.torch import get_default_rtn_config, quantize

        fp32_model = build_simple_torch_model()
        qmodel = quantize(fp32_model, quant_config=get_default_rtn_config())
        self.assertIsNotNone(qmodel)

    def test_quantize_rtn_from_dict_beginner(self):
        from neural_compressor.torch import quantize

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
        from neural_compressor.torch import RTNWeightQuantConfig, quantize

        quant_config = RTNWeightQuantConfig(weight_bits=4, weight_dtype="nf4", weight_group_size=32)
        fp32_model = build_simple_torch_model()
        qmodel = quantize(fp32_model, quant_config)
        self.assertIsNotNone(qmodel)

    def test_quantize_rtn_from_dict_advance(self):
        from neural_compressor.torch import quantize

        fp32_model = build_simple_torch_model()
        quant_config = {
            "rtn_weight_only_quant": {
                "global": {
                    "weight_dtype": "nf4",
                    "weight_bits": 4,
                    "weight_group_size": 32,
                },
                "operator_name": {
                    "fc1": {
                        "weight_dtype": "int8",
                        "weight_bits": 4,
                    }
                },
            }
        }
        qmodel = quantize(fp32_model, quant_config)
        self.assertIsNotNone(qmodel)

    def test_quantize_rtn_from_class_advance(self):
        from neural_compressor.torch import RTNWeightQuantConfig, quantize

        quant_config = RTNWeightQuantConfig(weight_bits=4, weight_dtype="nf4")
        # # set operator type
        # linear_config = RTNWeightQuantConfig(weight_bits=6, weight_dtype="nf4")
        # quant_config._set_operator_type(torch.nn.Linear, linear_config)
        # set operator instance
        fc1_config = RTNWeightQuantConfig(weight_bits=4, weight_dtype="int8")
        quant_config.set_operator_name("model.fc1", fc1_config)

        # get model and quantize
        fp32_model = build_simple_torch_model()
        qmodel = quantize(fp32_model, quant_config)
        self.assertIsNotNone(qmodel)


if __name__ == "__main__":
    unittest.main()
