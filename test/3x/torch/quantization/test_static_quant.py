import copy
import unittest

import intel_extension_for_pytorch as ipex
import torch

from neural_compressor.torch.quantization import StaticQuantConfig, get_default_static_config, quantize
from neural_compressor.torch.utils import get_model_info, logger


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


def run_fn(model):
    model(torch.rand((1, 30)))
    model(torch.rand((1, 30)))


class TestStaticQuant(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.fp32_model = build_simple_torch_model()
        self.input = torch.randn(1, 30)

    @classmethod
    def tearDownClass(self):
        pass

    def setUp(self):
        logger.info(f"Running TestStaticQuant test: {self.id()}")

    def test_quantize_rtn_from_dict_default(self):
        fp32_model = copy.deepcopy(self.fp32_model)
        quant_config = get_default_static_config()
        example_inputs = self.input
        out1 = fp32_model(example_inputs)
        qmodel = quantize(fp32_model, quant_config=quant_config, run_fn=run_fn, example_inputs=example_inputs)
        out2 = qmodel(example_inputs)
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=1e-05))
        self.assertIsNotNone(qmodel)


if __name__ == "__main__":
    unittest.main()
