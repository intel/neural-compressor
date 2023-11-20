import unittest

import torch

from neural_compressor.common.logger import Logger

logger = Logger().get_logger()


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


class TestRTNQuant(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def setUp(self):
        # print the test name
        logger.info(f"Running TestRTNQuant test: {self.id()}")

    def _apply_rtn(self, quant_config):
        logger.info(f"Test RTN with config {quant_config}")
        from neural_compressor.torch import quantize

        fp32_model = build_simple_torch_model()
        qmodel = quantize(fp32_model, quant_config)
        self.assertIsNotNone(qmodel)
        return qmodel

    def test_rtn(self):
        from neural_compressor.torch import RTNWeightQuantConfig

        # some tests were skipped to accelerate the CI
        rnt_options = {
            "weight_dtype": ["int", "int8", "nf4", "fp4_e2m1_bnb"],
            "weight_bits": [4, 1, 8],
            "weight_group_size": [32, -1, 1024],
            "weight_sym": [True, False],
            "act_dtype": ["fp32"],
            "enable_full_range": [False, True],
            "enable_mse_search": [False, True],
            "group_dim": [1, 0],
        }
        from itertools import product

        keys = RTNWeightQuantConfig.params_list[:-1]
        for value in product(*rnt_options.values()):
            d = dict(zip(keys, value))
            if (d["weight_dtype"] != "int" and d["weight_bits"] != 4) or (
                d["enable_full_range"] and d["enable_mse_search"]
            ):
                continue
            quant_config = RTNWeightQuantConfig.from_dict(d)
            self._apply_rtn(quant_config)

    def test_rtn_return_type(self):
        from neural_compressor.torch import RTNWeightQuantConfig

        for return_int in [True, False]:
            quant_config = RTNWeightQuantConfig(return_int=return_int)
            qmodel = self._apply_rtn(quant_config)

    def test_rtn_recover(self):
        from neural_compressor.torch import RTNWeightQuantConfig

        quant_config = RTNWeightQuantConfig(return_int=True)
        qmodel = self._apply_rtn(quant_config)
        recovered_fc1 = qmodel.fc1.recover()
        self.assertIsNotNone(recovered_fc1)


if __name__ == "__main__":
    unittest.main()
