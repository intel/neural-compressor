import os
import shutil
import unittest

from optimum.exporters.onnx import main_export

from neural_compressor.common.logger import Logger

logger = Logger().get_logger()

def find_onnx_file(folder_path):
    # return first .onnx file path in folder_path
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".onnx"):
                return os.path.join(root, file)
    return None

class TestRTNQuant(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        main_export(
            "hf-internal-testing/tiny-random-gptj",
            output="gptj",
        )
        self.gptj = find_onnx_file("./gptj")

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("gptj", ignore_errors=True)

    def setUp(self):
        # print the test name
        logger.info(f"Running ONNXRT TestRTNQuant test: {self.id()}")

    def _count_woq_matmul(self, q_model, bits=4, group_size=32):
        op_names = [
            i.name
            for i in q_model.graph.node
            if i.op_type.startswith("MatMul") and i.input[1].endswith("_Q{}G{}".format(bits, group_size))
        ]
        return len(op_names)

    def _apply_rtn(self, quant_config):
        logger.info(f"Test RTN with config {quant_config}")
        from neural_compressor.onnxrt.quantization.quantize import _quantize

        fp32_model = self.gptj
        qmodel = _quantize(fp32_model, quant_config)
        self.assertIsNotNone(qmodel)
        return qmodel

    def test_rtn(self):
        from neural_compressor.onnxrt import RTNConfig

        # some tests were skipped to accelerate the CI
        # TODO: check params combination.
        # TODO: Add number check for group_size.
        rtn_options = {
            "weight_dtype": ["int"],
            "weight_bits": [4, 3, 8],
            "weight_group_size": [32],
            "weight_sym": [True, False],
            "act_dtype": ["fp32"],
        }
        from itertools import product

        keys = RTNConfig.params_list
        for value in product(*rtn_options.values()):
            d = dict(zip(keys, value))
            quant_config = RTNConfig(**d)
            qmodel = self._apply_rtn(quant_config)
            self.assertEqual(self._count_woq_matmul(qmodel, bits=value[1], group_size=value[2]), 30)


if __name__ == "__main__":
    unittest.main()
