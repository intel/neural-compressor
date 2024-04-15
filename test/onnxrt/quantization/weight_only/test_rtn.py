import os
import copy
import shutil
import unittest

from optimum.exporters.onnx import main_export

from neural_compressor_ort.common import Logger

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

    def _check_model_is_quantized(self, model):
        node_optypes = [node.op_type for node in model.graph.node]
        return "MatMulNBits" in node_optypes or "MatMulFpQ4" in node_optypes

    def _check_node_is_quantized(self, model, node_name):
        for node in model.graph.node:
            if (node.name == node_name or node.name == node_name + "_Q4") and node.op_type in [
                "MatMulNBits",
                "MatMulFpQ4",
            ]:
                return True
        return False

    def _count_woq_matmul(self, q_model, bits=4, group_size=32):
        op_names = [
            i.name
            for i in q_model.graph.node
            if i.op_type.startswith("MatMul") and i.input[1].endswith("_Q{}G{}".format(bits, group_size))
        ]
        return len(op_names)

    def _apply_rtn(self, quant_config):
        logger.info(f"Test RTN with config {quant_config}")
        from neural_compressor_ort.quantization.quantize import _quantize

        fp32_model = copy.deepcopy(self.gptj)
        qmodel = _quantize(fp32_model, quant_config)
        self.assertIsNotNone(qmodel)
        return qmodel

class TestRTNQuantWithInternalAPI(TestRTNQuant):
    def test_rtn_params_combination(self):
        from neural_compressor_ort.quantization import RTNConfig

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

    def test_rtn_config(self):
        from neural_compressor_ort.quantization import RTNConfig

        rtn_config1 = RTNConfig(weight_bits=4)
        quant_config_dict = {
            "rtn": {"weight_bits": 4},
        }
        rtn_config2 = RTNConfig.from_dict(quant_config_dict["rtn"])
        self.assertEqual(rtn_config1.to_dict(), rtn_config2.to_dict())

    def test_quantize_rtn_from_dict_default(self):
        from neural_compressor_ort.quantization import get_default_rtn_config
        from neural_compressor_ort.quantization.quantize import _quantize

        qmodel = self._apply_rtn(quant_config=get_default_rtn_config())
        self.assertIsNotNone(qmodel)
        self.assertTrue(self._check_model_is_quantized(qmodel))

    def test_quantize_rtn_from_dict_beginner(self):
        from neural_compressor_ort.quantization.quantize import _quantize

        quant_config = {
            "rtn": {
                "weight_bits": 4,
                "weight_group_size": 32,
            },
        }
        qmodel = self._apply_rtn(quant_config)
        self.assertIsNotNone(qmodel)
        self.assertIsNotNone(qmodel)
        self.assertTrue(self._check_model_is_quantized(qmodel))

    def test_quantize_rtn_from_class_beginner(self):
        from neural_compressor_ort.quantization import RTNConfig
        from neural_compressor_ort.quantization.quantize import _quantize

        quant_config = RTNConfig(weight_bits=4, weight_group_size=32)
        qmodel = self._apply_rtn(quant_config)
        self.assertIsNotNone(qmodel)

    def test_quantize_rtn_fallback_from_class_beginner(self):
        from neural_compressor_ort.quantization import RTNConfig
        from neural_compressor_ort.quantization.quantize import _quantize

        fp32_config = RTNConfig(weight_dtype="fp32")
        quant_config = RTNConfig(
            weight_bits=4,
            weight_dtype="int",
            weight_sym=False,
            weight_group_size=32,
        )
        quant_config.set_local("/h.4/mlp/fc_out/MatMul", fp32_config)
        qmodel = self._apply_rtn(quant_config)
        self.assertIsNotNone(qmodel)
        self.assertEqual(self._count_woq_matmul(qmodel), 29)
        self.assertFalse(self._check_node_is_quantized(qmodel, "/h.4/mlp/fc_out/MatMul"))

    def test_quantize_rtn_from_dict_advance(self):
        from neural_compressor_ort.quantization.quantize import _quantize

        quant_config = {
            "rtn": {
                "global": {
                    "weight_bits": 4,
                    "weight_group_size": 32,
                },
                "local": {
                    "/h.4/mlp/fc_out/MatMul": {
                        "weight_dtype": "fp32",
                    }
                },
            }
        }
        qmodel = self._apply_rtn(quant_config)
        self.assertIsNotNone(qmodel)
        self.assertEqual(self._count_woq_matmul(qmodel), 29)
        self.assertFalse(self._check_node_is_quantized(qmodel, "/h.4/mlp/fc_out/MatMul"))

        quant_config = {
            "rtn": {
                "global": {
                    "weight_bits": 4,
                    "weight_group_size": 32,
                },
                "local": {
                    "/h.4/mlp/fc_out/MatMul": {
                        "weight_bits": 8,
                        "weight_group_size": 32,
                    }
                },
            }
        }
        qmodel = self._apply_rtn(quant_config)
        self.assertIsNotNone(qmodel)
        for node in qmodel.graph.node:
            if node.name == "/h.4/mlp/fc_out/MatMul":
                self.assertTrue(node.input[1].endswith("Q8G32"))

class TestRTNQuantWithORTLikeAPI(TestRTNQuant):
    def test_rtn_config_4bits(self):
        from neural_compressor_ort.quantization import matmul_4bits_quantizer

        algo_config = matmul_4bits_quantizer.RTNWeightOnlyQuantConfig()

        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(copy.deepcopy(self.gptj),
                                                            block_size=32,
                                                            is_symmetric=False,
                                                            algo_config=algo_config,)
        quant.process()
        self.assertIsNotNone(quant.model)
        self.assertTrue(self._check_model_is_quantized(quant.model))

    def test_rtn_config_4bits_with_exclude_node(self):
        from neural_compressor_ort.quantization import matmul_4bits_quantizer

        algo_config = matmul_4bits_quantizer.RTNWeightOnlyQuantConfig()

        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(copy.deepcopy(self.gptj),
                                                            block_size=32,
                                                            is_symmetric=False,
                                                            algo_config=algo_config,
                                                            nodes_to_exclude=["/h.4/mlp/fc_out/MatMul"])
        quant.process()
        self.assertIsNotNone(quant.model)
        self.assertTrue(self._check_model_is_quantized(quant.model))
        self.assertFalse(self._check_node_is_quantized(quant.model, "/h.4/mlp/fc_out/MatMul"))

    def test_rtn_config_nbits(self):
        from neural_compressor_ort.quantization import matmul_nbits_quantizer

        algo_config = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig()

        for n_bits in [3, 4, 8]:
            quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(copy.deepcopy(self.gptj),
                                                                n_bits=n_bits,
                                                                block_size=32,
                                                                is_symmetric=False,
                                                                algo_config=algo_config,)
            quant.process()
            self.assertIsNotNone(quant.model)
            self.assertEqual(self._count_woq_matmul(quant.model, bits=n_bits, group_size=32), 30)

    def test_rtn_config_nbits_with_exclude_node(self):
        from neural_compressor_ort.quantization import matmul_nbits_quantizer

        algo_config = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig()

        for n_bits in [3, 4, 8]:
            quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(copy.deepcopy(self.gptj),
                                                                n_bits=n_bits,
                                                                block_size=32,
                                                                is_symmetric=False,
                                                                algo_config=algo_config,
                                                                nodes_to_exclude=["/h.4/mlp/fc_out/MatMul"])
            quant.process()
            self.assertIsNotNone(quant.model)
            self.assertEqual(self._count_woq_matmul(quant.model, bits=n_bits, group_size=32), 29)


if __name__ == "__main__":
    unittest.main()
