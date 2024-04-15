import os
import shutil
import unittest

import torch
import copy
from optimum.exporters.onnx import main_export
from transformers import AutoTokenizer

from neural_compressor_ort.common import Logger
from neural_compressor_ort.quantization.calibrate import CalibrationDataReader

logger = Logger().get_logger()


def find_onnx_file(folder_path):
    # return first .onnx file path in folder_path
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".onnx"):
                return os.path.join(root, file)
    return None


class DummyNLPDataloader(CalibrationDataReader):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sequence_a = "intel-extension-for-transformers is based in SH"
        self.sequence_b = "Where is intel-extension-for-transformers based? NYC or SH"

        self.encoded_list = []
        encoded_input = dict(self.tokenizer(self.sequence_a, self.sequence_b, return_tensors="pt"))
        input_shape = encoded_input["input_ids"].shape
        encoded_input["position_ids"] = (
            torch.arange(0, input_shape[-1], dtype=torch.long).unsqueeze(0).view(-1, input_shape[-1])
        )

        # convert torch tensor to numpy
        for input_name, input_value in encoded_input.items():
            if isinstance(input_value, torch.Tensor):
                encoded_input[input_name] = input_value.numpy()

        self.encoded_list.append(encoded_input)
        self.iter_next = iter(self.encoded_list)

    def get_next(self):
        return next(self.iter_next, None)

    def rewind(self):
        self.iter_next = iter(self.encoded_list)


class TestAWQQuant(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        main_export(
            "hf-internal-testing/tiny-random-gptj",
            output="gptj",
        )
        self.gptj = find_onnx_file("./gptj")
        self.calibration_data_reader = DummyNLPDataloader("hf-internal-testing/tiny-random-gptj")

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("gptj", ignore_errors=True)

    def setUp(self):
        # print the test name
        logger.info(f"Running ONNXRT TestAWQQuant test: {self.id()}")

    def _count_woq_matmul(self, q_model, bits=4, group_size=32):
        op_names = [
            i.name
            for i in q_model.graph.node
            if i.op_type.startswith("MatMul") and i.input[1].endswith("_Q{}G{}".format(bits, group_size))
        ]
        return len(op_names)

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

    def _apply_awq(self, quant_config):
        logger.info(f"Test AWQ with config {quant_config}")
        from neural_compressor_ort.quantization.quantize import _quantize

        fp32_model = copy.deepcopy(self.gptj)
        qmodel = _quantize(fp32_model, quant_config, calibration_data_reader=self.calibration_data_reader)
        self.assertIsNotNone(qmodel)
        return qmodel

class TestAWQQuantWithInternalAPI(TestAWQQuant):
    def test_awq_params_combination(self):
        from neural_compressor_ort.quantization import AWQConfig

        # some tests were skipped to accelerate the CI
        # TODO: check params combination.
        # TODO: Add number check for group_size.
        awq_options = {
            "weight_dtype": ["int"],
            "weight_bits": [4, 3, 8],
            "weight_group_size": [32],
            "weight_sym": [True, False],
            "act_dtype": ["fp32"],
            "accuracy_level": [0],
            "enable_auto_scale": [True, False],
            "enable_mse_search": [True, False],
        }
        from itertools import product

        keys = AWQConfig.params_list
        for value in product(*awq_options.values()):
            d = dict(zip(keys, value))
            print(d)
            quant_config = AWQConfig(**d)
            qmodel = self._apply_awq(quant_config)
            self.assertEqual(self._count_woq_matmul(qmodel, bits=value[1], group_size=value[2]), 30)

    def test_awq_config(self):
        from neural_compressor_ort.quantization import AWQConfig

        awq_config1 = AWQConfig(weight_bits=4)
        quant_config_dict = {
            "awq": {"weight_bits": 4},
        }
        awq_config2 = AWQConfig.from_dict(quant_config_dict["awq"])
        self.assertEqual(awq_config1.to_dict(), awq_config2.to_dict())

    def test_quantize_awq_from_dict_default(self):
        from neural_compressor_ort.quantization import get_default_awq_config

        qmodel = self._apply_awq(quant_config=get_default_awq_config())
        self.assertIsNotNone(qmodel)
        self.assertTrue(self._check_model_is_quantized(qmodel))

    def test_quantize_awq_from_dict_beginner(self):
        quant_config = {
            "awq": {
                "weight_bits": 4,
                "weight_group_size": 32,
            },
        }
        qmodel = self._apply_awq(quant_config)
        self.assertIsNotNone(qmodel)
        self.assertIsNotNone(qmodel)
        self.assertTrue(self._check_model_is_quantized(qmodel))

    def test_quantize_awq_from_class_beginner(self):
        from neural_compressor_ort.quantization import AWQConfig

        quant_config = AWQConfig(weight_bits=4, weight_group_size=32)
        qmodel = self._apply_awq(quant_config)
        self.assertIsNotNone(qmodel)

    def test_quantize_awq_fallback_from_class_beginner(self):
        from neural_compressor_ort.quantization import AWQConfig

        fp32_config = AWQConfig(weight_dtype="fp32")
        quant_config = AWQConfig(
            weight_bits=4,
            weight_dtype="int",
            weight_sym=False,
            weight_group_size=32,
        )
        quant_config.set_local("/h.4/mlp/fc_out/MatMul", fp32_config)
        qmodel = self._apply_awq(quant_config)
        self.assertIsNotNone(qmodel)
        self.assertEqual(self._count_woq_matmul(qmodel), 29)
        self.assertFalse(self._check_node_is_quantized(qmodel, "/h.4/mlp/fc_out/MatMul"))

    def test_quantize_awq_from_dict_advance(self):
        quant_config = {
            "awq": {
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
        qmodel = self._apply_awq(quant_config)
        self.assertIsNotNone(qmodel)
        self.assertEqual(self._count_woq_matmul(qmodel), 29)
        self.assertFalse(self._check_node_is_quantized(qmodel, "/h.4/mlp/fc_out/MatMul"))

        quant_config = {
            "awq": {
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
        qmodel = self._apply_awq(quant_config)
        self.assertIsNotNone(qmodel)
        for node in qmodel.graph.node:
            if node.name == "/h.4/mlp/fc_out/MatMul":
                self.assertTrue(node.input[1].endswith("Q8G32"))

class TestAWQQuantWithORTLikeAPI(TestAWQQuant):
    def test_awq_config_4bits(self):
        from neural_compressor_ort.quantization import matmul_4bits_quantizer

        algo_config = matmul_4bits_quantizer.AWQWeightOnlyQuantConfig(
            calibration_data_reader=self.calibration_data_reader)

        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(copy.deepcopy(self.gptj),
                                                            block_size=32,
                                                            is_symmetric=False,
                                                            algo_config=algo_config,)
        quant.process()
        self.assertIsNotNone(quant.model)
        self.assertTrue(self._check_model_is_quantized(quant.model))

    def test_awq_config_4bits_with_exclude_node(self):
        from neural_compressor_ort.quantization import matmul_4bits_quantizer

        algo_config = matmul_4bits_quantizer.AWQWeightOnlyQuantConfig(
            calibration_data_reader=self.calibration_data_reader)

        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(copy.deepcopy(self.gptj),
                                                            block_size=32,
                                                            is_symmetric=False,
                                                            algo_config=algo_config,
                                                            nodes_to_exclude=["/h.4/mlp/fc_out/MatMul"])
        quant.process()
        self.assertIsNotNone(quant.model)
        self.assertTrue(self._check_model_is_quantized(quant.model))
        self.assertFalse(self._check_node_is_quantized(quant.model, "/h.4/mlp/fc_out/MatMul"))

    def test_awq_config_nbits(self):
        from neural_compressor_ort.quantization import matmul_nbits_quantizer

        algo_config = matmul_nbits_quantizer.AWQWeightOnlyQuantConfig(
            calibration_data_reader=self.calibration_data_reader)

        for n_bits in [3, 4, 8]:
            quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(copy.deepcopy(self.gptj),
                                                                n_bits=n_bits,
                                                                block_size=32,
                                                                is_symmetric=False,
                                                                algo_config=algo_config,)
            quant.process()
            self.assertIsNotNone(quant.model)
            self.assertEqual(self._count_woq_matmul(quant.model, bits=n_bits, group_size=32), 30)

    def test_awq_config_nbits_with_exclude_node(self):
        from neural_compressor_ort.quantization import matmul_nbits_quantizer

        algo_config = matmul_nbits_quantizer.AWQWeightOnlyQuantConfig(
            calibration_data_reader=self.calibration_data_reader)

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
