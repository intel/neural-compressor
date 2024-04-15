import copy
import os
import shutil
import unittest
from copy import deepcopy

import onnx
import onnxruntime as ort
import onnxruntime.tools.symbolic_shape_infer as symbolic_shape_infer
import torch
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


class TestLayerWiseQuant(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # onnx model exported with transformers>=4.38.0 is different with low version
        # which will cause layer-wise quant ut to fail
        # limit transformers to 4.37.2
        # TODO: remove transformers version limitation
        llama_id = "yujiepan/llama-2-tiny-3layers-random"
        main_export(llama_id, output="llama-2-tiny-3layers-random", task="text-generation")
        model_path = find_onnx_file("llama-2-tiny-3layers-random")

        model = onnx.load(model_path)
        model = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(model, auto_merge=True)
        infer_shape_model_path = "llama-2-tiny-3layers-random/model-infer-shape.onnx"
        onnx.save(model, infer_shape_model_path)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.optimized_model_filepath = "llama-2-tiny-3layers-random/optimized_model.onnx"
        ort.InferenceSession(infer_shape_model_path, sess_options)

        self.llama = "llama-2-tiny-3layers-random/optimized_model.onnx"
        self.calibration_data_reader = DummyNLPDataloader(llama_id)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("llama-2-tiny-3layers-random", ignore_errors=True)

    def setUp(self):
        # print the test name
        logger.info(f"Running ONNXRT TestLayerWiseQuant test: {self.id()}")

    def _check_model_is_quantized(self, model):
        node_optypes = [node.op_type for node in model.graph.node]
        return "MatMulNBits" in node_optypes or "MatMulFpQ4" in node_optypes

    def _get_quantized_matmul_weight(self, model, matmul_name):
        weight_init_name = None
        for node in model.graph.node:
            if node.name == matmul_name:
                weight_init_name = node.input[1]
        if weight_init_name is None:
            return None

        weight_init = None
        for init in model.graph.initializer:
            if init.name == weight_init_name:
                weight_init = onnx.numpy_helper.to_array(init)
        return weight_init

    def _apply_quantize(self, quant_config, data_reader=None):
        from neural_compressor_ort.quantization.quantize import _quantize

        fp32_model = copy.deepcopy(self.llama)
        if data_reader is None:
            qmodel = _quantize(fp32_model, quant_config)
        else:
            qmodel = _quantize(fp32_model, quant_config, data_reader)
        self.assertIsNotNone(qmodel)
        return qmodel

    def test_rtn_layer_wise(self):
        from neural_compressor_ort.quantization import RTNConfig

        rtn_config = RTNConfig(layer_wise_quant=True)
        qmodel_lwq = self._apply_quantize(rtn_config)
        self.assertTrue(self._check_model_is_quantized(qmodel_lwq))

        rtn_config = RTNConfig(layer_wise_quant=False)
        qmodel = self._apply_quantize(rtn_config)
        self.assertTrue(self._check_model_is_quantized(qmodel))

        lwq_quantized_weight = self._get_quantized_matmul_weight(qmodel_lwq, "/lm_head/MatMul_Q4")
        self.assertIsNotNone(lwq_quantized_weight)
        quantized_weight = self._get_quantized_matmul_weight(qmodel, "/lm_head/MatMul_Q4")
        self.assertIsNotNone(quantized_weight)
        self.assertTrue((lwq_quantized_weight == quantized_weight).all())

    def test_rtn_layer_wise_with_ort_like_api(self):
        from neural_compressor_ort.quantization import matmul_4bits_quantizer

        # get qmodel without layer_wise_quant
        algo_config = matmul_4bits_quantizer.RTNWeightOnlyQuantConfig(layer_wise_quant=False)
        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
            copy.deepcopy(self.llama),
            algo_config=algo_config,
        )
        quant.process()
        qmodel = quant.model
        self.assertIsNotNone(qmodel)
        self.assertTrue(self._check_model_is_quantized(qmodel))

        # get qmodel with layer_wise_quant
        algo_config = matmul_4bits_quantizer.RTNWeightOnlyQuantConfig(layer_wise_quant=True)
        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
            copy.deepcopy(self.llama),
            algo_config=algo_config,
        )
        quant.process()
        qmodel_lwq = quant.model
        self.assertIsNotNone(qmodel_lwq)
        self.assertTrue(self._check_model_is_quantized(qmodel_lwq))

        # compare qmodel
        lwq_quantized_weight = self._get_quantized_matmul_weight(qmodel_lwq, "/lm_head/MatMul_Q4")
        self.assertIsNotNone(lwq_quantized_weight)
        quantized_weight = self._get_quantized_matmul_weight(qmodel, "/lm_head/MatMul_Q4")
        self.assertIsNotNone(quantized_weight)
        self.assertTrue((lwq_quantized_weight == quantized_weight).all())

    def test_gptq_layer_wise(self):
        from neural_compressor_ort.quantization import GPTQConfig

        self.calibration_data_reader.rewind()
        gptq_config = GPTQConfig(layer_wise_quant=True)
        qmodel_lwq = self._apply_quantize(gptq_config, self.calibration_data_reader)
        self.assertTrue(self._check_model_is_quantized(qmodel_lwq))

        self.calibration_data_reader.rewind()
        gptq_config = GPTQConfig(layer_wise_quant=False)
        qmodel = self._apply_quantize(gptq_config, self.calibration_data_reader)
        self.assertTrue(self._check_model_is_quantized(qmodel))

        lwq_quantized_weight = self._get_quantized_matmul_weight(qmodel_lwq, "/lm_head/MatMul_Q4")
        self.assertIsNotNone(lwq_quantized_weight)
        quantized_weight = self._get_quantized_matmul_weight(qmodel, "/lm_head/MatMul_Q4")
        self.assertIsNotNone(quantized_weight)
        self.assertTrue((lwq_quantized_weight == quantized_weight).all())

    def test_gptq_layer_wise_with_ort_like_api(self):
        from neural_compressor_ort.quantization import matmul_4bits_quantizer

        # get qmodel without layer_wise_quant
        algo_config = matmul_4bits_quantizer.GPTQWeightOnlyQuantConfig(
            layer_wise_quant=False, calibration_data_reader=self.calibration_data_reader
        )
        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
            copy.deepcopy(self.llama),
            algo_config=algo_config,
        )
        quant.process()
        qmodel = quant.model
        self.assertIsNotNone(qmodel)
        self.assertTrue(self._check_model_is_quantized(qmodel))

        # get qmodel with layer_wise_quant
        algo_config = matmul_4bits_quantizer.GPTQWeightOnlyQuantConfig(
            layer_wise_quant=True, calibration_data_reader=self.calibration_data_reader
        )
        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
            copy.deepcopy(self.llama),
            algo_config=algo_config,
        )
        quant.process()
        qmodel_lwq = quant.model
        self.assertIsNotNone(qmodel_lwq)
        self.assertTrue(self._check_model_is_quantized(qmodel_lwq))

        # compare qmodel
        lwq_quantized_weight = self._get_quantized_matmul_weight(qmodel_lwq, "/lm_head/MatMul_Q4")
        self.assertIsNotNone(lwq_quantized_weight)
        quantized_weight = self._get_quantized_matmul_weight(qmodel, "/lm_head/MatMul_Q4")
        self.assertIsNotNone(quantized_weight)
        self.assertTrue((lwq_quantized_weight == quantized_weight).all())


if __name__ == "__main__":
    unittest.main()
