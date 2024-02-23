import os
import torch
import shutil
import unittest
from copy import deepcopy
from transformers import AutoTokenizer

import onnx
from optimum.exporters.onnx import main_export
import onnxruntime as ort
import onnxruntime.tools.symbolic_shape_infer as symbolic_shape_infer

from neural_compressor.onnxrt.quantization.calibrate import CalibrationDataReader
from neural_compressor.common import Logger

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
        llama_id = "yujiepan/llama-2-tiny-3layers-random"
        main_export(llama_id, output="llama-2-tiny", task="text-generation")
        model_path = find_onnx_file("llama-2-tiny")

        model = onnx.load(model_path)
        model = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(model, auto_merge=True)
        infer_shape_model_path = 'llama-2-tiny/model-infer-shape.onnx'
        onnx.save(model, infer_shape_model_path)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.optimized_model_filepath = "llama-2-tiny/optimized_model.onnx"
        ort.InferenceSession(infer_shape_model_path, sess_options)

        self.llama = "llama-2-tiny/optimized_model.onnx"
        self.calibration_data_reader = DummyNLPDataloader(llama_id)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("llama-2-tiny", ignore_errors=True)

    def setUp(self):
        # print the test name
        logger.info(f"Running ONNXRT TestLayerWiseQuant test: {self.id()}")

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

    def inference(self, modelproto, data):
        sess = ort.InferenceSession(modelproto.SerializeToString(), providers=["CPUExecutionProvider"])
        out = sess.run(None, data)
        return out

    def _apply_quantize(self, quant_config, data_reader=None):
        from neural_compressor.onnxrt.quantization.quantize import _quantize

        fp32_model = deepcopy(self.llama)
        if data_reader is None:
            qmodel = _quantize(fp32_model, quant_config)
        else:
            qmodel = _quantize(fp32_model, quant_config, data_reader)
        self.assertIsNotNone(qmodel)
        return qmodel

    def test_rtn_layer_wise(self):
        from neural_compressor.onnxrt.quantization import RTNConfig

        rtn_config = RTNConfig(layer_wise_quant=True)
        qmodel_lwq = self._apply_quantize(rtn_config)
        self.assertTrue(self._check_model_is_quantized(qmodel_lwq))

        rtn_config = RTNConfig(layer_wise_quant=False)
        qmodel = self._apply_quantize(rtn_config)
        self.assertTrue(self._check_model_is_quantized(qmodel))

        self.calibration_data_reader.rewind()
        while True:
            inputs = self.calibration_data_reader.get_next()
            if not inputs:
                break
            layerwise_q_out = self.inference(qmodel_lwq, inputs)
            q_out = self.inference(qmodel, inputs)
            self.assertTrue((layerwise_q_out[0] == q_out[0]).all())

    def test_gptq_layer_wise(self):
        from neural_compressor.onnxrt.quantization import GPTQConfig

        gptq_config = GPTQConfig(layer_wise_quant=True)
        qmodel_lwq = self._apply_quantize(gptq_config, self.calibration_data_reader)
        self.assertTrue(self._check_model_is_quantized(qmodel_lwq))

        gptq_config = GPTQConfig(layer_wise_quant=False)
        qmodel = self._apply_quantize(gptq_config, self.calibration_data_reader)
        self.assertTrue(self._check_model_is_quantized(qmodel))

        self.calibration_data_reader.rewind()
        while True:
            inputs = self.calibration_data_reader.get_next()
            if not inputs:
                break
            layerwise_q_out = self.inference(qmodel_lwq, inputs)
            q_out = self.inference(qmodel, inputs)
            self.assertTrue((layerwise_q_out[0] == q_out[0]).all())


if __name__ == "__main__":
    unittest.main()
