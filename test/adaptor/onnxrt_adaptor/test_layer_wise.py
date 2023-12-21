import os
import shutil
import subprocess
import unittest

import onnx
import onnxruntime as ort
from transformers import AutoTokenizer

from neural_compressor import PostTrainingQuantConfig, quantization
from neural_compressor.utils.constant import FP32


def Inference(model_path, data):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, data)
    return out


class DummyNLPDataloader(object):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sequence_a = "intel-extension-for-transformers is based in SH"
        self.sequence_b = "Where is intel-extension-for-transformers based? NYC or SH"
        self.encoded_dict = self.tokenizer(self.sequence_a, self.sequence_b, return_tensors="pt")
        self.encoded_dict["labels"] = 1
        self.batch_size = 1

    def __iter__(self):
        yield {
            "input_ids": self.encoded_dict["input_ids"].detach().cpu().numpy(),
            "attention_mask": self.encoded_dict["attention_mask"].detach().cpu().numpy(),
        }, self.encoded_dict["labels"]


class TestWeightOnlyAdaptor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        cmd = "optimum-cli export onnx --model yujiepan/llama-2-tiny-3layers-random --task text-generation --legacy tiny-llama/"
        p = subprocess.Popen(
            cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
        )  # nosec
        p.communicate()

        self.model = onnx.load("tiny-llama/decoder_model.onnx")
        self.dataloader = DummyNLPDataloader("yujiepan/llama-2-tiny-3layers-random")

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("nc_workspace", ignore_errors=True)
        shutil.rmtree("tiny-llama", ignore_errors=True)

    def test_layer_wise_W8A8_quant(self):
        # layer-wise quantization
        layerwise_quantized_model_path = "tiny-llama/layerwise_quantized_decoder_model.onnx"
        config = PostTrainingQuantConfig(
            calibration_sampling_size=[1], recipes={"layer_wise_quant": True}, op_type_dict={"^((?!(MatMul)).)*$": FP32}
        )
        q_model = quantization.fit("tiny-llama/decoder_model.onnx", config, calib_dataloader=self.dataloader)
        q_model.save(layerwise_quantized_model_path)

        # not layer-wise quantization
        quantized_model_path = "tiny-llama/quantized_decoder_model.onnx"
        config = PostTrainingQuantConfig(
            calibration_sampling_size=[1],
            recipes={"layer_wise_quant": False, "graph_optimization_level": "ENABLE_BASIC"},
            op_type_dict={"^((?!(MatMul)).)*$": FP32},
        )
        q_model = quantization.fit("tiny-llama/decoder_model.onnx", config, calib_dataloader=self.dataloader)
        q_model.save(quantized_model_path)

        for data, _ in self.dataloader:
            layerwise_q_out = Inference(layerwise_quantized_model_path, data)
            q_out = Inference(quantized_model_path, data)
            self.assertTrue((layerwise_q_out[0] == q_out[0]).all())


if __name__ == "__main__":
    unittest.main()
