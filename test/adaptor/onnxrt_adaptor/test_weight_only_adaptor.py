import os
import shutil
import subprocess
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from transformers import AutoTokenizer

from neural_compressor import PostTrainingQuantConfig, quantization


def Inference(model, data):
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
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
        cmd = "optimum-cli export onnx --model hf-internal-testing/tiny-random-gptj --task text-generation gptj/"
        p = subprocess.Popen(
            cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
        )  # nosec
        p.communicate()

        self.model = onnx.load("gptj/decoder_model.onnx")
        self.dataloader = DummyNLPDataloader("hf-internal-testing/tiny-random-gptj")

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("nc_workspace", ignore_errors=True)
        shutil.rmtree("gptj", ignore_errors=True)

    def test_RTN_quant(self):
        conf = PostTrainingQuantConfig(
            approach="weight_only",
        )
        q_model = quantization.fit(self.model, conf)
        for data, _ in self.dataloader:
            q_out = Inference(q_model.model, data)
            org_out = Inference(self.model, data)
            for q, org in zip(q_out, org_out):
                self.assertTrue((np.abs(q_out[0] - org_out[0]) < 0.5).all())

        conf = PostTrainingQuantConfig(
            approach="weight_only",
            op_type_dict={
                ".*": {  # re.match
                    "weight": {
                        "bits": 8,  # 1-8 bits
                        "group_size": -1,  # -1 (per-channel)
                        "scheme": "sym",
                        "algorithm": "RTN",
                    },
                },
            },
        )
        q_model = quantization.fit(self.model, conf, calib_dataloader=self.dataloader)
        for data, _ in self.dataloader:
            q_out = Inference(q_model.model, data)
            org_out = Inference(self.model, data)
            for q, org in zip(q_out, org_out):
                self.assertTrue((np.abs(q_out[0] - org_out[0]) < 0.5).all())

    def test_AWQ_quant(self):
        conf = PostTrainingQuantConfig(
            approach="weight_only",
            op_type_dict={
                ".*": {  # re.match
                    "weight": {
                        "bits": 4,  # 1-8 bits
                        "group_size": -1,  # -1 (per-channel)
                        "scheme": "sym",
                        "algorithm": "AWQ",
                    },
                },
            },
            recipes={
                "awq_args": {"enable_auto_scale": True, "enable_mse_search": True},
            },
        )
        q_model = quantization.fit(self.model, conf, calib_dataloader=self.dataloader)
        for data, _ in self.dataloader:
            q_out = Inference(q_model.model, data)
            org_out = Inference(self.model, data)
            for q, org in zip(q_out, org_out):
                self.assertTrue((np.abs(q_out[0] - org_out[0]) < 0.5).all())

        conf = PostTrainingQuantConfig(
            approach="weight_only",
            op_type_dict={
                ".*": {  # re.match
                    "weight": {
                        "bits": 4,  # 1-8 bits
                        "group_size": 32,
                        "scheme": "sym",
                        "algorithm": "AWQ",
                    },
                },
            },
            recipes={
                "awq_args": {"enable_auto_scale": False, "enable_mse_search": True},
            },
        )
        q_model = quantization.fit(self.model, conf, calib_dataloader=self.dataloader)
        for data, _ in self.dataloader:
            q_out = Inference(q_model.model, data)
            org_out = Inference(self.model, data)
            for q, org in zip(q_out, org_out):
                self.assertTrue((np.abs(q_out[0] - org_out[0]) < 0.5).all())

        conf = PostTrainingQuantConfig(
            approach="weight_only",
            op_type_dict={
                ".*": {  # re.match
                    "weight": {
                        "bits": 4,  # 1-8 bits
                        "group_size": 32,
                        "scheme": "asym",
                        "algorithm": "AWQ",
                    },
                },
            },
            recipes={
                "awq_args": {"enable_auto_scale": True, "enable_mse_search": False},
            },
        )
        q_model = quantization.fit(self.model, conf, calib_dataloader=self.dataloader)
        for data, _ in self.dataloader:
            q_out = Inference(q_model.model, data)
            org_out = Inference(self.model, data)
            for q, org in zip(q_out, org_out):
                self.assertTrue((np.abs(q_out[0] - org_out[0]) < 0.5).all())

    def test_GPTQ_quant(self):
        conf = PostTrainingQuantConfig(
            approach="weight_only",
            op_type_dict={
                ".*": {  # re.match
                    "weight": {
                        "bits": 4,  # 1-8 bits
                        "group_size": -1,  # -1 (per-channel)
                        "scheme": "sym",
                        "algorithm": "GPTQ",
                    },
                },
            },
        )
        q_model = quantization.fit(self.model, conf, calib_dataloader=self.dataloader)
        for data, _ in self.dataloader:
            q_out = Inference(q_model.model, data)
            org_out = Inference(self.model, data)
            for q, org in zip(q_out, org_out):
                self.assertTrue((np.abs(q_out[0] - org_out[0]) < 0.5).all())

        conf = PostTrainingQuantConfig(
            approach="weight_only",
            op_type_dict={
                ".*": {  # re.match
                    "weight": {
                        "bits": 4,  # 1-8 bits
                        "group_size": 32,  # -1 (per-channel)
                        "scheme": "asym",
                        "algorithm": "GPTQ",
                    },
                },
            },
        )
        q_model = quantization.fit(self.model, conf, calib_dataloader=self.dataloader)
        for data, _ in self.dataloader:
            q_out = Inference(q_model.model, data)
            org_out = Inference(self.model, data)
            for q, org in zip(q_out, org_out):
                self.assertTrue((np.abs(q_out[0] - org_out[0]) < 0.7).all())

        conf = PostTrainingQuantConfig(
            approach="weight_only",
            op_type_dict={
                ".*": {  # re.match
                    "weight": {
                        "bits": 4,  # 1-8 bits
                        "group_size": -1,  # -1 (per-channel)
                        "scheme": "sym",
                        "algorithm": "GPTQ",
                    },
                },
            },
            recipes={
                "gptq_args": {"actorder": True, "mse": True, "perchannel": False},
            },
        )
        q_model = quantization.fit(self.model, conf, calib_dataloader=self.dataloader)
        for data, _ in self.dataloader:
            q_out = Inference(q_model.model, data)
            org_out = Inference(self.model, data)
            for q, org in zip(q_out, org_out):
                self.assertTrue((np.abs(q_out[0] - org_out[0]) < 0.5).all())


if __name__ == "__main__":
    unittest.main()
