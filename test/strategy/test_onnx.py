"""Tests for quantization."""
import logging
import os
import shutil
import subprocess
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from transformers import AutoTokenizer

from neural_compressor import PostTrainingQuantConfig, quantization
from neural_compressor.config import TuningCriterion

logger = logging.getLogger("neural_compressor")


def check_model_is_same(model_proto1, model_proto2):
    # Compare if both models have the same number of nodes
    if len(model_proto1.graph.node) != len(model_proto2.graph.node):
        return False

    # Compare individual nodes in both models
    for node1, node2 in zip(model_proto1.graph.node, model_proto2.graph.node):
        print(node1.name, node2.name)
        # Check node name, input, output, and op_type
        if (
            node1.name != node2.name
            or node1.op_type != node2.op_type
            or node1.input != node2.input
            or node1.output != node2.output
        ):
            return False

        # Check node attribute
        if len(node1.attribute) != len(node2.attribute):
            return False

        for attr1, attr2 in zip(node1.attribute, node2.attribute):
            if attr1.name == attr2.name:
                if attr1.type == onnx.AttributeProto.FLOATS:
                    # Compare float attributes using numpy.allclose
                    if not attr1.floats == attr2.floats:
                        return False
                elif attr1.type == onnx.AttributeProto.INTS:
                    # Compare int attributes
                    if attr1.ints != attr2.ints:
                        return False
    # Compare initializer
    init1 = {init.name: init for init in model_proto1.graph.initializer}
    init2 = {init.name: init for init in model_proto2.graph.initializer}
    for name in init1.keys():
        if name not in init2 or not (numpy_helper.to_array(init1[name]) == numpy_helper.to_array(init2[name])).all():
            return False

    # Compare model inputs and outputs
    if model_proto1.graph.input != model_proto2.graph.input or model_proto1.graph.output != model_proto2.graph.output:
        return False

    return True


def check_model_is_same(model_proto1, model_proto2):
    # Compare if both models have the same number of nodes
    if len(model_proto1.graph.node) != len(model_proto2.graph.node):
        return False

    # Compare individual nodes in both models
    for node1, node2 in zip(model_proto1.graph.node, model_proto2.graph.node):
        print(node1.name, node2.name)
        # Check node name, input, output, and op_type
        if (
            node1.name != node2.name
            or node1.op_type != node2.op_type
            or node1.input != node2.input
            or node1.output != node2.output
        ):
            return False

        # Check node attribute
        if len(node1.attribute) != len(node2.attribute):
            return False

        for attr1, attr2 in zip(node1.attribute, node2.attribute):
            if attr1.name == attr2.name:
                if attr1.type == onnx.AttributeProto.FLOATS:
                    # Compare float attributes using numpy.allclose
                    if not attr1.floats == attr2.floats:
                        return False
                elif attr1.type == onnx.AttributeProto.INTS:
                    # Compare int attributes
                    if attr1.ints != attr2.ints:
                        return False
    # Compare initializer
    init1 = {init.name: init for init in model_proto1.graph.initializer}
    init2 = {init.name: init for init in model_proto2.graph.initializer}
    for name in init1.keys():
        if name not in init2 or not (numpy_helper.to_array(init1[name]) == numpy_helper.to_array(init2[name])).all():
            return False

    # Compare model inputs and outputs
    if model_proto1.graph.input != model_proto2.graph.input or model_proto1.graph.output != model_proto2.graph.output:
        return False

    return True


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


class TestWeightOnlyQuantTuning(unittest.TestCase):
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

    def test_common(self):
        conf = PostTrainingQuantConfig(
            approach="weight_only",
        )
        q_model = quantization.fit(self.model, conf)
        for data, _ in self.dataloader:
            q_out = Inference(q_model.model, data)
            org_out = Inference(self.model, data)
            for q, org in zip(q_out, org_out):
                self.assertTrue((np.abs(q_out[0] - org_out[0]) < 0.5).all())

    def test_basic(self):
        conf = PostTrainingQuantConfig(
            approach="weight_only",
            quant_level=1,
            op_type_dict={
                ".*": {  # re.match
                    "weight": {
                        "bits": 8,  # 1-8 bits
                        "group_size": -1,  # -1 (per-channel)
                        "scheme": "sym",
                    },
                },
            },
        )

        from neural_compressor.strategy.utils.constant import WoqTuningParams

        acc_res_lst = [1.0] * len(WoqTuningParams)
        acc_res_lst[2] = 1.2
        acc_res_lst = [1.3] + acc_res_lst + [1.2, 1.3]

        def fake_eval(model):
            acc = acc_res_lst.pop(0)
            return acc

        q_model = quantization.fit(self.model, conf, calib_dataloader=self.dataloader, eval_func=fake_eval)
        for data, _ in self.dataloader:
            q_out = Inference(q_model.model, data)
            org_out = Inference(self.model, data)
            for q, org in zip(q_out, org_out):
                self.assertTrue((np.abs(q_out[0] - org_out[0]) < 0.5).all())

        # set algorithm
        conf = PostTrainingQuantConfig(
            approach="weight_only",
            quant_level=1,
            tuning_criterion=TuningCriterion(strategy="basic"),
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

        acc_res_lst = [1.3] + [1.2, 1.3]
        acc_res_lst = [1.3] + acc_res_lst + [1.2, 1.3]

        def fake_eval(model):
            acc = acc_res_lst.pop(0)
            return acc

        q_model = quantization.fit(self.model, conf, calib_dataloader=self.dataloader, eval_func=fake_eval)

        for data, _ in self.dataloader:
            q_out = Inference(q_model.model, data)
            org_out = Inference(self.model, data)
            for q, org in zip(q_out, org_out):
                self.assertTrue((np.abs(q_out[0] - org_out[0]) < 0.5).all())

    def test_auto(self):
        conf = PostTrainingQuantConfig(
            approach="weight_only",
            quant_level="auto",
        )
        from neural_compressor.strategy.utils.constant import WoqTuningParams

        acc_res_lst = [1.0] * len(WoqTuningParams)
        acc_res_lst[2] = 1.2
        acc_res_lst = [1.3] + acc_res_lst + [1.2, 1.3]

        def fake_eval(model):
            acc = acc_res_lst.pop(0)
            return acc

        q_model = quantization.fit(self.model, conf, calib_dataloader=self.dataloader, eval_func=fake_eval)
        logger.info(
            f"The best tuning config with WeightOnlyQuant should be "
            f"{list(WoqTuningParams)[len(WoqTuningParams)//2].name}."
        )
        for data, _ in self.dataloader:
            q_out = Inference(q_model.model, data)
            org_out = Inference(self.model, data)
            for q, org in zip(q_out, org_out):
                self.assertTrue((np.abs(q_out[0] - org_out[0]) < 0.5).all())


if __name__ == "__main__":
    unittest.main()
