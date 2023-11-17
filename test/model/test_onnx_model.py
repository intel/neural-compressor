import os
import shutil
import subprocess
import sys
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from neural_compressor import PostTrainingQuantConfig, quantization
from neural_compressor.data import DATALOADERS, Datasets
from neural_compressor.model.onnx_model import ONNXModel


def get_onnx_model():
    import torch
    import torchvision
    from torch.autograd import Variable

    model = torchvision.models.resnet18()
    x = Variable(torch.randn(1, 3, 224, 224))
    torch_out = torch.onnx.export(model, x, "resnet18.onnx", export_params=True, verbose=True)


def generate_input_initializer(tensor_shape, tensor_dtype, input_name):
    """Helper function to generate initializers for test inputs."""
    tensor = np.random.ranf(tensor_shape).astype(tensor_dtype)
    init = numpy_helper.from_array(tensor, input_name)
    return init


class TestOnnxModel(unittest.TestCase):
    def setUp(self):
        #   Relu
        #    |      \
        #   Conv     \
        #    |        \
        #   Relu       |
        #    |       Conv
        #   Conv      /
        #      \     /
        #         |
        #        Add

        input0 = helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1, 3, 1, 3])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 1, 3])

        X1_weight = generate_input_initializer([3, 3, 1, 1], np.float32, "X1_weight")
        X1_bias = generate_input_initializer([3], np.float32, "X1_bias")
        X3_weight = generate_input_initializer([3, 3, 1, 1], np.float32, "X3_weight")
        X3_bias = generate_input_initializer([3], np.float32, "X3_bias")
        X5_weight = generate_input_initializer([3, 3, 1, 1], np.float32, "X5_weight")
        X5_bias = generate_input_initializer([3], np.float32, "X5_bias")

        relu_node_1 = onnx.helper.make_node("Relu", ["input0"], ["X1"], name="Relu1")
        conv_node_1 = onnx.helper.make_node("Conv", ["X1", "X1_weight", "X1_bias"], ["X2"], name="Conv1")
        relu_node_2 = onnx.helper.make_node("Relu", ["X2"], ["X3"], name="Relu2")
        conv_node_2 = onnx.helper.make_node("Conv", ["X3", "X3_weight", "X3_bias"], ["X4"], name="Conv2")
        conv_node_3 = onnx.helper.make_node("Conv", ["X1", "X5_weight", "X5_bias"], ["X5"], name="Conv3")
        add_node = onnx.helper.make_node("Add", ["X4", "X5"], ["output"], name="Add")

        graph = helper.make_graph(
            [relu_node_1, conv_node_1, relu_node_2, conv_node_2, conv_node_3, add_node],
            "test_graph_6",
            [input0],
            [output],
        )
        graph.initializer.add().CopyFrom(X1_weight)
        graph.initializer.add().CopyFrom(X1_bias)
        graph.initializer.add().CopyFrom(X3_weight)
        graph.initializer.add().CopyFrom(X3_bias)
        graph.initializer.add().CopyFrom(X5_weight)
        graph.initializer.add().CopyFrom(X5_bias)

        model = helper.make_model(graph)
        test_model_path = "./test_model_6.onnx"
        onnx.save(model, test_model_path)
        model = onnx.load(test_model_path)
        self.model = ONNXModel(model)

        #    QuantizeLinear
        #        |
        #    QLinearConv
        #        |
        #    DequantizeLinear
        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 1, 5, 5])
        A_scale = helper.make_tensor_value_info("A_scale", TensorProto.FLOAT, [1])
        a_scale = generate_input_initializer([1], np.float32, "A_scale")
        A_zero = helper.make_tensor_value_info("A_zero_point", TensorProto.INT8, [1])
        a_zero_point = generate_input_initializer([1], np.int8, "A_zero_point")
        B_scale = helper.make_tensor_value_info("B_scale", TensorProto.FLOAT, [1])
        b_scale = generate_input_initializer([1], np.float32, "B_scale")
        B_zero = helper.make_tensor_value_info("B_zero_point", TensorProto.INT8, [1])
        b_zero_point = generate_input_initializer([1], np.int8, "B_zero_point")
        C = helper.make_tensor_value_info("C", TensorProto.INT8, [1, 1, 5, 5])
        c = generate_input_initializer([1, 1, 5, 5], np.int8, "C")
        C_scale = helper.make_tensor_value_info("C_scale", TensorProto.FLOAT, [1])
        c_scale = generate_input_initializer([1], np.float32, "C_scale")
        C_zero = helper.make_tensor_value_info("C_zero_point", TensorProto.INT8, [1])
        c_zero_point = generate_input_initializer([1], np.int8, "C_zero_point")
        E = helper.make_tensor_value_info("E", TensorProto.INT32, [1])
        e = generate_input_initializer([1], np.int32, "E")
        D_scale = helper.make_tensor_value_info("D_scale", TensorProto.FLOAT, [1])
        d_scale = generate_input_initializer([1], np.float32, "D_scale")
        D_zero = helper.make_tensor_value_info("D_zero_point", TensorProto.INT8, [1])
        d_zero_point = generate_input_initializer([1], np.int8, "D_zero_point")
        D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [1, 1, 5, 5])
        quantize_node = onnx.helper.make_node(
            "QuantizeLinear", ["A", "A_scale", "A_zero_point"], ["B_quantized"], name="A_QuantizeLinear"
        )
        conv_node = onnx.helper.make_node(
            "QLinearConv",
            [
                "B_quantized",
                "B_scale",
                "B_zero_point",
                "C_quantized",
                "C_scale",
                "C_zero_point",
                "D_scale",
                "D_zero_point",
                "E",
            ],
            ["D_quantized"],
            name="conv_quant",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
        dequantize_node = onnx.helper.make_node(
            "DequantizeLinear", ["D_quantized", "D_scale", "D_zero_point"], ["D"], name="D_DequantizeLinear"
        )
        graph = helper.make_graph(
            [quantize_node, conv_node, dequantize_node],
            "test_graph_7",
            [A, A_scale, A_zero, C, C_scale, C_zero, E, D_scale, D_zero],
            [D],
        )
        graph.initializer.add().CopyFrom(a_scale)
        graph.initializer.add().CopyFrom(a_zero_point)
        graph.initializer.add().CopyFrom(b_scale)
        graph.initializer.add().CopyFrom(b_zero_point)
        graph.initializer.add().CopyFrom(c)
        graph.initializer.add().CopyFrom(c_scale)
        graph.initializer.add().CopyFrom(c_zero_point)
        graph.initializer.add().CopyFrom(e)
        graph.initializer.add().CopyFrom(d_scale)
        graph.initializer.add().CopyFrom(d_zero_point)
        model = helper.make_model(graph)
        self.q_model = ONNXModel(model)

        #      MatMul
        #        |
        #       Add
        #        |
        #     Reshape
        #        |
        #     Reshape
        #        |
        #      MatMul
        #        |
        #       Add

        input = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 4])

        W1 = onnx.helper.make_tensor_value_info("W1", onnx.TensorProto.FLOAT, [4, 5])
        w1 = generate_input_initializer([4, 5], np.float32, "W1")
        B1 = onnx.helper.make_tensor_value_info("b1", onnx.TensorProto.FLOAT, [5])
        b1 = generate_input_initializer([5], np.float32, "b1")
        shape = numpy_helper.from_array(np.array((2, 5)).astype(np.int64), name="shape")
        W2 = onnx.helper.make_tensor_value_info("W2", onnx.TensorProto.FLOAT, [5, 6])
        w2 = generate_input_initializer([5, 6], np.float32, "W2")
        B2 = onnx.helper.make_tensor_value_info("b2", onnx.TensorProto.FLOAT, [6])
        b2 = generate_input_initializer([6], np.float32, "b2")
        output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 6])

        node1 = onnx.helper.make_node("MatMul", inputs=["input", "W1"], outputs=["y1"])
        node2 = onnx.helper.make_node("Add", inputs=["y1", "b1"], outputs=["y1_add_b1"])
        node3 = onnx.helper.make_node("Reshape", inputs=["y1_add_b1", "shape"], outputs=["y2"])
        node4 = onnx.helper.make_node("Reshape", inputs=["y2", "shape"], outputs=["y3"])
        node5 = onnx.helper.make_node("MatMul", inputs=["y3", "W2"], outputs=["y4"])
        node6 = onnx.helper.make_node("Add", inputs=["y4", "b2"], outputs=["output"])

        graph = onnx.helper.make_graph(
            [node1, node2, node3, node4, node5, node6], "test_matmul_reshape_graph", [input, W1, B1, W2, B2], [output]
        )
        graph.initializer.add().CopyFrom(w1)
        graph.initializer.add().CopyFrom(b1)
        graph.initializer.add().CopyFrom(w2)
        graph.initializer.add().CopyFrom(b2)
        graph.initializer.add().CopyFrom(shape)

        model = onnx.helper.make_model(graph, **{"opset_imports": [onnx.helper.make_opsetid("", 14)]})
        self.matmul_reshape_model = model

        cmd = "optimum-cli export onnx --model hf-internal-testing/tiny-random-gptj --task text-generation gptj/"
        p = subprocess.Popen(
            cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
        )  # nosec
        p.communicate()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./gptj", ignore_errors=True)
        shutil.rmtree("./hf_test", ignore_errors=True)
        os.remove("model.onnx")

    def test_hf_model(self):
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoConfig, AutoTokenizer

        os.mkdir("hf_test")
        model = ONNXModel("gptj/decoder_model.onnx")
        model.save("./hf_test/decoder_model.onnx")
        self.assertTrue(os.path.exists("hf_test/config.json"))

        config = AutoConfig.from_pretrained("hf_test")
        sessions = ORTModelForCausalLM.load_model("hf_test/decoder_model.onnx")
        model = ORTModelForCausalLM(sessions[0], config, "hf_test", use_cache=False, use_io_binding=False)
        self.assertNotEqual(model, None)

    def test_nodes(self):
        self.assertEqual(len(self.model.nodes()), 6)
        nodes_name = [node.name for node in self.model.nodes()]
        nodes = ["Relu1", "Conv1", "Relu2", "Conv2", "Conv3", "Add"]
        for node in nodes:
            self.assertTrue(node in nodes_name)

    def test_initializer(self):
        self.assertEqual(len(self.model.initializer()), 6)
        inits_name = [init.name for init in self.model.initializer()]
        inits = ["X1_weight", "X1_bias", "X3_weight", "X3_bias", "X5_weight", "X5_bias"]
        for init in inits:
            self.assertTrue(init in inits_name)

    def test_remove_node(self):
        for node in self.model.nodes():
            if node.op_type == "Add":
                self.model.remove_node(node)
        self.assertEqual(len(self.model.nodes()), 5)
        nodes_name = [node.name for node in self.model.nodes()]
        nodes = ["Relu1", "Conv1", "Relu2", "Conv2", "Conv3"]
        for node in nodes:
            self.assertTrue(node in nodes_name)

    def test_remove_nodes(self):
        nodes_to_remove = []
        for node in self.model.nodes():
            if node.name == "Conv3" or node.name == "Add":
                nodes_to_remove.append(node)
        self.model.remove_nodes(nodes_to_remove)
        self.assertEqual(len(self.model.nodes()), 4)
        nodes_name = [node.name for node in self.model.nodes()]
        nodes = ["Relu1", "Conv1", "Relu2", "Conv2"]
        for node in nodes:
            self.assertTrue(node in nodes_name)

    def test_add_node(self):
        node_to_add = onnx.helper.make_node("Relu", ["output"], ["output1"], keepdims=0)
        self.model.add_node(node_to_add)
        last_node = self.model.nodes()[-1]
        self.assertEqual(last_node.op_type, "Relu")

    def test_add_nodes(self):
        nodes_to_add = []
        for i in range(2):
            node_to_add = onnx.helper.make_node(
                "Relu", ["add_node{}_input".format(str(i))], ["add_node{}_output".format(str(i))], keepdims=0
            )
            nodes_to_add.append(node_to_add)
        self.model.add_nodes(nodes_to_add)
        self.assertEqual(self.model.nodes()[-1].input, ["add_node1_input"])
        self.assertEqual(self.model.nodes()[-2].input, ["add_node0_input"])
        self.assertEqual(self.model.nodes()[-1].output, ["add_node1_output"])
        self.assertEqual(self.model.nodes()[-2].output, ["add_node0_output"])

    def test_get_initializer(self):
        inits = ["X1_weight", "X1_bias", "X3_weight", "X3_bias", "X5_weight", "X5_bias"]
        for init in inits:
            self.assertIsNotNone(self.model.get_initializer(init))

    def test_remove_initializer(self):
        for init in self.model.initializer():
            if init.name == "X1_weight":
                self.model.remove_initializer(init)
        self.assertEqual(len(self.model.initializer()), 5)
        inits_name = [init.name for init in self.model.initializer()]
        inits = ["X1_bias", "X3_weight", "X3_bias", "X5_weight", "X5_bias"]
        for init in inits:
            self.assertTrue(init in inits_name)

    def test_remove_initializers(self):
        init_to_remove = []
        for init in self.model.initializer():
            if "bias" in init.name:
                init_to_remove.append(init)
        self.model.remove_initializers(init_to_remove)
        self.assertEqual(len(self.model.initializer()), 3)
        inits_name = [init.name for init in self.model.initializer()]
        inits = ["X1_weight", "X3_weight", "X5_weight"]
        for init in inits:
            self.assertTrue(init in inits_name)

    def test_input_name_to_nodes(self):
        self.assertEqual(len(self.model.input_name_to_nodes), 12)
        ipts_name = [name for name in self.model.input_name_to_nodes]
        ipts = ["input0", "X1", "X2", "X3", "X3_weight", "X3_bias", "X5_weight", "X5_bias", "X4", "X5"]
        for ipt in ipts:
            self.assertTrue(ipt in ipts_name)

    def test_output_name_to_node(self):
        self.assertEqual(len(self.model.output_name_to_node), 6)
        opts_name = [name for name in self.model.output_name_to_node]
        opts = ["X1", "X2", "X3", "X4", "X5", "output"]
        for opt in opts:
            self.assertTrue(opt in opts_name)

    def test_get_siblings(self):
        for node in self.model.nodes():
            if node.name == "Conv1":
                siblings = self.model.get_siblings(node)
        self.assertEqual(len(siblings), 1)
        siblings_name = [sibling.name for sibling in siblings]
        names = ["Conv3"]
        for name in names:
            self.assertTrue(name in siblings_name)

    def test_get_children(self):
        for node in self.model.nodes():
            if node.name == "Relu1":
                children = self.model.get_children(node)
        self.assertEqual(len(children), 2)
        children_name = [child.name for child in children]
        names = ["Conv1", "Conv3"]
        for name in names:
            self.assertTrue(name in children_name)

    def test_get_parents(self):
        for node in self.model.nodes():
            if node.op_type == "Add":
                parents = self.model.get_parents(node)
        self.assertEqual(len(parents), 2)
        parents_name = [parent.name for parent in parents]
        names = ["Conv2", "Conv3"]
        for name in names:
            self.assertTrue(name in parents_name)

    def test_get_parent(self):
        for node in self.model.nodes():
            if node.op_type == "Add":
                node_to_get_parent = node
        parent = self.model.get_parent(node, 0)
        self.assertEqual(parent.name, "Conv2")
        parent = self.model.get_parent(node, 1)
        self.assertEqual(parent.name, "Conv3")
        parent = self.model.get_parent(node, 2)
        self.assertIsNone(parent)

    def test_find_nodes_by_initializer(self):
        for init in self.model.initializer():
            if init.name == "X1_weight":
                initializer = init
        nodes = self.model.find_nodes_by_initializer(self.model.graph(), initializer)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].name, "Conv1")

    def test_get_scale_zero(self):
        import time

        result = [0.1]

        def sub_eval(model, result):
            time.sleep(0.001 * len(result))
            return result[0]

        def eval(model):
            return sub_eval(model, result)

        dataset = Datasets("onnxrt_qdq")["dummy"]((4, 4), low=0.0, high=0.0, dtype="float32")
        dataloader = DATALOADERS["onnxrt_qdq"](dataset, 2)
        config = PostTrainingQuantConfig()
        q_model = quantization.fit(self.matmul_reshape_model, config, calib_dataloader=dataloader, eval_func=eval)
        q_model.save("test.onnx")
        scale, zp = q_model.get_scale_zero("y3_QuantizeInput_quantized")
        self.assertEqual(scale.name, "y1_add_b1_scale")
        self.assertEqual(zp.name, "y1_add_b1_zero_point")

        scale, zp = q_model.get_scale_zero("input_quantized")
        self.assertEqual(scale.name, "input_scale")
        self.assertEqual(zp.name, "input_zero_point")

    def test_save(self):
        self.model.save_model_to_file("./test_model_6.onnx", use_external_data_format=True)

    def test_find_by_name(self):
        from neural_compressor.adaptor.ox_utils.util import dtype_mapping, dtype_to_name, find_by_name

        initializer = find_by_name("X1_weight", self.model.initializer())
        self.assertIsNotNone(initializer)
        initializer = find_by_name("X1", self.model.initializer())
        self.assertIsNone(initializer)

    def test_remove_unused_nodes(self):
        self.assertEqual(len(self.model.nodes()), 6)
        node_to_add = onnx.helper.make_node("Relu", ["output1"], ["output2"], keepdims=0, name="added_relu")
        self.model.add_node(node_to_add)
        self.assertEqual(len(self.model.nodes()), 7)
        self.model.remove_unused_nodes()
        self.assertEqual(len(self.model.nodes()), 6)

    def test_check_large_model(self):
        import onnx
        import torch
        import torch.nn as nn

        from neural_compressor.model.onnx_model import ONNXModel

        class Net(nn.Module):
            def __init__(self, in_features, out_features):
                super(Net, self).__init__()
                self.fc = nn.Linear(in_features, out_features)

            def forward(self, x):
                x = self.fc(x)
                return x

        # model > 2GB
        model = Net(512, 1024 * 1024)
        input = torch.randn(512, requires_grad=True)
        with torch.no_grad():
            torch.onnx.export(model, (input,), "model.onnx", do_constant_folding=True, opset_version=13)
        model = onnx.load("model.onnx")
        model = ONNXModel(model)  # pass ModelProto
        model.check_is_large_model()
        self.assertTrue(model.is_large_model)

        model = ONNXModel("model.onnx")  # pass string
        model.check_is_large_model()
        self.assertTrue(model.is_large_model)

        model = onnx.load("model.onnx", load_external_data=False)  # not load init
        model = ONNXModel(model)
        model.check_is_large_model()
        self.assertTrue(model.is_large_model)

        # model < 2GB
        model = Net(10, 10 * 10)
        input = torch.randn(10, requires_grad=True)
        with torch.no_grad():
            torch.onnx.export(model, (input,), "model.onnx", do_constant_folding=True, opset_version=13)
        model = onnx.load("model.onnx")
        model = ONNXModel(model)  # pass ModelProto
        model.check_is_large_model()
        self.assertFalse(model.is_large_model)

        model = ONNXModel("model.onnx")  # pass string
        model.check_is_large_model()
        self.assertFalse(model.is_large_model)

        model = ONNXModel("model.onnx", load_external_data_for_model=False)  # not load init
        model.check_is_large_model()
        self.assertFalse(model.is_large_model)


if __name__ == "__main__":
    unittest.main()
