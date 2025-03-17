import os
import shutil
import sys
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.append("..")
from neural_compressor.adaptor.ox_utils.calibration import ONNXRTAugment
from neural_compressor.data import DATALOADERS, Datasets
from neural_compressor.data.datasets.dataset import Dataset
from neural_compressor.model.onnx_model import ONNXModel


def generate_input_initializer(tensor_shape, tensor_dtype, input_name):
    """Helper function to generate initializers for test inputs."""
    tensor = np.random.ranf(tensor_shape).astype(tensor_dtype)
    init = numpy_helper.from_array(tensor, input_name)
    return init


def create_cv_session():
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 1, 5, 5])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 1, 3, 3])
    b_value = np.random.randn(1, 1, 3, 3).astype(np.float32)
    B_init = helper.make_tensor("B", TensorProto.FLOAT, [1, 1, 3, 3], b_value.reshape(9).tolist())
    D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [1, 1, 5, 5])
    conv_node = onnx.helper.make_node("Conv", ["A", "B"], ["C"], name="conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    relu_node = onnx.helper.make_node("Relu", ["C"], ["D"], name="relu")
    graph = helper.make_graph([conv_node, relu_node], "test_graph_1", [A, B], [D], [B_init])
    model = helper.make_model(graph, **{"opset_imports": [helper.make_opsetid("", 13)]})

    dataset = TestDataset2()
    dataloader = DATALOADERS["onnxrt_qlinearops"](dataset)
    return model, dataloader


def create_nlp_session():
    a_value = np.random.randn(100, 4).astype(np.float32)
    A_init = helper.make_tensor("A", TensorProto.FLOAT, [100, 4], a_value.reshape(400).tolist())
    b_value = np.random.randint(2, size=(10)).astype(np.int32)
    B_init = helper.make_tensor("B", TensorProto.INT32, [10], b_value.reshape(10).tolist())
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 100, 4])
    D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [100, 4])
    squeeze = onnx.helper.make_node("Squeeze", ["A"], ["D"], name="squeeze")
    B = helper.make_tensor_value_info("B", TensorProto.INT32, [10])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [10, 4])
    node = onnx.helper.make_node("Gather", ["D", "B"], ["C"], name="gather")
    graph = helper.make_graph([squeeze, node], "test_graph_1", [A], [C], [B_init])
    model = helper.make_model(graph, **{"opset_imports": [helper.make_opsetid("", 13)]})
    datasets = Datasets("onnxrt_qlinearops")
    dataset = datasets["dummy_v2"](input_shape=(100, 4), label_shape=(1,))

    dataloader = DATALOADERS["onnxrt_qlinearops"](dataset)
    return model, dataloader


class TestDataset(Dataset):
    """Configuration for Imagenet dataset."""

    def __init__(self):
        data_list = []
        data_list.append(
            (np.array([[[[0.45, 0.60, 0.75]], [[0.25, 0.50, 0.75]], [[0.90, 0.70, 0.50]]]]).astype(np.float32), 0)
        )
        data_list.append(
            (np.array([[[[0.62, 0.94, 0.38]], [[0.70, 0.13, 0.07]], [[0.89, 0.75, 0.84]]]]).astype(np.float32), 0)
        )
        data_list.append(
            (np.array([[[[0.64, 0.24, 0.97]], [[0.82, 0.58, 0.27]], [[0.019, 0.34, 0.02]]]]).astype(np.float32), 0)
        )
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        return data


class TestDataset2(Dataset):
    """Configuration for Imagenet dataset."""

    def __init__(self):
        data_list = []
        data_list.append(np.random.random([1, 5, 5]).astype(np.float32))
        data_list.append(np.random.random([1, 5, 5]).astype(np.float32))
        data_list.append(np.random.random([1, 5, 5]).astype(np.float32))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        return data, 0


class TestAugment(unittest.TestCase):
    work_space = "./onnxrt_calib_test"

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.work_space)
        cls.cv_session = create_cv_session()
        cls.nlp_session = create_nlp_session()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work_space, ignore_errors=True)

    def test_dump_tensor(self):
        model, dataloader = self.cv_session
        augment = ONNXRTAugment(ONNXModel(model), dataloader, [], iterations=[0, 1], white_nodes=["conv"])
        map_dumped_tensors = augment.dump_tensor()
        self.assertTrue("conv" in map_dumped_tensors["activation"][0])
        self.assertTrue("A" in map_dumped_tensors["activation"][0]["conv"])
        self.assertTrue("conv" in map_dumped_tensors["activation"][1])
        self.assertTrue("A" in map_dumped_tensors["activation"][1]["conv"])

        model, dataloader = self.cv_session
        augment = ONNXRTAugment(ONNXModel(model), dataloader, [], iterations=[0], white_nodes=["conv", "relu"])
        map_dumped_tensors = augment.dump_tensor(weight=True)
        self.assertTrue("conv" in map_dumped_tensors["activation"][0])
        self.assertTrue("relu" in map_dumped_tensors["activation"][0])
        self.assertTrue("conv" in map_dumped_tensors["weight"])

        model, dataloader = self.nlp_session
        augment = ONNXRTAugment(ONNXModel(model), dataloader, [], iterations=[0], white_nodes=["gather"])
        map_dumped_tensors = augment.dump_tensor()
        self.assertTrue("gather" in map_dumped_tensors["activation"][0])

    def test_dump_calibration(self):
        model, dataloader = self.cv_session
        augment = ONNXRTAugment(ONNXModel(model), dataloader, ["Conv", "Relu"], iterations=[0])
        calib_params = augment.dump_calibration({})
        self.assertTrue("A" in calib_params and "B" in calib_params and "D" in calib_params and "C" in calib_params)

    def test_augment_graph(self):
        """TEST_CONFIG_1."""

        #     Conv
        #      |
        #     Clip
        #      |
        #     MatMul

        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 1, 5, 5])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 1, 3, 3])
        E = helper.make_tensor_value_info("E", TensorProto.FLOAT, [1, 1, 5, 1])
        F = helper.make_tensor_value_info("F", TensorProto.FLOAT, [1, 1, 5, 1])
        conv_node = onnx.helper.make_node(
            "Conv", ["A", "B"], ["C"], name="Conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        )
        clip_node = onnx.helper.make_node("Clip", ["C"], ["D"], name="Clip")
        matmul_node = onnx.helper.make_node("MatMul", ["D", "E"], ["F"], name="MatMul")
        graph = helper.make_graph([conv_node, clip_node, matmul_node], "test_graph_1", [A, B, E], [F])
        model = helper.make_model(graph)

        # Augmenting graph
        data_reader = None
        augment = ONNXRTAugment(ONNXModel(model), data_reader, ["Conv", "MatMul"])
        augment.augment_graph()
        augmented_model = augment.augmented_model

        # Checking if output exists
        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = ["Conv", "Clip", "MatMul"]
        added_outputs = ["A", "B", "C", "D", "E", "F"]
        # Original 3 nodes (exclude graph input/output)
        self.assertEqual(len(augmented_model_node_names), 3)
        # Original 1 graph output + 5 intermediate outputs
        self.assertEqual(len(augmented_model_outputs), 6)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)

        print("Finished TEST_CONFIG_1")
        """TEST_CONFIG_2."""

        #   Conv
        #    |
        #   Conv

        G = helper.make_tensor_value_info("G", TensorProto.FLOAT, [1, 1, 5, 5])
        H = helper.make_tensor_value_info("H", TensorProto.FLOAT, [1, 1, 3, 3])
        J = helper.make_tensor_value_info("J", TensorProto.FLOAT, [1, 1, 3, 3])
        K = helper.make_tensor_value_info("K", TensorProto.FLOAT, [1, 1, 5, 5])
        conv_node_1 = onnx.helper.make_node(
            "Conv", ["G", "H"], ["I"], name="Conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        )
        conv_node_2 = onnx.helper.make_node(
            "Conv", ["I", "J"], ["K"], name="Conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        )
        graph = helper.make_graph([conv_node_1, conv_node_2], "test_graph_2", [G, H, J], [K])
        model = helper.make_model(graph)

        # Augmenting graph
        data_reader = None
        augment = ONNXRTAugment(
            ONNXModel(model),
            data_reader,
            ["Conv", "MatMul"],
        )
        augment.augment_graph()
        augmented_model = augment.augmented_model

        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = ["Conv", "Conv"]
        added_outputs = ["I", "J", "H", "G", "K"]
        # Original 2 nodes
        self.assertEqual(len(augmented_model_node_names), 2)
        # Original 1 graph output + 4 intermediate outputs
        self.assertEqual(len(augmented_model_outputs), 5)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)

        print("Finished TEST_CONFIG_2")
        """TEST_CONFIG_3."""

        #   Relu
        #    |
        #   Conv  \
        #    |     |
        #   Clip   |
        #    |    /
        #   MatMul

        L = helper.make_tensor_value_info("L", TensorProto.FLOAT, [1, 1, 5, 5])
        N = helper.make_tensor_value_info("N", TensorProto.FLOAT, [1, 1, 3, 3])
        Q = helper.make_tensor_value_info("Q", TensorProto.FLOAT, [1, 1, 5, 5])
        relu_node = onnx.helper.make_node("Relu", ["L"], ["M"], name="Relu")
        conv_node = onnx.helper.make_node(
            "Conv", ["M", "N"], ["O"], name="Conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        )
        clip_node = onnx.helper.make_node("Clip", ["O"], ["P"], name="Clip")
        matmul_node = onnx.helper.make_node("MatMul", ["P", "M"], ["Q"], name="MatMul")
        graph = helper.make_graph([relu_node, conv_node, clip_node, matmul_node], "test_graph_3", [L, N], [Q])
        model = helper.make_model(graph)

        # Augmenting graph
        data_reader = None
        augment = ONNXRTAugment(ONNXModel(model), data_reader, ["Conv", "MatMul"])
        augment.augment_graph()
        augmented_model = augment.augmented_model

        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = ["Relu", "Conv", "Clip", "MatMul"]
        added_outputs = ["P", "M", "N", "O", "Q"]
        # Original 4 nodes
        self.assertEqual(len(augmented_model_node_names), 4)
        # Original 1 graph output + 4 intermediate outputs
        self.assertEqual(len(augmented_model_outputs), 5)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)

        print("Finished TEST_CONFIG_3")
        """TEST_CONFIG_4."""

        #   Attention
        #    |
        #   MatMul

        Attention_weight = helper.make_tensor_value_info("Attention_weight", TensorProto.FLOAT, [13, 7])
        Attention_bias = helper.make_tensor_value_info("Attention_bias", TensorProto.FLOAT, [13, 7])
        Attention_mask = helper.make_tensor_value_info("Attention_mask", TensorProto.INT32, [13, 7])
        S = helper.make_tensor_value_info("S", TensorProto.FLOAT, [13, 7])
        T = helper.make_tensor_value_info("T", TensorProto.FLOAT, [13, 7])
        attention_node = onnx.helper.make_node(
            "Attention", ["Attention_weight", "Attention_bias", "Attention_mask"], ["R"], name="Attention"
        )
        matmul_node = onnx.helper.make_node("MatMul", ["R", "S"], ["T"], name="MatMul")
        graph = helper.make_graph(
            [attention_node, matmul_node], "test_graph_4", [Attention_weight, Attention_bias, Attention_mask, S], [T]
        )
        model = helper.make_model(graph)

        # Augmenting graph
        data_reader = None
        augment = ONNXRTAugment(ONNXModel(model), data_reader, ["Conv", "MatMul", "Attention"])
        augment.augment_graph()
        augmented_model = augment.augmented_model

        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = ["Attention", "MatMul"]
        added_outputs = ["R", "Attention_mask", "S", "T", "Attention_bias", "Attention_weight"]
        # Original 2 nodes
        self.assertEqual(len(augmented_model_node_names), 2)
        # Original 1 graph output + 5 intermediate outputs
        self.assertEqual(len(augmented_model_outputs), 6)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)

        print("Finished TEST_CONFIG_4")

        #    QAttention
        #        |
        #    QuantizeLinear

        Attention_input = helper.make_tensor_value_info("input_quantized", TensorProto.INT8, [7, 13])
        Attention_weight = helper.make_tensor_value_info("weight_quantized", TensorProto.INT8, [13, 7])
        weight_quantized = generate_input_initializer([13, 7], np.int8, "weight_quantized")
        Attention_bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, [13, 7])
        bias = generate_input_initializer([13, 7], np.float32, "bias")
        Input_scale = helper.make_tensor_value_info("input_scale", TensorProto.FLOAT, [1])
        input_scale = generate_input_initializer([1], np.float32, "input_scale")
        Weight_scale = helper.make_tensor_value_info("weight_scale", TensorProto.FLOAT, [1])
        weight_scale = generate_input_initializer([1], np.float32, "weight_scale")
        Attention_mask = helper.make_tensor_value_info("mask", TensorProto.INT32, [13, 7])
        mask = generate_input_initializer([13, 7], np.int32, "mask")
        Input_zo = helper.make_tensor_value_info("input_zero_point", TensorProto.INT8, [1])
        input_zero_point = generate_input_initializer([1], np.int8, "input_zero_point")
        Weight_zo = helper.make_tensor_value_info("weight_zero_point", TensorProto.INT8, [1])
        weight_zero_point = generate_input_initializer([1], np.int8, "weight_zero_point")
        Q_scale = helper.make_tensor_value_info("attn_output_scale", TensorProto.FLOAT, [1])
        attn_output_scale = generate_input_initializer([1], np.float32, "attn_output_scale")
        Q_zo = helper.make_tensor_value_info("attn_output_zero_point", TensorProto.INT8, [1])
        attn_output_zero_point = generate_input_initializer([1], np.int8, "attn_output_zero_point")
        Output = helper.make_tensor_value_info("attn_output_quantized", TensorProto.INT8, [13, 7])
        attention_node = onnx.helper.make_node(
            "QAttention",
            [
                "input_quantized",
                "weight_quantized",
                "bias",
                "input_scale",
                "weight_scale",
                "mask",
                "input_zero_point",
                "weight_zero_point",
            ],
            ["attn_output"],
            name="attention_quant",
        )
        qlinear_node = onnx.helper.make_node(
            "QuantizeLinear",
            ["attn_output", "attn_output_scale", "attn_output_zero_point"],
            ["attn_output_quantized"],
            name="attn_output_QuantizeLinear",
        )
        graph = helper.make_graph(
            [attention_node, qlinear_node],
            "test_graph_5",
            [
                Attention_input,
                Attention_weight,
                Attention_bias,
                Input_scale,
                Weight_scale,
                Attention_mask,
                Input_zo,
                Weight_zo,
                Q_scale,
                Q_zo,
            ],
            [Output],
        )
        graph.initializer.add().CopyFrom(weight_quantized)
        graph.initializer.add().CopyFrom(bias)
        graph.initializer.add().CopyFrom(input_scale)
        graph.initializer.add().CopyFrom(weight_scale)
        graph.initializer.add().CopyFrom(mask)
        graph.initializer.add().CopyFrom(input_zero_point)
        graph.initializer.add().CopyFrom(weight_zero_point)
        graph.initializer.add().CopyFrom(attn_output_scale)
        graph.initializer.add().CopyFrom(attn_output_zero_point)
        model = helper.make_model(graph)

        # Augmenting graph
        data_reader = None
        augment = ONNXRTAugment(ONNXModel(model), data_reader, [], white_nodes=["attention"])
        augment.augment_nodes = ["DequantizeLinear"]
        augment.already_quantized = True

        augment.augment_graph()
        augmented_model = augment.augmented_model

        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = ["attention_quant", "attn_output_QuantizeLinear", "input_quantized_DequantizeLinear"]
        added_outputs = ["attn_output_quantized", "input_quantized_output", "attn_output"]
        self.assertEqual(len(augmented_model_node_names), 3)
        self.assertEqual(len(augmented_model_outputs), 3)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)

        print("Finished TEST_CONFIG_5")

        #    QuantizeLinear
        #        |
        #    QLinearConv
        #        |
        #    DequantizeLinear
        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 1, 5, 5])
        A_scale = helper.make_tensor_value_info("A_scale", TensorProto.FLOAT, [1])
        a_scale = generate_input_initializer([1], np.float32, "A_scale")
        A_zo = helper.make_tensor_value_info("A_zero_point", TensorProto.INT8, [1])
        a_zero_point = generate_input_initializer([1], np.int8, "A_zero_point")
        C = helper.make_tensor_value_info("C", TensorProto.INT8, [1, 1, 5, 5])
        c = generate_input_initializer([1, 1, 5, 5], np.int8, "C")
        C_scale = helper.make_tensor_value_info("C_scale", TensorProto.FLOAT, [1])
        c_scale = generate_input_initializer([1], np.float32, "C_scale")
        C_zo = helper.make_tensor_value_info("C_zero_point", TensorProto.INT8, [1])
        c_zero_point = generate_input_initializer([1], np.int8, "C_zero_point")
        E = helper.make_tensor_value_info("E", TensorProto.INT32, [1])
        e = generate_input_initializer([1], np.int32, "E")
        D_scale = helper.make_tensor_value_info("D_scale", TensorProto.FLOAT, [1])
        d_scale = generate_input_initializer([1], np.float32, "D_scale")
        D_zo = helper.make_tensor_value_info("D_zero_point", TensorProto.INT8, [1])
        d_zero_point = generate_input_initializer([1], np.int8, "D_zero_point")
        D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [1, 1, 5, 5])
        quantize_node = onnx.helper.make_node(
            "QuantizeLinear", ["A", "A_scale", "A_zero_point"], ["A_quantized"], name="A_QuantizeLinear"
        )
        conv_node = onnx.helper.make_node(
            "QLinearConv",
            [
                "A_quantized",
                "A_scale",
                "A_zero_point",
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
            "test_graph_5",
            [A, A_scale, A_zo, C, C_scale, C_zo, E, D_scale, D_zo],
            [D],
        )
        graph.initializer.add().CopyFrom(a_scale)
        graph.initializer.add().CopyFrom(a_zero_point)
        graph.initializer.add().CopyFrom(c)
        graph.initializer.add().CopyFrom(c_scale)
        graph.initializer.add().CopyFrom(c_zero_point)
        graph.initializer.add().CopyFrom(e)
        graph.initializer.add().CopyFrom(d_scale)
        graph.initializer.add().CopyFrom(d_zero_point)
        model = helper.make_model(graph)

        # Augmenting graph
        data_reader = None
        augment = ONNXRTAugment(ONNXModel(model), data_reader, [], white_nodes=["conv"])
        augment.augment_nodes = ["DequantizeLinear"]
        augment.already_quantized = True
        augment.augment_graph()
        augmented_model = augment.augmented_model

        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = [
            "A_QuantizeLinear",
            "conv_quant",
            "D_DequantizeLinear",
            "D_quantized_DequantizeLinear",
            "A_quantized_DequantizeLinear",
        ]
        added_outputs = ["D", "D_quantized_output", "A_quantized_output"]
        self.assertEqual(len(augmented_model_node_names), 5)
        self.assertEqual(len(augmented_model_outputs), 3)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)

    def test_quant_param_calculation(self):
        """TEST_CONFIG_6."""

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
            "test_graph_5",
            [input0],
            [output],
        )
        graph.initializer.add().CopyFrom(X1_weight)
        graph.initializer.add().CopyFrom(X1_bias)
        graph.initializer.add().CopyFrom(X3_weight)
        graph.initializer.add().CopyFrom(X3_bias)
        graph.initializer.add().CopyFrom(X5_weight)
        graph.initializer.add().CopyFrom(X5_bias)
        model = helper.make_model(graph, **{"opset_imports": [helper.make_opsetid("", 13)]})
        data_reader = TestDataset()
        augment = ONNXRTAugment(ONNXModel(model), data_reader, ["Conv", "MatMul"])

        # test calculation of quantization params
        # TO_DO: check rmin/rmax
        quantization_params_dict = augment.dump_calibration({})
        node_output_names, output_dicts_list = augment.get_intermediate_outputs({})
        dict_for_quantization = augment._map_calibration(node_output_names, output_dicts_list)
        # check the size of the quantization dictionary
        self.assertEqual(len(quantization_params_dict), 12)

        # check the computation of zp and scale
        for key, value in quantization_params_dict.items():
            self.assertTrue(value is not None)
            self.assertTrue(len(value) == 2)

            thresholds = dict_for_quantization[key]
            rmin = min(thresholds[0], 0)
            rmax = max(thresholds[1], 0)
            if key == "X2":  # next_node is Relu
                if rmin < 0:
                    rmin = 0

            scale_expected = np.float32((rmax - rmin) / 255 if rmin != rmax else 1)
            zp_expected = np.uint8(round(max(0, min(255, (0 - rmin) / scale_expected))))
            zp_actual = value[0]
            scale_actual = value[1]

            self.assertEqual(zp_expected, zp_actual)
            self.assertEqual(scale_expected, scale_actual)

        print("Finished" + " test calculation of quantization params.")


if __name__ == "__main__":
    unittest.main()
