import copy
import os
import shutil
import unittest
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, onnx_pb

from neural_compressor.adaptor.ox_utils.quantizer import Quantizer
from neural_compressor.adaptor.ox_utils.util import QuantizationMode, QuantizedInitializer, QuantizedValue
from neural_compressor.config import ONNXQlinear2QDQConfig
from neural_compressor.model import Model

OPSET = onnx.OperatorSetIdProto()
OPSET.version = 17


def build_model():
    initializers = []
    input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 15, 15])
    output = helper.make_tensor_value_info("reshape_output", TensorProto.FLOAT, [88, 11])

    add_node = onnx.helper.make_node("Add", ["input", "add_init"], ["add_out"], name="add")

    conv1_weight_initializer = numpy_helper.from_array(
        np.random.randint(-1, 2, [3, 3, 3, 3]).astype(np.float32), name="conv1_weight"
    )
    conv1_node = helper.make_node("Conv", ["add_out", "conv1_weight"], ["conv1_output"], name="conv1")

    conv2_weight_initializer = numpy_helper.from_array(
        np.random.randint(-1, 2, [5, 3, 3, 3]).astype(np.float32), name="conv2_weight"
    )
    conv2_node = helper.make_node("Conv", ["add_out", "conv2_weight"], ["conv2_output"], name="conv2")

    # 1, 8, 13, 13
    concat_node = helper.make_node("Concat", ["conv1_output", "conv2_output"], ["concat_output"], name="Concat", axis=1)
    # 1, 8, 11, 11
    avg_args = {"kernel_shape": [3, 3]}
    avgpool_node = helper.make_node("AveragePool", ["concat_output"], ["avg_output"], name="AveragePool", **avg_args)
    reshape_node = onnx.helper.make_node("Reshape", ["avg_output", "shape"], ["reshape_output"], name="Reshape")

    initializers = [conv1_weight_initializer, conv2_weight_initializer]
    initializers.append(onnx.numpy_helper.from_array(np.array([88, 11], dtype=np.int64), name="shape"))
    initializers.append(onnx.numpy_helper.from_array(np.zeros((1, 3, 15, 15), dtype=np.float32), name="add_init"))
    graph = helper.make_graph(
        [conv1_node, conv2_node, concat_node, avgpool_node, reshape_node, add_node],
        "test",
        [input],
        [output],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model

class TestAdaptorONNXRT(unittest.TestCase):
    qlinear_backend = QuantizationMode.QLinearOps
    qdq_backend = "qdqops"
    integer_backend = QuantizationMode.IntegerOps
    static_q_config = {
        "weight": {"dtype": 3, "algorithm": "minmax", "scheme": "sym", "granularity": "per_tensor"},
        "activation": {
            "dtype": 2,
            "algorithm": "minmax",
            "scheme": "asym",
            "granularity": "per_tensor",
            "quant_mode": "static",
        },
    }
    dynamic_q_config = {
        "weight": {"dtype": 3, "algorithm": "minmax", "scheme": "sym", "granularity": "per_tensor"},
        "activation": {
            "dtype": 2,
            "algorithm": "minmax",
            "scheme": "asym",
            "granularity": "per_tensor",
            "quant_mode": "dynamic",
        },
    }
    config = ONNXQlinear2QDQConfig()

    @classmethod
    def setUpClass(cls):
        os.makedirs("./onnxrt_test")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("./onnxrt_test", ignore_errors=True)
        os.remove("test.onnx")

    def qlinear_test(self, model, q_config, quantize_params, quantizable_op_types, **kwargs):
        quantizer = Quantizer(
            copy.deepcopy(model), q_config, self.qlinear_backend, True, quantize_params, quantizable_op_types, **kwargs
        )
        model = quantizer.quantize_model()
        return Model(model)

    def dynamic_test(self, model, q_config, quantize_params, quantizable_op_types):
        quantizer = Quantizer(
            copy.deepcopy(model), q_config, self.integer_backend, False, quantize_params, quantizable_op_types
        )
        model = quantizer.quantize_model()
        return Model(model)

    def test_argmax(self):
        input_name = "input"
        output_name = "output"
        input_shape = [1, 256, 128, 128]
        output_shape = [1, 32, 128]
        initializers = []

        # make Conv node
        conv_weight_name = "conv_weight"
        conv_weight_arr = np.random.randint(-1, 2, [32, 256, 1, 1]).astype(np.float32)
        conv_weight_initializer = onnx.numpy_helper.from_array(conv_weight_arr, name=conv_weight_name)
        conv_output_name = "conv_output"
        conv_inputs = [input_name, conv_weight_name]
        conv_outputs = [conv_output_name]
        conv_name = "conv_node"
        conv_node = onnx.helper.make_node(
            "Conv",
            conv_inputs,
            conv_outputs,
            dilations=[1, 1],
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            name=conv_name,
        )

        # make ArgMax node
        argmax_inputs = [conv_output_name]
        argmax_outputs = [output_name]
        argmax_name = "argmax_node"
        argmax_node = onnx.helper.make_node(
            "ArgMax",
            argmax_inputs,
            argmax_outputs,
            axis=3,
            keepdims=0,
            name=argmax_name,
        )

        initializers = [conv_weight_initializer]

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.INT64, output_shape)
        graph_name = "ArgMax_Quant_Test"
        graph = helper.make_graph(
            [conv_node, argmax_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7  # use stable onnx ir version
        q_config = {"conv_node": self.static_q_config, "argmax_node": self.static_q_config}
        quantize_params = {
            "input": [np.uint8(0), np.float32(10.0)],
            "conv_weight": [np.uint8(0), np.float32(10.0)],
            "conv_output": [np.uint8(0), np.float32(10.0)],
            "output": [np.uint8(0), np.float32(10.0)],
        }
        q_model = self.qlinear_test(model, q_config, quantize_params, ["Conv", "ArgMax"])
        q_model.export("./test.onnx", self.config)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["DequantizeLinear"], 3)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["QuantizeLinear"], 2)



    def test_gemm(self):
        input_name = "input"
        output_name = "output"
        initializers = []
        weight_shape = [100, 10]
        weight_name = "linear1.weight"
        bias_shape = [100]
        bias_name = "linear1.bias"
        node_name = "gemm"

        weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(weight_data, name=weight_name))

        bias_data = np.random.normal(0, 0.1, bias_shape).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(bias_data, name=bias_name))

        gemm1_node = onnx.helper.make_node(
            "Gemm", [input_name, weight_name, bias_name], [output_name], alpha=1.0, beta=1.0, transB=1, name=node_name
        )

        gemm1_output_name = "gemm1_output"
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [-1, 10])
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [-1, 100])
        graph_name = "gemm_test"
        graph = helper.make_graph(
            [gemm1_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7  # use stable onnx ir version
        q_config = {"gemm": self.static_q_config}
        quantize_params = {
            "input": [np.uint8(0), np.float32(10.0)],
            "linear1.weight": [np.uint8(0), np.float32(10.0)],
            "linear1.bias": [np.uint8(0), np.float32(10.0)],
            "output": [np.uint8(0), np.float32(10.0)],
        }
        q_model = self.qlinear_test(model, q_config, quantize_params, ["Gemm"])
        q_model.export("./test.onnx", self.config)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["DequantizeLinear"], 3)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["QuantizeLinear"], 2)

        bias_tensor = helper.make_tensor_value_info(bias_name, TensorProto.FLOAT, [100])
        gemm2_node = onnx.helper.make_node(
            "Gemm", [input_name, weight_name, bias_name], [output_name], alpha=1.0, beta=1.0, transB=1, name=node_name
        )
        initializers = []
        initializers.append(onnx.numpy_helper.from_array(weight_data, name=weight_name))
        graph_name = "gemm_test"
        graph = helper.make_graph(
            [gemm2_node],
            graph_name,
            [input_tensor, bias_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7
        q_model = self.qlinear_test(model, q_config, quantize_params, ["Gemm"])
        q_model.export("./test.onnx", self.config)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["Gemm"], 1)

    def test_embed(self):
        input_ids_shape = [1, 4]
        input_ids_tensor = helper.make_tensor_value_info("input_ids", TensorProto.INT32, input_ids_shape)

        segment_ids_shape = [1, 4]
        segment_ids_tensor = helper.make_tensor_value_info("segment_ids", TensorProto.INT32, segment_ids_shape)

        mask_shape = [1, 4]
        mask_tensor = helper.make_tensor_value_info("mask", TensorProto.INT32, input_ids_shape)

        # EmbedLayerNormalization Node Constants and Weights:
        word_embed_shape = [32, 4]
        word_embed_weights = np.random.random_sample(word_embed_shape).astype(dtype="float32")
        word_embed_initializer = onnx.numpy_helper.from_array(word_embed_weights, name="word_embed")

        pos_embed_shape = [16, 4]
        pos_embed_weights = np.random.random_sample(pos_embed_shape).astype(dtype="float32")
        pos_embed_initializer = onnx.numpy_helper.from_array(pos_embed_weights, name="pos_embed")

        seg_embed_shape = [2, 4]
        seg_embed_weights = np.random.random_sample(seg_embed_shape).astype(dtype="float32")
        seg_embed_initializer = onnx.numpy_helper.from_array(seg_embed_weights, name="seg_embed")

        gamma_shape = [4]
        gamma = np.random.random_sample(gamma_shape).astype(dtype="float32")
        gamma_initializer = onnx.numpy_helper.from_array(gamma, name="gamma")

        beta_shape = [4]
        beta = np.random.random_sample(beta_shape).astype(dtype="float32")
        beta_initializer = onnx.numpy_helper.from_array(beta, name="beta")

        # EmbedLayerNormalization Outputs:
        layernorm_out_shape = [1, 4, 4]
        layernorm_out_tensor = helper.make_tensor_value_info("layernorm_out", TensorProto.FLOAT, layernorm_out_shape)

        mask_index_out_shape = [1]
        mask_index_out_tensor = helper.make_tensor_value_info("mask_index_out", TensorProto.INT32, mask_index_out_shape)

        # EmbedLayerNormalization Node:
        embed_layer_norm_inputs = [
            "input_ids",
            "segment_ids",
            "word_embed",
            "pos_embed",
            "seg_embed",
            "gamma",
            "beta",
            "mask",
        ]
        embed_layer_norm_outputs = ["layernorm_out", "mask_index_out"]
        embed_layer_norm_node = helper.make_node(
            "EmbedLayerNormalization",
            embed_layer_norm_inputs,
            embed_layer_norm_outputs,
            domain="com.microsoft",
            name="Embed",
        )

        # Construct the Graph and Model:
        nodes = [embed_layer_norm_node]
        graph_name = "embed_layernorm_graph"
        inputs = [input_ids_tensor, segment_ids_tensor, mask_tensor]
        outputs = [layernorm_out_tensor, mask_index_out_tensor]
        initializers = [
            word_embed_initializer,
            pos_embed_initializer,
            seg_embed_initializer,
            gamma_initializer,
            beta_initializer,
        ]

        graph = helper.make_graph(nodes, graph_name, inputs, outputs, initializer=initializers)
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("com.microsoft", 1), helper.make_opsetid("ai.onnx", 12)]
        )
        model.ir_version = 7  # use stable onnx ir version

        q_config = {"Embed": self.static_q_config}
        quantize_params = {
            "word_embed": [np.uint8(10.0), np.float32(0)],
            "pos_embed": [np.uint8(10.0), np.float32(0)],
            "seg_embed": [np.uint8(10.0), np.float32(0)],
            "gamma": [np.uint8(10.0), np.float32(0)],
            "beta": [np.uint8(10.0), np.float32(0)],
            "layernorm_out": [np.uint8(10.0), np.float32(0)],
            "mask_index_out": [np.uint8(10.0), np.float32(0)],
            "input_ids": [np.uint8(10.0), np.float32(0)],
        }
        q_model = self.qlinear_test(model, q_config, quantize_params, ["EmbedLayerNormalization"])
        q_model.export("./test.onnx", self.config)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["QEmbedLayerNormalization"], 1)

    def test_concat_reshape_pooling(self):
        model = build_model()
        q_config = {
            "Reshape": self.static_q_config,
            "conv1": self.static_q_config,
            "conv2": self.static_q_config,
            "Concat": self.static_q_config,
            "AveragePool": self.static_q_config,
            "add": self.static_q_config,
        }
        quantize_params = {
            "input": [np.uint8(10.0), np.float32(0)],
            "conv1_weight": [np.uint8(10.0), np.float32(0)],
            "conv1_output": [np.uint8(10.0), np.float32(0)],
            "conv2_weight": [np.uint8(10.0), np.float32(0)],
            "conv2_output": [np.uint8(10.0), np.float32(0)],
            "concat_output": [np.uint8(10.0), np.float32(0)],
            "avg_output": [np.uint8(10.0), np.float32(0)],
            "add_out": [np.uint8(10.0), np.float32(0)],
            "add_init": [np.uint8(10.0), np.float32(0)],
            "shape": [np.uint8(10.0), np.float32(0)],
            "reshape_output": [np.uint8(10.0), np.float32(0)],
        }
        quantizable_op_types = ["Reshape", "Conv", "Concat", "AveragePool", "Add"]
        q_model = self.qlinear_test(
            model, q_config, quantize_params, quantizable_op_types, **{"dedicated_qdq_pair": True}
        )
        q_model.export("./test.onnx", self.config)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["DequantizeLinear"], 8)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["QuantizeLinear"], 5)

        q_config = {
            "Reshape": self.static_q_config,
            "conv1": "fp32",
            "conv2": self.static_q_config,
            "Concat": self.static_q_config,
            "AveragePool": self.static_q_config,
        }
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        q_model.export("./test.onnx", self.config)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["DequantizeLinear"], 3)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["QuantizeLinear"], 2)

        q_config = {
            "Reshape": self.static_q_config,
            "conv1": "fp32",
            "conv2": "fp32",
            "Concat": self.static_q_config,
            "AveragePool": self.static_q_config,
        }
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        q_model.export("./test.onnx", self.config)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["DequantizeLinear"], 0)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["QuantizeLinear"], 0)

        q_config = {
            "Reshape": self.static_q_config,
            "conv1": self.static_q_config,
            "conv2": self.static_q_config,
            "Concat": self.static_q_config,
            "AveragePool": "fp32",
        }
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        q_model.export("./test.onnx", self.config)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["DequantizeLinear"], 7)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["QuantizeLinear"], 4)
        

        quantize_params = {
            "input": [np.uint8(10.0), np.float32(0)],
            "conv1_weight": [np.uint8(10.0), np.float32(0)],
            "conv1_output": [np.uint8(10.0), np.float32(0)],
            "conv2_weight": [np.uint8(10.0), np.float32(0)],
            "conv2_output": [np.uint8(10.0), np.float32(0)],
            "concat_output": [np.uint8(10.0), np.float32(0)],
            "avg_output": [np.uint8(10.0), np.float32(0)],
            "shape": [np.uint8(10.0), np.float32(0)],
            "add_out": [np.uint8(10.0), np.float32(0)],
            "add_init": [np.uint8(10.0), np.float32(0)],
            "reshape_output": [np.uint8(10.0), np.float32(0)],
        }
        q_config = {
            "Reshape": self.static_q_config,
            "conv1": self.static_q_config,
            "conv2": self.static_q_config,
            "Concat": self.static_q_config,
            "AveragePool": self.static_q_config,
        }
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        q_model.export("./test.onnx", self.config)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["DequantizeLinear"], 8)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["QuantizeLinear"], 5)

    def test_conv(self):
        op = "Conv"
        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 5, 5, 1])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 3, 3, 1])
        C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 1, 5, 1])
        conv_node = onnx.helper.make_node(
            op, ["A", "B"], ["C"], name=op, kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        )
        graph = helper.make_graph([conv_node], "test_graph_1", [A, B], [C])
        model = helper.make_model(graph, opset_imports=[OPSET])
        q_config = {op: self.static_q_config}
        quantize_params = {
            "A": [np.uint8(10.0), np.float32(0)],
            "B": [np.uint8(10.0), np.float32(0)],
            "C": [np.uint8(10.0), np.float32(0)],
        }
        quantizable_op_types = [op]
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        q_model.export("./test.onnx", self.config)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["DequantizeLinear"], 3)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["QuantizeLinear"], 3)

    def test_matmul(self):
        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 1, 5, 5])
        B_init = helper.make_tensor(
            "B", TensorProto.FLOAT, [1, 1, 5, 1], np.random.random((1, 1, 5, 1)).reshape(5).tolist()
        )
        C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 1, 5, 1])
        matmul_node = onnx.helper.make_node("MatMul", ["A", "B"], ["C"], name="Matmul")
        graph = helper.make_graph([matmul_node], "test_graph_1", [A], [C], [B_init])
        model = helper.make_model(graph, opset_imports=[OPSET])
        q_config = {"Matmul": self.static_q_config}
        quantize_params = {
            "A": [np.uint8(10.0), np.float32(0)],
            "B": [np.uint8(10.0), np.float32(0)],
            "C": [np.uint8(10.0), np.float32(0)],
        }
        quantizable_op_types = ["Matmul"]
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        q_model.export("./test.onnx", self.config)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["DequantizeLinear"], 3)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["QuantizeLinear"], 2)

        q_config = {"Matmul": self.dynamic_q_config}
        q_model = self.dynamic_test(model, q_config, None, quantizable_op_types)
        q_model.export("./test.onnx", self.config)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["MatMulInteger"], 1)

        q_config = {"Matmul": self.dynamic_q_config}
        quantize_params = {}
        q_model = self.dynamic_test(model, q_config, quantize_params, quantizable_op_types)
        q_model.export("./test.onnx", self.config)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["MatMulInteger"], 1)

    def test_attention(self):
        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 128, 768])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [768, 2304])
        C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2304])
        D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [1, 128, 768])
        mask = helper.make_tensor_value_info("mask", TensorProto.INT32, [1, 128])

        node = onnx.helper.make_node("Attention", ["A", "B", "C", "mask"], ["D"], name="Attention", num_heads=1)
        graph = helper.make_graph([node], "test_graph_1", [A, B, C, mask], [D])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        q_config = {"Attention": self.static_q_config}
        quantize_params = {
            "A": [np.uint8(0), np.float32(0.5)],
            "B": [np.uint8(0), np.float32(0.5)],
            "C": [np.uint8(0), np.float32(0.5)],
            "D": [np.uint8(0), np.float32(0.5)],
        }
        quantizable_op_types = ["Attention"]
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        q_model.export("./test.onnx", self.config)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["DequantizeLinear"], 1)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["QuantizeLinear"], 3)

    def test_gather(self):
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 2])
        matmul_weight = helper.make_tensor(
            "matmul_weight", TensorProto.FLOAT, [2, 2], np.random.random((2, 2)).reshape(4).tolist()
        )
        matmul_output = helper.make_tensor_value_info("matmul_output", TensorProto.FLOAT, [2, 2])
        matmul_node = onnx.helper.make_node("MatMul", ["input", "matmul_weight"], ["matmul_output"], name="MatMul")

        gather_indices = helper.make_tensor("gather_indices", TensorProto.INT32, [2], [0, 0])
        gather_output = helper.make_tensor_value_info("gather_output", TensorProto.FLOAT, [2, 2])
        gather_node = onnx.helper.make_node("Gather", ["matmul_output", "gather_indices"], ["gather_output"], name="Gather")

        graph = helper.make_graph(
            [matmul_node, gather_node], 
            "test_graph", 
            [input_tensor], 
            [gather_output],
            initializer=[matmul_weight, gather_indices])
        model = helper.make_model(graph, opset_imports=[OPSET])

        q_config = {"Gather": self.static_q_config, "MatMul": self.static_q_config}
        quantize_params = {
            "input": [np.uint8(10.0), np.float32(0)],
            "matmul_weight": [np.uint8(10.0), np.float32(0)],
            "matmul_output": [np.uint8(10.0), np.float32(0)],
            "gather_output": [np.uint8(10.0), np.float32(0)],
        }
        quantizable_op_types = ["Gather", "MatMul"]
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        q_model.export("./test.onnx", self.config)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["DequantizeLinear"], 3)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["QuantizeLinear"], 2)

    def test_binary(self):
        for op in ["Mul", "Add"]:
            input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3])
            matmul_weight = helper.make_tensor(
                "matmul_weight", TensorProto.FLOAT, [3, 2], np.random.random((3, 2)).reshape(6).tolist()
            )
            matmul_output = helper.make_tensor_value_info("matmul_output", TensorProto.FLOAT, [2, 2])
            matmul_node = onnx.helper.make_node("MatMul", ["input", "matmul_weight"], ["matmul_output"], name="Matmul")

            op_weight = helper.make_tensor(
                op + "_weight", TensorProto.FLOAT, [2, 2], np.random.random((2, 2)).reshape(4).tolist()
            )
            op_output = helper.make_tensor_value_info(op + "_output", TensorProto.FLOAT, [2, 2])
            op_node = onnx.helper.make_node(op, ["matmul_output", op + "_weight"], [op + "_output"], name=op)

            graph = helper.make_graph(
                [matmul_node, op_node], 
                "test_graph", 
                [input_tensor], 
                [op_output],
                initializer=[matmul_weight, op_weight])
            model = helper.make_model(graph, opset_imports=[OPSET])

            q_config = {op: self.static_q_config, 
                        "Matmul": self.static_q_config,}
            quantize_params = {
                "input": [np.uint8(10.0), np.float32(0)],
                "matmul_weight": [np.uint8(10.0), np.float32(0)],
                "matmul_output": [np.uint8(10.0), np.float32(0)],
                op + "_weight": [np.uint8(10.0), np.float32(0)],
                op + "_output": [np.uint8(10.0), np.float32(0)],
            }
            quantizable_op_types = [op]
            q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
            q_model.export("./test.onnx", self.config)
            self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["DequantizeLinear"], 5)
            self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["QuantizeLinear"], 3)


    def test_activation(self):
        config = {
            "weight": {"dtype": 2, "algorithm": "minmax", "scheme": "asym", "granularity": "per_tensor"},
            "activation": {
                "dtype": 2,
                "algorithm": "minmax",
                "scheme": "asym",
                "granularity": "per_tensor",
                "quant_mode": "static",
            },
        }

        for op in ["Relu", "LeakyRelu", "Sigmoid"]:
            B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 10])
            A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 10])
            node = onnx.helper.make_node(op, ["A"], ["B"], name=op)
            graph = helper.make_graph([node], "test_graph_1", [A], [B])
            model = helper.make_model(graph, opset_imports=[OPSET])
            q_config = {op: config}
            quantize_params = {"A": [np.uint8(10.0), np.float32(0)], "B": [np.uint8(10.0), np.float32(0)]}
            quantizable_op_types = [op]
            q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
            q_model.export("./test.onnx", self.config)

            a_value = np.random.randn(1, 10).astype(np.float32)
            A_init = helper.make_tensor("A", TensorProto.FLOAT, [1, 10], a_value.reshape(10).tolist())
            graph = helper.make_graph([node], "test_graph_1", [A], [B], [A_init])
            model = helper.make_model(graph, opset_imports=[OPSET])
            q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
            q_model.export("./test.onnx", self.config)

    def test_pooling(self):
        op = "MaxPool"
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 5, 5, 1])
        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 5, 5, 1])
        node = onnx.helper.make_node(op, ["A"], ["B"], name=op, kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        graph = helper.make_graph([node], "test_graph_1", [A], [B])
        q_config = {op: self.static_q_config}
        quantize_params = {"A": [np.uint8(10.0), np.float32(0)], "B": [np.uint8(10.0), np.float32(0)]}
        quantizable_op_types = [op]
        for opset_version in [12, 13]:
            opset = onnx.OperatorSetIdProto()
            opset.version = opset_version
            model = helper.make_model(graph, opset_imports=[opset])
            q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
            q_model.export("./test.onnx", self.config)

        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 1, 5, 5])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 1, 3, 3])
        D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [1, 1, 5, 5])
        conv_node = onnx.helper.make_node(
            "Conv", ["A", "B"], ["C"], name="Conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        )
        pool_node = onnx.helper.make_node(op, ["C"], ["D"], name=op, kernel_shape=[1, 1])
        graph = helper.make_graph([conv_node, pool_node], "test_graph_1", [A, B], [D])
        model = helper.make_model(graph, opset_imports=[OPSET])

        q_config = {"Conv": self.static_q_config, op: self.static_q_config}
        quantize_params = {
            "A": [np.uint8(10.0), np.float32(0)],
            "B": [np.uint8(10.0), np.float32(0)],
            "C": [np.uint8(10.0), np.float32(0)],
            "D": [np.uint8(10.0), np.float32(0)],
        }
        quantizable_op_types = ["Conv", op]
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        q_model.export("./test.onnx", self.config)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["DequantizeLinear"], 3)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["QuantizeLinear"], 3)

        op = "GlobalAveragePool"
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 5, 1, 1])
        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 5, 5, 1])
        node = onnx.helper.make_node(op, ["A"], ["B"], name=op)
        graph = helper.make_graph([node], "test_graph_1", [A], [B])
        q_config = {op: self.static_q_config}
        quantize_params = {"A": [np.uint8(10.0), np.float32(0)], "B": [np.uint8(10.0), np.float32(0)]}
        quantizable_op_types = [op]
        for opset_version in [12, 13]:
            opset = onnx.OperatorSetIdProto()
            opset.version = opset_version
            model = helper.make_model(graph, opset_imports=[opset])
            q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
            q_model.export("./test.onnx", self.config)
            self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["DequantizeLinear"], 2)
            self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["QuantizeLinear"], 2)

        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 1, 5, 5])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 1, 3, 3])
        D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [1, 1, 1, 1])
        conv_node = onnx.helper.make_node(
            "Conv", ["A", "B"], ["C"], name="Conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        )
        pool_node = onnx.helper.make_node(op, ["C"], ["D"], name=op)
        graph = helper.make_graph([conv_node, pool_node], "test_graph_1", [A, B], [D])
        model = helper.make_model(graph, opset_imports=[OPSET])

        q_config = {"Conv": self.static_q_config, op: self.static_q_config}
        quantize_params = {
            "A": [np.uint8(10.0), np.float32(0)],
            "B": [np.uint8(10.0), np.float32(0)],
            "C": [np.uint8(10.0), np.float32(0)],
            "D": [np.uint8(10.0), np.float32(0)],
        }
        quantizable_op_types = ["Conv", op]
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        q_model.export("./test.onnx", self.config)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["DequantizeLinear"], 4)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["QuantizeLinear"], 4)

    def test_exclude_node(self):
        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 5, 5, 1])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 3, 1, 1])
        D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [1, 3, 5, 1])
        conv_node = onnx.helper.make_node(
            "Conv", ["A", "B"], ["C"], name="Conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        )
        pool_node = onnx.helper.make_node("MaxPool", ["C"], ["D"], name="MaxPool", kernel_shape=[1, 1])
        graph = helper.make_graph([conv_node, pool_node], "test_graph_1", [A, B], [D])
        model = helper.make_model(graph, opset_imports=[OPSET])

        q_config = {"Conv": self.static_q_config, "MaxPool": "fp32"}
        quantize_params = {
            "A": [np.uint8(10.0), np.float32(0)],
            "B": [np.uint8(10.0), np.float32(0)],
            "C": [np.uint8(10.0), np.float32(0)],
            "D": [np.uint8(10.0), np.float32(0)],
        }
        quantizable_op_types = ["Conv", "MaxPool"]
        self.config.exclude_output_quantization = ["Conv"]
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        q_model.export("./test.onnx", self.config)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["DequantizeLinear"], 3)
        self.assertEqual(Counter([node.op_type for node in q_model.graph().node])["QuantizeLinear"], 3)


if __name__ == "__main__":
    unittest.main()
