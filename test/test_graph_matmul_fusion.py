#
#  -*- coding: utf-8 -*-
#
import unittest
import tensorflow as tf

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from ilit.adaptor.tf_utils.quantize_graph.quantize_graph_common import QuantizeGraphHelper
from ilit.adaptor.tf_utils.quantize_graph.quantize_graph_matmul import FuseNodeStartWithMatmul

class TestGraphMatMulFusion(unittest.TestCase):

    def test_matmul_biasadd_relu_fusion(self):
        tf.compat.v1.disable_eager_execution()

        input_constant_name = "input_constant"
        relu_name = "relu"
        float_graph_def = graph_pb2.GraphDef()
        input_constant = QuantizeGraphHelper.create_constant_node(
            input_constant_name,
            value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            dtype=dtypes.float32,
            shape=[1, 2, 6, 1])
        float_graph_def.node.extend([input_constant])
        relu_node = QuantizeGraphHelper.create_node("Relu", relu_name,
                                                [input_constant_name])
        QuantizeGraphHelper.set_attr_dtype(relu_node, "T", dtypes.float32)
        float_graph_def.node.extend([relu_node])

        b_constant_name = "b_constant"
        mat_mul_name = "mat_mul"

        b_constant = QuantizeGraphHelper.create_constant_node(
            b_constant_name, value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=dtypes.float32, shape=[2, 6])
        float_graph_def.node.extend([b_constant])

        mat_mul_node = QuantizeGraphHelper.create_node("MatMul", mat_mul_name,
                                                    [relu_name, b_constant_name])
        QuantizeGraphHelper.set_attr_dtype(mat_mul_node, "T", dtypes.float32)
        QuantizeGraphHelper.set_attr_bool(mat_mul_node, "transpose_a", False)
        QuantizeGraphHelper.set_attr_bool(mat_mul_node, "transpose_b", False)
        float_graph_def.node.extend([mat_mul_node])

        bias_add_name = "bias_add"
        offset_constant_name = "offset_constant"

        offset_constant = QuantizeGraphHelper.create_constant_node(
            offset_constant_name,
            value=[1, 2, 3, 4, 5, 6],
            dtype=dtypes.float32,
            shape=[6])
        float_graph_def.node.extend([offset_constant])
        bias_add_node = QuantizeGraphHelper.create_node(
            "BiasAdd", bias_add_name, [mat_mul_name, offset_constant_name])
        QuantizeGraphHelper.set_attr_dtype(bias_add_node, "T", dtypes.float32)
        float_graph_def.node.extend([bias_add_node])

        post_relu_name = "post_relu"
        post_relu_node = QuantizeGraphHelper.create_node("Relu", post_relu_name,
                                                        [bias_add_name])
        float_graph_def.node.extend([post_relu_node])

        worker = FuseNodeStartWithMatmul(
            float_graph_def, mat_mul_name, False, mat_mul_name)
        output_graph = worker.apply_the_transform()
        found_quantized_matmul = False
        for i in output_graph.node:
            if i.op == 'QuantizedMatMulWithBiasAndRelu':
                found_quantized_matmul = True
                break

        self.assertEqual(found_quantized_matmul, True)

if __name__ == '__main__':
    unittest.main()
