#
#  -*- coding: utf-8 -*-
#
import unittest
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes

from neural_compressor.adaptor.tf_utils.quantize_graph.quantize_graph_common import QuantizeGraphHelper

from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.graph_cse_optimizer import GraphCseOptimizer
from neural_compressor.adaptor.tf_utils.util import disable_random


class TestGraphCommonSequenceElimated(unittest.TestCase):
    @disable_random()
    def test_graph_cse(self):

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
        identity_name = "identity"
        b_constant = QuantizeGraphHelper.create_constant_node(
            b_constant_name, value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=dtypes.float32, shape=[2, 6])
        float_graph_def.node.extend([b_constant])

        mat_mul_node = QuantizeGraphHelper.create_node("MatMul", mat_mul_name,
                                                       [relu_name, b_constant_name])
        QuantizeGraphHelper.set_attr_dtype(mat_mul_node, "T", dtypes.float32)
        QuantizeGraphHelper.set_attr_bool(mat_mul_node, "transpose_a", False)
        QuantizeGraphHelper.set_attr_bool(mat_mul_node, "transpose_b", False)
        float_graph_def.node.extend([mat_mul_node])

        identity_node = QuantizeGraphHelper.create_node("Identity", identity_name,
                                                        [mat_mul_name])
        float_graph_def.node.extend([identity_node])

        bias_add_name = "bias_add"
        offset_constant_name = "offset_constant"

        offset_constant = QuantizeGraphHelper.create_constant_node(
            offset_constant_name,
            value=[1, 2, 3, 4, 5, 6],
            dtype=dtypes.float32,
            shape=[6])
        float_graph_def.node.extend([offset_constant])
        bias_add_node = QuantizeGraphHelper.create_node(
            "BiasAdd", bias_add_name, [identity_name, offset_constant_name])
        QuantizeGraphHelper.set_attr_dtype(bias_add_node, "T", dtypes.float32)
        float_graph_def.node.extend([bias_add_node])

        post_relu_name = "post_relu"
        post_relu_node = QuantizeGraphHelper.create_node("Relu", post_relu_name,
                                                         [bias_add_name])
        float_graph_def.node.extend([post_relu_node])

        last_identity_node_name = 'last_identity'
        last_identity_node = QuantizeGraphHelper.create_node("Identity", last_identity_node_name,
                                                             [post_relu_name])
        float_graph_def.node.extend([last_identity_node])

        left_relu_name = "final_left_relu"
        left_relu_node = QuantizeGraphHelper.create_node("Relu", left_relu_name,
                                                         [last_identity_node_name])
        float_graph_def.node.extend([left_relu_node])
        right_relu_name = "final_right_relu"
        right_relu_node = QuantizeGraphHelper.create_node("Relu", right_relu_name,
                                                          [last_identity_node_name])
        float_graph_def.node.extend([right_relu_node])

        cse_left_node_name = "cse_left_node"
        cse_left_node = QuantizeGraphHelper.create_node("Identity", cse_left_node_name,
                                                        [left_relu_name])
        float_graph_def.node.extend([cse_left_node])

        cse_right_node_name = "cse_right_node"
        cse_right_node = QuantizeGraphHelper.create_node("Identity", cse_right_node_name,
                                                         [right_relu_name])
        float_graph_def.node.extend([cse_right_node])

        # post_graph = QuantizeGraphHelper().graph_cse_optimization (
        #     float_graph_def)
        post_graph = GraphCseOptimizer(float_graph_def).do_transformation()

        right_relu_optimized_flag = True
        for i in post_graph.node:
            if i.name == right_relu_name:
                right_relu_optimized_flag = False
                break

        self.assertEqual(right_relu_optimized_flag, True)


if __name__ == '__main__':
    unittest.main()
