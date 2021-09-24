#
#  -*- coding: utf-8 -*-
#
import unittest
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes

from neural_compressor.adaptor.tf_utils.quantize_graph.quantize_graph_common import QuantizeGraphHelper

from neural_compressor.adaptor.tf_utils.graph_rewriter.graph_util import GraphAnalyzer


class TestGraphSearchPatterns(unittest.TestCase):

    def test_graph_search_partten_post_branch(self):
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()

        input_constant_name = "input_constant"
        relu_name = "relu"
        float_graph_def = graph_pb2.GraphDef()
        input_constant = QuantizeGraphHelper.create_constant_node(
            input_constant_name,
            value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            dtype=dtypes.float32,
            shape=[1, 2, 6, 1])
        float_graph_def.node.extend([input_constant])
        relu_node = QuantizeGraphHelper.create_node("Relu", relu_name, [input_constant_name])
        QuantizeGraphHelper.set_attr_dtype(relu_node, "T", dtypes.float32)
        float_graph_def.node.extend([relu_node])

        b_constant_name = "b_constant"
        mat_mul_name = "mat_mul"
        b_constant = QuantizeGraphHelper.create_constant_node(
            b_constant_name,
            value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            dtype=dtypes.float32,
            shape=[2, 6])
        float_graph_def.node.extend([b_constant])

        mat_mul_node = QuantizeGraphHelper.create_node("MatMul", mat_mul_name,
                                                       [relu_name, b_constant_name])
        QuantizeGraphHelper.set_attr_dtype(mat_mul_node, "T", dtypes.float32)
        QuantizeGraphHelper.set_attr_bool(mat_mul_node, "transpose_a", False)
        QuantizeGraphHelper.set_attr_bool(mat_mul_node, "transpose_b", False)
        float_graph_def.node.extend([mat_mul_node])

        bias_add_name = "bias_add"
        offset_constant_name = "offset_constant"

        offset_constant = QuantizeGraphHelper.create_constant_node(offset_constant_name,
                                                                   value=[1, 2, 3, 4, 5, 6],
                                                                   dtype=dtypes.float32,
                                                                   shape=[6])
        float_graph_def.node.extend([offset_constant])
        bias_add_node = QuantizeGraphHelper.create_node("BiasAdd", bias_add_name,
                                                        [mat_mul_name, offset_constant_name])
        QuantizeGraphHelper.set_attr_dtype(bias_add_node, "T", dtypes.float32)
        float_graph_def.node.extend([bias_add_node])

        post_relu_name = "post_relu"
        post_relu_node = QuantizeGraphHelper.create_node("Relu", post_relu_name, [bias_add_name])
        float_graph_def.node.extend([post_relu_node])

        last_identity_node_name = 'last_identity'
        last_identity_node = QuantizeGraphHelper.create_node("Identity", last_identity_node_name,
                                                             [post_relu_name])
        float_graph_def.node.extend([last_identity_node])

        last_identity_right_node_name = 'last_identity_right'
        last_identity_node_right = QuantizeGraphHelper.create_node("Identity", last_identity_right_node_name,
                                                                   [post_relu_name])
        float_graph_def.node.extend([last_identity_node_right])
        analyzer = GraphAnalyzer()
        analyzer.graph = float_graph_def
        analyzer.parse_graph()
        res = analyzer.query_fusion_pattern_nodes([['BiasAdd'], ("Relu"), ("Identity")])
        self.assertEqual(2, len(res))

    def test_graph_search_pattern_straight(self):
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()

        input_constant_name = "input_constant"
        relu_name = "relu"
        float_graph_def = graph_pb2.GraphDef()
        input_constant = QuantizeGraphHelper.create_constant_node(
            input_constant_name,
            value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            dtype=dtypes.float32,
            shape=[1, 2, 6, 1])
        float_graph_def.node.extend([input_constant])
        relu_node = QuantizeGraphHelper.create_node("Relu", relu_name, [input_constant_name])
        QuantizeGraphHelper.set_attr_dtype(relu_node, "T", dtypes.float32)
        float_graph_def.node.extend([relu_node])

        b_constant_name = "b_constant"
        mat_mul_name = "mat_mul"
        b_constant = QuantizeGraphHelper.create_constant_node(
            b_constant_name,
            value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            dtype=dtypes.float32,
            shape=[2, 6])
        float_graph_def.node.extend([b_constant])

        mat_mul_node = QuantizeGraphHelper.create_node("MatMul", mat_mul_name,
                                                       [relu_name, b_constant_name])
        QuantizeGraphHelper.set_attr_dtype(mat_mul_node, "T", dtypes.float32)
        QuantizeGraphHelper.set_attr_bool(mat_mul_node, "transpose_a", False)
        QuantizeGraphHelper.set_attr_bool(mat_mul_node, "transpose_b", False)
        float_graph_def.node.extend([mat_mul_node])

        bias_add_name = "bias_add"
        offset_constant_name = "offset_constant"

        offset_constant = QuantizeGraphHelper.create_constant_node(offset_constant_name,
                                                                   value=[1, 2, 3, 4, 5, 6],
                                                                   dtype=dtypes.float32,
                                                                   shape=[6])
        float_graph_def.node.extend([offset_constant])
        bias_add_node = QuantizeGraphHelper.create_node("BiasAdd", bias_add_name,
                                                        [mat_mul_name, offset_constant_name])
        QuantizeGraphHelper.set_attr_dtype(bias_add_node, "T", dtypes.float32)
        float_graph_def.node.extend([bias_add_node])

        post_relu_name = "post_relu"
        post_relu_node = QuantizeGraphHelper.create_node("Relu", post_relu_name, [bias_add_name])
        float_graph_def.node.extend([post_relu_node])

        last_identity_node_name = 'last_identity'
        last_identity_node = QuantizeGraphHelper.create_node("Identity", last_identity_node_name,
                                                             [post_relu_name])
        float_graph_def.node.extend([last_identity_node])

        analyzer = GraphAnalyzer()
        analyzer.graph = float_graph_def
        analyzer.parse_graph()
        res = analyzer.query_fusion_pattern_nodes([['MatMul'], ("BiasAdd"), ("Relu")])
        self.assertEqual(3, len(res[0][-1]))


if __name__ == '__main__':
    unittest.main()
