#
#  -*- coding: utf-8 -*-
#

from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import dtypes

from ..graph_base import GraphRewriterBase
from ..graph_util import TFGraphAnalyzer
from ..graph_util import TFGraphRewriterHelper as Helper

import numpy as np
import math


class FoldBatchNormNodesOptimizer(GraphRewriterBase):
    INPUT_ORDER = {
        # Order of inputs for BatchNormWithGlobalNormalization.
        "BatchNormWithGlobalNormalization":
        ["conv_op", "mean_op", "var_op", "beta_op", "gamma_op"],
        # Order of inputs for FusedBatchNorm.
        "FusedBatchNorm": ["conv_op", "gamma_op", "beta_op", "mean_op", "var_op"],
        "FusedBatchNormV3": ["conv_op", "gamma_op", "beta_op", "mean_op", "var_op"]
    }
    # Name of the attribute epsilon value is stored in.
    EPSILON_ATTR = {
        "BatchNormWithGlobalNormalization": "variance_epsilon",
        "FusedBatchNorm": "epsilon",
        "FusedBatchNormV3": "epsilon"
    }

    def __init__(self, model):
        super(FoldBatchNormNodesOptimizer, self).__init__(model)

    def scale_after_normalization(self, node):
        if node.op == "BatchNormWithGlobalNormalization":
            return node.attr["scale_after_normalization"].b
        return True

    def do_transformation(self):
        """Removes batch normalization ops by folding them into convolutions.

        Batch normalization during training has multiple dynamic parameters that are
        updated, but once the graph is finalized these become constants. That means
        there's an opportunity to reduce the computations down to a scale and
        addition, rather than the more expensive multiple ops, and even bake the
        scaling into the convolution weights. This function identifies the typical
        pattern of batch normalization subgraphs, and performs the transformation to
        fold the computations down into a simpler form. It currently only spots batch
        normalization that's performed by the BatchNormWithGlobalNormalization and
        FusedBatchNorm ops, and will need to be extended in the future to handle the
        newer style.

        Returns:
          Modified graph with BN ops removed, and modified weights.

        Raises:
          ValueError: If the graph is badly formed with duplicate node names.
        """
        cur_graph = TFGraphAnalyzer()
        cur_graph.graph = self.model

        graph_info = cur_graph.parse_graph()
        target_nodes = cur_graph.search_patterns(
            [["Conv2D", "DepthwiseConv2dNative"], ("BiasAdd", "Add", "AddV2"),
             ["BatchNormWithGlobalNormalization", "FusedBatchNorm", "FusedBatchNormV3"]])
        for node_combination in target_nodes:
            matched_node = node_combination[:-1]
            has_add_op = True if len(node_combination[-1]) == 3 else False
            conv_node = graph_info[Helper.node_name_from_input(matched_node[0])].node
            weights_node_name = graph_info[Helper.node_name_from_input(
                matched_node[0])].node.input[1]
            weights_node = graph_info[Helper.node_name_from_input(weights_node_name)].node
            bn_node = graph_info[Helper.node_name_from_input(matched_node[-1])].node

            if weights_node.op != "Const":
                self.logger.warning("Didn't find expected conv Constant input to '%s',"
                                    " found %s instead. Maybe because freeze_graph wasn't"
                                    " run first?" % (bn_node.name, weights_node_name))
                continue
            weights = Helper.values_from_const(weights_node)

            if conv_node.op == "Conv2D":
                channel_count = weights.shape[3]
            elif conv_node.op == "DepthwiseConv2dNative":
                channel_count = weights.shape[2] * weights.shape[3]

            mean_node_name = Helper.node_name_from_input(
                bn_node.input[self.INPUT_ORDER[bn_node.op].index("mean_op")])
            mean_node = graph_info[mean_node_name].node

            mean_value = Helper.values_from_const(mean_node)

            if has_add_op:
                bias_node_name = graph_info[Helper.node_name_from_input(
                    matched_node[1])].node.input[1]
                bias_node = graph_info[Helper.node_name_from_input(bias_node_name)].node
                mean_value = mean_value - Helper.values_from_const(bias_node)
                cur_graph.remove_node(bias_node.name)
                cur_graph.remove_node(matched_node[1])

            if mean_value.shape != (channel_count, ):
                self.logger.warning("Incorrect shape for mean, found %s, expected %s,"
                                    " for node %s" % (str(mean_value.shape), str(
                                        (channel_count, )), conv_node.name))
                continue
            var_node_name = Helper.node_name_from_input(
                bn_node.input[self.INPUT_ORDER[bn_node.op].index("var_op")])
            var_node = graph_info[var_node_name].node

            var_value = Helper.values_from_const(var_node)
            beta_node_name = Helper.node_name_from_input(
                bn_node.input[self.INPUT_ORDER[bn_node.op].index("beta_op")])
            beta_node = graph_info[beta_node_name].node

            beta_value = Helper.values_from_const(beta_node)
            gamma_node_name = Helper.node_name_from_input(
                bn_node.input[self.INPUT_ORDER[bn_node.op].index("gamma_op")])
            gamma_node = graph_info[gamma_node_name].node

            gamma_value = Helper.values_from_const(gamma_node)

            variance_epsilon_value = bn_node.attr[self.EPSILON_ATTR[bn_node.op]].f

            if self.scale_after_normalization(bn_node):
                scale_value = (
                    (1.0 / np.vectorize(math.sqrt)(var_value + variance_epsilon_value)) *
                    gamma_value)
            else:
                scale_value = (1.0 / np.vectorize(math.sqrt)(var_value + variance_epsilon_value))

            offset_value = (-mean_value * scale_value) + beta_value
            scaled_weights = np.copy(weights)

            it = np.nditer(scaled_weights, flags=["multi_index"], op_flags=["readwrite"])

            if conv_node.op == "Conv2D":
                while not it.finished:
                    current_scale = scale_value[it.multi_index[3]]
                    it[0] *= current_scale
                    it.iternext()
            elif conv_node.op == "DepthwiseConv2dNative":
                channel_multiplier = weights.shape[3]
                while not it.finished:
                    current_scale = scale_value[it.multi_index[2] * channel_multiplier +
                                                it.multi_index[3]]
                    it[0] *= current_scale
                    it.iternext()
            scaled_weights_node = node_def_pb2.NodeDef()
            scaled_weights_node.op = "Const"
            scaled_weights_node.name = weights_node_name + "_bn_offset"
            scaled_weights_node.attr["dtype"].CopyFrom(weights_node.attr["dtype"])
            scaled_weights_node.attr["value"].CopyFrom(
                attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
                    scaled_weights, weights.dtype.type, weights.shape)))
            cur_graph.replace_const_node(scaled_weights_node, [conv_node.name], weights_node_name)

            offset_node = node_def_pb2.NodeDef()
            offset_node.op = "Const"
            offset_node.name = conv_node.name + "_bn_offset"
            offset_node.attr["dtype"].CopyFrom(mean_node.attr["dtype"])
            offset_node.attr["value"].CopyFrom(
                attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
                    offset_value, mean_value.dtype.type, offset_value.shape)))
            bias_add_node = node_def_pb2.NodeDef()
            bias_add_node.op = "BiasAdd"
            bias_add_node.name = bn_node.name
            bias_add_node.attr["T"].CopyFrom(conv_node.attr["T"])
            bias_add_node.attr["data_format"].CopyFrom(conv_node.attr["data_format"])
            bias_add_node.input.extend([conv_node.name, offset_node.name])

            cur_graph.add_node(offset_node, [], [bias_add_node.name])
            cur_graph.add_node(bias_add_node, conv_node.name,
                               graph_info[Helper.node_name_from_input(matched_node[-1])].outputs)
            cur_graph.replace_const_node(scaled_weights_node, [conv_node.name], weights_node_name)

            cur_graph.remove_node(weights_node_name)
            cur_graph.remove_node(mean_node_name)
            cur_graph.remove_node(var_node_name)
            cur_graph.remove_node(beta_node_name)
            cur_graph.remove_node(gamma_node_name)

        return cur_graph.dump_graph()
