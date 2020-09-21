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


class FuseColumnWiseMulOptimizer(GraphRewriterBase):
    """Fuse Mul op into Conv2D/DepthwiseCond2dNative/MatMul
    Mul + Conv2D/DepthwiseCond2dNative/MatMul --> Conv2D/DepthwiseCond2dNative/MatMul
    """
    def __init__(self, model):
        super(FuseColumnWiseMulOptimizer, self).__init__(model)

    def do_transformation(self):
        cur_graph = TFGraphAnalyzer()
        cur_graph.graph = self.model

        graph_info = cur_graph.parse_graph()
        target_nodes = cur_graph.search_patterns([["Conv2D", "DepthwiseCond2dNative", "MatMul"],
                                                  "Mul"])

        for node_combination in target_nodes:
            upper_node = graph_info[node_combination[0]].node
            mul_node = graph_info[node_combination[1]].node
            weights_node = graph_info[graph_info[node_combination[0]].inputs[1]].node
            mul_value_node = graph_info[graph_info[node_combination[1]].inputs[1]].node
            upper_node_type = upper_node.op

            if upper_node_type == 'Conv2D':
                weights_col = weights_node.attr['value'].tensor.tensor_shape.dim[3].size
            elif upper_node_type == 'DepthwiseConv2dNative':
                weights_col = weights_node.attr['value'].tensor.tensor_shape.dim[2].size * \
                    weights_node.attr['value'].tensor.tensor_shape.dim[3].size
            else:
                weights_col = weights_node.attr['value'].tensor.tensor_shape.dim[1].size
            mul_value_node_tensor = mul_value_node.attr['value'].tensor
            weights_node_tensor = weights_node.attr['value'].tensor
            if len(mul_value_node_tensor.tensor_shape.dim
                   ) != 1 or mul_value_node_tensor.tensor_shape.dim[0].size != weights_col:
                self.logger.debug("Invalid Mul OP fusion.")
                return None

            mul_value_node_list = [i for i in tensor_util.MakeNdarray(mul_value_node_tensor).flat]
            new_weights = []
            for index, i in enumerate(tensor_util.MakeNdarray(weights_node_tensor).flat):
                new_weights_value = i * mul_value_node_list[index % len(mul_value_node_list)]
                new_weights.append(new_weights_value)

            weights_node.attr['value'].CopyFrom(
                attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
                    new_weights, dtypes.float32,
                    tensor_util.MakeNdarray(weights_node_tensor).shape)))

            cur_graph.remove_node_with_single_input_output(mul_node.name)
            cur_graph.remove_node(mul_node.input[1])

        return cur_graph.dump_graph()
