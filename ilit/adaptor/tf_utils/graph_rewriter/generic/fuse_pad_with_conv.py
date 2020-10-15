#
#  -*- coding: utf-8 -*-
#

from tensorflow.python.framework import tensor_util

from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer
from ..graph_util import GraphRewriterHelper as Helper


class FusePadWithConv2DOptimizer(GraphRewriterBase):
    """Fuse Pad op into Conv2D
    Pad + Conv2D --> Conv2D
    """

    def do_transformation(self):
        cur_graph = GraphAnalyzer()
        cur_graph.graph = self.model

        graph_info = cur_graph.parse_graph()
        target_nodes = cur_graph.query_fusion_pattern_nodes([["Pad"], ["Conv2D"]])

        for node_combination in target_nodes:
            pad_node = graph_info[node_combination[0]].node
            padding_tensor = tensor_util.MakeNdarray(
                graph_info[pad_node.input[1]].node.attr["value"].tensor).flatten()

            cur_graph.remove_node_with_single_input_output(pad_node.name)
            cur_graph.remove_node(pad_node.input[1])
            conv_node = graph_info[node_combination[1]].node
            Helper.set_attr_int_list(conv_node, "padding_list", padding_tensor)

        return cur_graph.dump_graph()
