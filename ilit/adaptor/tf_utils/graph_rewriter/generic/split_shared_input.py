#
#  -*- coding: utf-8 -*-
#

from tensorflow.core.framework import node_def_pb2

from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer
from ..graph_util import GraphRewriterHelper as Helper


class SplitSharedInputOptimizer(GraphRewriterBase):

    def do_transformation(self):
        cur_graph = GraphAnalyzer()
        cur_graph.graph = self.model

        graph_info = cur_graph.parse_graph()

        is_shared_input = False
        # map of: input_name - op_name
        input_map = {}
        for node_name in list(graph_info.keys()):
            node = graph_info[node_name].node
            for _, input_node_name in enumerate(node.input):
                if graph_info[Helper.node_name_from_input(input_node_name)].node.op == 'Const':
                    # is shared and current node is not the first one
                    # sharing the input
                    if input_node_name in input_map:
                        is_shared_input = True
                        input_map[input_node_name].append(node.name)
                        new_input_node = node_def_pb2.NodeDef()
                        new_input_node.CopyFrom(graph_info[input_node_name].node)
                        new_input_node.name = input_node_name + '_ilit_share_' + str(
                            len(input_map[input_node_name]))
                        cur_graph.replace_const_node(
                            new_input_node, [node.name], input_node_name, False)
                    else:
                        input_map[input_node_name] = [node.name]

        return cur_graph.dump_graph() if is_shared_input else self.model
