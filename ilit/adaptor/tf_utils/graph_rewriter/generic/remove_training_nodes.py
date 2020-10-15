#
#  -*- coding: utf-8 -*-
#

from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer


class RemoveTrainingNodesOptimizer(GraphRewriterBase):
    def __init__(self, model, protected_nodes=[], types_to_splice=['Identity', 'CheckNumerics']):
        super().__init__(model)
        self.protected_nodes = protected_nodes
        self.types_to_splice = types_to_splice

    def do_transformation(self):
        graph_handle = GraphAnalyzer()
        graph_handle.graph = self.model

        graph_info = graph_handle.parse_graph()
        # input_nodes = input_graph.node

        control_input_names = set()
        node_names_with_control_input = set()
        names_to_splice = {}

        for node_name, v in graph_info.items():
            for node_input in v.node.input:
                if "^" in node_input:
                    control_input_names.add(node_input.replace("^", ""))
                    node_names_with_control_input.add(node_name)

        for node_name, v in graph_info.items():
            if v.node.op in self.types_to_splice and v.node.name not in self.protected_nodes:
                # We don't want to remove nodes that have control edge inputs, because
                # they might be involved in subtle dependency issues that removing them
                # will jeopardize.
                if node_name not in node_names_with_control_input:
                    names_to_splice[node_name] = v.node.input[0]

        # We also don't want to remove nodes which are used as control edge inputs.
        names_to_splice = {
            name: value
            for name, value in names_to_splice.items()
            if name not in control_input_names
        }
        for k, _ in names_to_splice.items():
            graph_handle.remove_node_with_single_input_output(k)

        return graph_handle.dump_graph()
