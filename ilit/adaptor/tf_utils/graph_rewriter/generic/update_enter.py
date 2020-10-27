#
#  -*- coding: utf-8 -*-
#


from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer


class UpdateEnterOptimizer(GraphRewriterBase):
    supported_ops = ["MatMul", "BiasAdd"]

    def __init__(self, model):
        super().__init__(model)
        self.graph_analyzer = GraphAnalyzer()
        self.graph_analyzer.graph = self.model
        self.exclude_node_names = []
        self.graph_info = self.graph_analyzer.parse_graph()

    def do_transformation(self):
        """ replace all enter ops whose output is matmul with const

        Args:
          input_graph_def (graphdef): graphdef object

        Returns:
           [graphdef]: optimized graph
        """
        for _, node_detail in self.graph_info.copy().items():
            if node_detail.outputs:
                if node_detail.node.op == "Enter" and self.graph_info[node_detail.outputs[0]
                                                      ].node.op in self.supported_ops:
                    self.exclude_node_names.append(node_detail.outputs[0])

        output_graph_def = self.graph_analyzer.dump_graph()

        return output_graph_def, self.exclude_node_names
