#
#  -*- coding: utf-8 -*-
#

from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2

from ..graph_base import GraphRewriterBase
from ..graph_util import TFGraphAnalyzer
from ..graph_util import TFGraphRewriterHelper as Helper


class StripUnusedNodesOptimizer(GraphRewriterBase):
    def __init__(self, model, input_node_names, output_node_names):
        super(StripUnusedNodesOptimizer, self).__init__(model)
        self.input_node_names = input_node_names
        self.output_node_names = output_node_names

    def do_transformation(self):
        cur_graph = TFGraphAnalyzer()
        cur_graph.graph = self.model

        graph_info = cur_graph.parse_graph()

        for name in self.input_node_names:
            if ':' in name:
                self.logger.debug("Name {} appears to refer to a Tensor, "
                                  "not a Operation.".format(name))
                return False

        type_attr={"Sub":"T"}

        not_found = {name for name in self.input_node_names}
        for node_name, _ in graph_info.items():
            if node_name in not_found:
                not_found.remove(node_name)
                node = graph_info[node_name].node
                original_output = graph_info[node_name].outputs
                placeholder_node = node_def_pb2.NodeDef()
                placeholder_node.op = "Placeholder"
                placeholder_node.name = node.name

                if "dtype" in node.attr:
                    placeholder_node.attr["dtype"].CopyFrom(
                        attr_value_pb2.AttrValue(type=node.attr["dtype"].type))
                elif node.op in type_attr.keys():
                    placeholder_node.attr["dtype"].CopyFrom(
                        attr_value_pb2.AttrValue(type=node.attr[type_attr[node.op]].type))
                else:
                    raise KeyError("%s op's type attribute is not found,"
                        "you should add it to type_attr dict"%node.op)
                if "_output_shapes" in node.attr:
                    placeholder_node.attr["_output_shapes"].CopyFrom(node.attr["_output_shapes"])
                if "shape" in node.attr:
                    placeholder_node.attr["shape"].CopyFrom(node.attr["shape"])

                cur_graph.remove_node(node_name)

                cur_graph.replace_const_node(placeholder_node, [node_name], original_output)

        import tensorflow as tf
        return tf.compat.v1.graph_util.extract_sub_graph(cur_graph.dump_graph(),
                                                         self.output_node_names)
