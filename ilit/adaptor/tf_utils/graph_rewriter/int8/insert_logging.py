#
#  -*- coding: utf-8 -*-
#

from tensorflow.python.framework import dtypes
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import node_def_pb2

from ..graph_base import GraphRewriterBase
from ..graph_util import TFGraphAnalyzer


class InsertLoggingTransformer(GraphRewriterBase):
    """Insert print op as the specified op type's successor.

    """

    def __init__(
        self,
        model,
        target_op_types,
        show_name=True,
        show_op=False,
        first_n=-1,
        summarize=1024,
        message="",
    ):
        """[summary]

        Args:
            model (graphdef): the input model graphdef object.
            target_op_types (string list): the specified op type that wiil insert print op
                                            as the successor.
            message (str, optional): The signature to locate the output data. Defaults to "".
        """
        super().__init__(model)
        self.target_op_types = target_op_types
        self.show_name = show_name
        self.show_op = show_op
        self.first_n = first_n
        self.summarize = summarize
        self.message = message
        self.cur_graph = TFGraphAnalyzer()
        self.node_info = self.cur_graph.parse_graph(self.model)

    def do_transformation(self):
        """
        Insert the Print op into the graph.
        """
        for node_name in list(self.node_info.keys()):
            details = self.node_info[node_name]
            op_type = details.node.op
            if self.target_op_types and op_type in self.target_op_types:
                name_suffix = "__print__"
                print_node = node_def_pb2.NodeDef()
                print_node.op = "Print"
                print_node.name = node_name + name_suffix
                node_message = ''
                if self.show_op:
                    node_message += ';' + self.node_mapping[node_name].op + ';'
                if self.show_name:
                    node_message += ';' + print_node.name + ';'
                node_message += self.message

                print_node.attr["message"].s = node_message.encode()
                print_node.attr["first_n"].i = self.first_n
                print_node.attr["summarize"].i = self.summarize

                print_node.input.append(node_name + ":0")
                print_node.attr["T"].CopyFrom(
                    attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))

                for index, _ in enumerate(self.node_info[node_name].outputs):
                    print_node.input.append(node_name + ':{}'.format(index))

                attr_u = [dtypes.float32.as_datatype_enum] * (len(
                    self.node_info[node_name].outputs))
                print_node.attr["U"].list.CopyFrom(attr_value_pb2.AttrValue.ListValue(type=attr_u))

                self.cur_graph.replace_const_node(print_node,
                                                  [self.node_info[node_name].outputs[0]],
                                                  node_name + ':0')

        return self.cur_graph.dump_graph()
