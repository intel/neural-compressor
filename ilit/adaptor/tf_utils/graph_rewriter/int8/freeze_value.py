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


class FreezeValueTransformer(GraphRewriterBase):
    def __init__(self, model, sampling_data, postfix, threshold=0.95):
        """Free Max/Min value into QuantizeV2 op.

        Args:
            model (graphdef): input model
            sampling_data (string list): the string context contains max/min values.
            postfix (string): the specified postfix to locate value.
            threshold (float, optional): The percentage of overall data.Defaults to 0.95.
        """
        super(FreezeValueTransformer, self).__init__(model)
        self.data = sampling_data
        self.threshold = threshold
        self.postfix = postfix
        self.cur_graph = TFGraphAnalyzer()
        self.cur_graph.graph = self.model

        self.graph_info = self.cur_graph.parse_graph()

    def _get_valid_log(self):
        output = []

        target_lines = [i.strip() for i in self.data if i.strip().find(';') != -1]
        for i in target_lines:
            semi_count = i.count(';')
            if semi_count == 2:
                output.append(i)
            elif semi_count % 2 != 0:
                self.logger.debug("Invalid line")
            else:
                loop_times = int(semi_count / 2)
                semi_index = [index for index, value in enumerate(i) if value == ";"]
                for index in range(loop_times - 1):
                    output.append(i[semi_index[index * 2]:semi_index[index * 2 + 2]])
                output.append(i[semi_index[loop_times * 2 - 2]:])
        return output

    def _parse_max_min_log(self):
        """
        Parse the max_ming log file
        :return: get the node name and value mapping
        """
        print_suffix = "__print__"
        lines = self._get_valid_log()

        res = {}
        temp = {}
        for i in lines:
            if i.find(print_suffix + ";" + self.postfix) == -1:
                continue
            max_line_data = i.split(';')
            name = max_line_data[1][:-len(print_suffix)]
            value = max_line_data[-1].split('[')[-1].split(']')[0]
            if "eightbit" in name and name not in temp:
                temp[name] = []
            if "eightbit" in name:
                temp[name].append(float(value))
        for key in temp:
            target_index = int(len(temp[key]) * self.threshold)
            if target_index > len(temp[key]) - 1:
                target_index = len(temp[key]) - 1
            res[key] = sorted(temp[key])[target_index]
        return res

    def generate_output_graph(self, max_name_value):
        """
        Generate transformed graph for freeze_max/freeze_min transformation.
        :param max_name_value: target values
        :return: transformed graph
        """
        for node_name, value in max_name_value.items():
            new_node = node_def_pb2.NodeDef()
            new_node.op = "Const"
            new_node_postfix = "/frozen_{}_only".format(''.join(
                [x for x in self.postfix if x.isalpha()]))
            new_node.name = node_name + new_node_postfix
            new_node.attr["dtype"].CopyFrom(
                attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
            new_node.attr["value"].CopyFrom(
                attr_value_pb2.AttrValue(
                    tensor=tensor_util.make_tensor_proto(float(value), dtypes.float32, [])))
            output_node_name = self.graph_info[node_name].outputs[0]
            self.cur_graph.replace_const_node(new_node,
                                              [Helper.node_name_from_input(output_node_name)],
                                              node_name)
            self.cur_graph.remove_node(node_name)

        return TFGraphAnalyzer().dump_graph()

    def do_transformation(self):
        max_name_value = self._parse_max_min_log()
        return self.generate_output_graph(max_name_value)
