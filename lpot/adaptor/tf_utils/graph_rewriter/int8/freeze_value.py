#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import dtypes

from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer
from ..graph_util import GraphRewriterHelper as Helper

import numpy as np
import re

class FreezeValueTransformer(GraphRewriterBase):
    def __init__(self, model, max_min_data, postfix, tensor_data=None, th=0.95, device='gpu'):
        """Free Max/Min value into QuantizeV2 op.

        Args:
            model (graphdef): input model
            max_min_data (string list): the string context contains max/min values.
            postfix (string): the specified postfix to locate value.
            tensor_data(dict): key is the op name while the value is the max/min values
                                which calculated by KL.
            th (float, optional): The percentage of overall data.Defaults to 0.95.
            device (string, optional): The hardware device type, 'cpu' or 'gpu'.
        """
        super().__init__(model)
        self.data = max_min_data
        if 0.0 < th <= 1.0:
            self.threshold = th
        else:
            self.logger.warning("The threshold value for clipping is invalid ," \
                "reset it to 0.95 by default")
            self.threshold = 0.95
        self.postfix = postfix
        self.device = device
        self.tensor_data = tensor_data
        self.cur_graph = GraphAnalyzer()
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
        pattern_def = r"{};{}\[\-?\d+\.?\d*e?\+?\d*\]".format(print_suffix, self.postfix)
        for i in lines:

            if not re.search(pattern_def, i):
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

    def _parse_requantization_ranges(self):
        """
        Parse the max_min log to get requantization values
        :return: dict saved the result
        """
        res = {}

        print_suffix = "__print__"
        lines = self._get_valid_log()
        temp_min = {}
        temp_max = {}
        pattern_def = r"{};{}:\[\-?\d+\.?\d*e?-?\+?\d*\]".format(print_suffix, self.postfix)
        for i in lines:
            if not re.search(pattern_def, i):
                continue

            max_line_data = i.split(print_suffix + ";" + self.postfix)[-1]
            min_value = max_line_data.split('][')[0].split('[')[1]
            max_value = max_line_data.split('][')[1].split(']')[0]
            name = i.split(';')[1].strip()[:-len(print_suffix)]
            if name not in temp_min:
                temp_min[name] = []
            if name not in temp_max:
                temp_max[name] = []

            temp_min[name].append(float(min_value))
            temp_max[name].append(float(max_value))

        for key in temp_min:
            target_min_index = int(np.ceil(len(temp_min[key]) * (1 - self.threshold)))

            if key not in res:
                res[key] = []

            if target_min_index > len(temp_min[key]) - 1:
                target_min_index = len(temp_min[key]) - 1
            res[key].append(sorted(temp_min[key])[target_min_index])

        for key in temp_max:
            target_max_index = int(np.floor(len(temp_max[key]) * self.threshold)) - 1

            if target_max_index > len(temp_max[key]) - 1:
                target_max_index = len(temp_max[key]) - 1

            res[key].append(sorted(temp_max[key])[target_max_index])

        if self.tensor_data:
            for k, v in self.tensor_data.items():
                if k in res:
                    self.logger.debug('Update node {} min to {}, max to {}'.format(k, v[2], v[3]))
                    res[k] = [v[2], v[3]]
        return res

    def generate_output_graph(self, max_name_value):
        """
        Generate transformed graph for freeze_max/freeze_min transformation.
        :param max_name_value: target values
        :return: transformed graph
        """
        for node_name, value in max_name_value.items():
            node_name = node_name.replace(":", "__port__").replace("^", "__hat__")
            if node_name not in self.graph_info:
                continue
            new_node = node_def_pb2.NodeDef()
            new_node.op = "Const"
            new_node_postfix = "/frozen_{}_only".format(''.join(
                [x for x in self.postfix if x.isalpha()]))
            new_node.name = node_name + new_node_postfix
            new_node.attr["dtype"].CopyFrom(
                attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
            new_node.attr["value"].CopyFrom(
                attr_value_pb2.AttrValue(
                    tensor=tensor_util.make_tensor_proto(float(value),
                    dtypes.float32, [])))
            output_node_name = self.graph_info[node_name].outputs[0]
            self.cur_graph.replace_const_node(new_node,
                                              [Helper.node_name_from_input(output_node_name)],
                                              node_name)
            self.cur_graph.remove_node(node_name)

        return GraphAnalyzer().dump_graph()

    def generate_output_graph_ranges(self, max_name_value):
        """
        Generate transformed graph for freeze_max/freeze_min transformation.
        :param max_name_value: target values
        :return: transformed graph
        """
        for node_name, value in max_name_value.items():
            if node_name not in self.graph_info:
                continue

            min_node = node_def_pb2.NodeDef()
            min_node.op = "HostConst" if self.device == "gpu" else "Const"
            min_node_postfix = "/frozen_min"
            min_node.name = node_name + min_node_postfix
            min_node.attr["dtype"].CopyFrom(
                attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
            min_node.attr["value"].CopyFrom(
                attr_value_pb2.AttrValue(
                    tensor=tensor_util.make_tensor_proto(float(value[0]),
                    dtypes.float32, [])))

            max_node = node_def_pb2.NodeDef()
            max_node.op = "HostConst" if self.device == "gpu" else "Const"
            max_node_postfix = "/frozen_max"
            max_node.name = node_name + max_node_postfix
            max_node.attr["dtype"].CopyFrom(
                attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
            max_node.attr["value"].CopyFrom(
                attr_value_pb2.AttrValue(
                    tensor=tensor_util.make_tensor_proto(float(value[1]),
                    dtypes.float32, [])))
            output_node_name = self.graph_info[node_name].outputs[0]
            self.cur_graph.replace_const_node(min_node,
                                              [Helper.node_name_from_input(output_node_name)],
                                              node_name + ':0')
            self.cur_graph.replace_const_node(max_node,
                                              [Helper.node_name_from_input(output_node_name)],
                                              node_name + ':1')
            self.cur_graph.remove_node(node_name)

        return GraphAnalyzer().dump_graph()

    def do_transformation(self):
        if self.postfix == '__requant_min_max':
            range_data = self._parse_requantization_ranges()

            return self.generate_output_graph_ranges(range_data)

        max_name_value = self._parse_max_min_log()
        return self.generate_output_graph(max_name_value)
