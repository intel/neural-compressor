#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2021 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import copy
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util

from ..graph_base import GraphRewriterBase
from ..graph_util import GraphAnalyzer
from ..graph_util import GraphRewriterHelper as Helper

# The lists are used to specify which attr field should be used to get the data type of the
# input/output of the node
# IMPORTANT NOTE: The order in below lists does matter! Don't change it as it impacts input and
# output data type inference!
input_dtype_attr  = ['Tinput', 'Tparams', 'SrcT', 'T', 'dtype', 'Tidx']
output_dtype_attr = ['DstT', 'Toutput', 'out_type', 'output_type', 'output', 'Tparams', \
                     'dtype', 'T', 'Tidx']
# The bool ops are special one as their output type is implicit rather than explicitly specified
bool_ops = ['Greater', 'GreaterEqual', 'Equal', 'NotEqual', 'IsFinite', 'IsInf', 'IsNan', 'Less', 'LessEqual']

class BF16Convert(GraphRewriterBase):
    """
    BF16 node convert transformation.
    """
    def __init__(self,
                 model,
                 fp32_ops=[],
                 bf16_ops=[]):
        super().__init__(model)

        self.cur_graph = GraphAnalyzer()
        self.cur_graph.graph = self.model
        self.fp32_ops = fp32_ops
        self.bf16_ops = bf16_ops
        self.converted_ops = []

    def _dtype(self, node, out=False):
        dtype_attr_list = input_dtype_attr if not out else output_dtype_attr
        for dtype in dtype_attr_list:
            if dtype in node.attr:
                return dtype
        return ''

    def _dtype_val(self, node, out=False):
        dtype = self._dtype(node, out)
        if out and node.op in bool_ops:
            return attr_value_pb2.AttrValue(type=dtypes.bool.as_datatype_enum)
        if node.op == "QuantizeV2":
            if out:
                return node.attr['T']
            else:
                return node.attr['dtype'] if 'dtype' in node.attr else \
                       attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum)
        elif node.op in ["TensorArraySplitV3", "TensorArrayConcatV3", "TensorArrayScatterV3", \
                         "TensorArrayV3"]:
            return attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum)
        elif node.op in ["TensorArraySizeV3"]:
            if out:
                return attr_value_pb2.AttrValue(type=dtypes.int32.as_datatype_enum)
            else:
                return attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum)
        else:
            return node.attr[dtype] if dtype in node.attr else None

    def _bf16_convert(self, bf16_node_name):
        bf16_node_detail = self.cur_graph.node_name_details[bf16_node_name]
        bf16_node = bf16_node_detail.node
        bf16_node_outputs = copy.deepcopy(bf16_node_detail.outputs)

        if bf16_node.name in self.converted_ops:
            return
        else:
            self.converted_ops.append(bf16_node.name)

        # skip the conversion for the node whose input data type is not FP32
        if self._dtype_val(bf16_node) != attr_value_pb2.AttrValue(
                                            type=dtypes.float32.as_datatype_enum):
            return

        for index, input_name in enumerate(bf16_node.input):
            if input_name.startswith('^'):
                continue
            if bf16_node.op == 'Fill' and index == 0:
                # Fill op's first input is a tensor of shape, skip it
                continue
            if bf16_node.op in ['FusedBatchNormV2', 'FusedBatchNormV3', '_FusedBatchNormEx'] \
               and index > 0:
                # FusedBatchNormV2/3/Ex op's 2nd and other inputs are fixed to float, skip them
                continue
            input_detail = self.cur_graph.node_name_details[Helper.node_name_from_input(
                input_name)]
            input_node = input_detail.node

            # skip the parent node if its output type is not FP32
            if self._dtype_val(input_node, True) != \
                       attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum):
                continue

            if input_node.name in self.converted_ops:
                continue
            elif input_node.op == 'Cast' and \
                 input_node.attr["SrcT"] == attr_value_pb2.AttrValue(
                        type=dtypes.bfloat16.as_datatype_enum) and \
                 input_node.attr["DstT"] == attr_value_pb2.AttrValue(
                        type=dtypes.float32.as_datatype_enum):
                parent_input_name = Helper.node_name_from_input(input_node.input[0])
                bf16_node.input[index] = parent_input_name
                outputs = self.cur_graph.node_name_details[parent_input_name].outputs
                outputs = list(map(lambda x: x.replace(input_name, bf16_node.name), outputs))
                self.cur_graph.remove_node(input_name)
            elif input_node.op == 'Cast' and \
                 input_node.attr["DstT"] == attr_value_pb2.AttrValue(
                        type=dtypes.float32.as_datatype_enum):
                input_node.attr["DstT"].CopyFrom(
                                  attr_value_pb2.AttrValue(type=dtypes.bfloat16.as_datatype_enum))
            elif input_node.op == "Const" and \
                 input_node.attr["dtype"] == attr_value_pb2.AttrValue(
                      type=dtypes.float32.as_datatype_enum):
                fp32_value = tensor_util.MakeNdarray(input_node.attr.get('value').tensor)
                Helper.set_attr_dtype(input_node, "dtype", dtypes.bfloat16)
                input_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(
                    tensor=tensor_util.make_tensor_proto(
                        fp32_value, dtypes.bfloat16, fp32_value.shape)))
                self.converted_ops.append(input_node.name)
            elif input_node.name in self.bf16_ops:
                self._bf16_convert(input_node.name)
            else:
                cast_node_name = input_name.replace(':', '_') + "/" + bf16_node_name + "_FP32toBF16"
                assert cast_node_name not in list(self.cur_graph.node_name_details.keys())
                input_cast_node = Helper.create_node(
                    "Cast", cast_node_name, [input_name])
                Helper.set_attr_dtype(input_cast_node, "DstT", dtypes.bfloat16)
                Helper.set_attr_dtype(input_cast_node, "SrcT", dtypes.float32)
                Helper.set_attr_bool(input_cast_node, "Truncate", False)
                bf16_node.input[index] = input_cast_node.name
                outputs = self.cur_graph.node_name_details[ \
                                  Helper.node_name_from_input(input_name)].outputs
                outputs = list(map(lambda x: x.replace(bf16_node.name, cast_node_name), outputs))
                self.cur_graph.add_node(input_cast_node, input_name, [bf16_node_name])

        bf16_node.attr[self._dtype(bf16_node)].CopyFrom(
                             attr_value_pb2.AttrValue(type=dtypes.bfloat16.as_datatype_enum))

        # skip the conversion for the node whose output data type is not FP32
        if self._dtype_val(bf16_node, True) != attr_value_pb2.AttrValue(
                                                   type=dtypes.float32.as_datatype_enum) and \
           self._dtype_val(bf16_node, True) != attr_value_pb2.AttrValue(
                                                   type=dtypes.bfloat16.as_datatype_enum):
            return

        bf16_node.attr[self._dtype(bf16_node, True)].CopyFrom(
            attr_value_pb2.AttrValue(type=dtypes.bfloat16.as_datatype_enum))

        for output_name in bf16_node_outputs:
            output_detail = self.cur_graph.node_name_details[output_name]
            output_node = output_detail.node

            # skip the control edge
            if '^' + bf16_node_name in output_node.input:
                continue

            if output_node.op == 'Cast' and \
               output_node.attr["SrcT"] == attr_value_pb2.AttrValue(
                      type=dtypes.bfloat16.as_datatype_enum) and \
               output_node.attr["DstT"] == attr_value_pb2.AttrValue(
                      type=dtypes.float32.as_datatype_enum):
                assert False
            if output_node.op == 'QuantizeV2' and 'dtype' in output_node.attr:
                output_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(
                      type=dtypes.bfloat16.as_datatype_enum))
            elif output_node.name not in self.bf16_ops and self._dtype_val(output_node) == \
                     attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum):
                cast_node_name = bf16_node_name + "/" + output_node.name + "_BF16toFP32"
                assert cast_node_name not in list(self.cur_graph.node_name_details.keys())
                output_cast_node = Helper.create_node(
                    "Cast", cast_node_name, [bf16_node_name])
                Helper.set_attr_dtype(output_cast_node, "DstT", dtypes.float32)
                Helper.set_attr_dtype(output_cast_node, "SrcT", dtypes.bfloat16)
                Helper.set_attr_bool(output_cast_node, "Truncate", False)
                index = [Helper.node_name_from_input(_) for _ in output_node.input].index(
                            bf16_node_name)
                output_node.input[index] = output_cast_node.name
                self.cur_graph.add_node(output_cast_node, bf16_node_name, [output_name])

    def _model_bf16_convert(self):
        logging.debug("start convert bf16 graph")
        self.cur_graph.parse_graph()
        for bf16_node_name in set(self.bf16_ops):
            if bf16_node_name not in self.cur_graph.node_name_details:
                self.bf16_ops.remove(bf16_node_name)
                continue
            for output_name in self.cur_graph.node_name_details[bf16_node_name].outputs:
                output_detail = self.cur_graph.node_name_details[output_name]
                output_node = output_detail.node
                if "Quantized" in output_node.op:
                    self.bf16_ops.remove(bf16_node_name)
                    break
        for bf16_node_name in set(self.bf16_ops):
            if bf16_node_name not in self.converted_ops:
                self._bf16_convert(bf16_node_name)
        return self.cur_graph.dump_graph()

    def do_transformation(self):
        """
        Execute BF16 convert.
        :return: Transformed graph
        """
        converted_graph_def = self._model_bf16_convert()
        converted_graph_def.library.CopyFrom(self.model.library)
        return converted_graph_def
