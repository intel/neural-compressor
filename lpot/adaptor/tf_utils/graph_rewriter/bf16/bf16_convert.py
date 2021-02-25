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


class BF16Convert(GraphRewriterBase):
    """
    BF16 node convert transformation.
    """

    # TODO: Analysis all of OPs, consider more structures
    # the set of ops that are considered numerically-safe (for execution
    # in bf16), performance-critical, and can run in bf16. These ops are always
    # converted to bf16.
    WHITE_LIST = ["Conv2D",
                  "Conv2DBackpropFilter",
                  "Conv2DBackpropInput",
                  "Conv3D",
                  "Conv3DBackpropFilterV2",
                  "Conv3DBackpropInputV2",
                  "DepthwiseConv2dNative",
                  "DepthwiseConv2dNativeBackpropFilter",
                  "DepthwiseConv2dNativeBackpropInput",
                  "MatMul",
                  "BatchMatMul",
                  "BatchMatMulV2",
                  ]
    # the set of ops that can run in bf16 and are considered numerically-
    # safe (for execution in bf16), but which may be made unsafe by an upstream
    # blacklist op.
    GRAY_LIST = ["Add",
                 "AddN",
                 "AddV2",
                 "AvgPool",
                 "AvgPool3D",
                 "AvgPool3DGrad",
                 "AvgPoolGrad",
                 "BiasAdd",
                 "BiasAddGrad",
                 "BiasAddV1",
                 "FusedBatchNormV2",
                 "FusedBatchNormGradV2",
                 "FusedBatchNormV3",
                 "FusedBatchNormGradV3",
                 "LeakyRelu",
                 "LeakyReluGrad",
                 "Mul",
                 "Sub",
                 ]
    # the set of ops that are considered numerically-dangerous (i.e.,
    # unsafe for execution in bf16) and whose effects may also be observed in
    # downstream nodes (e.g. for f16, in Exp -> Add, the Add is unsafe due to
    # the Exp).
    BLACK_LIST = ["Exp",
                  "Expm1",
                  "L2Loss",
                  "Mean",
                  "Pow",
                  "SaveV2",
                  "Softmax",
                  "SoftmaxCrossEntropyWithLogits",
                  "SparseSoftmaxCrossEntropyWithLogits",
                  "Sum",
                  ]
    # the set of ops that do not have numerically-significant effects
    # (i.e., they are always considered safe for execution in bf16 precision), and
    # can run in bf16.
    CLEAR_LIST = ["Concat",
                  "ConcatV2",
                  "Enter",
                  "EnsureShape",
                  "Equal",
                  "Exit",
                  "ExpandDims",
                  "Identity",
                  "MaxPool",
                  "MaxPool3D",
                  "MaxPool3DGrad",
                  "MaxPoolGrad",
                  "MaxPoolV2",
                  "Maximum",
                  "Merge",
                  "NextIteration",
                  "PreventGradient",
                  "Relu",
                  "Relu6",
                  "Relu6Grad",
                  "ReluGrad",
                  "Reshape",
                  # "Select",
                  # "SelectV2",
                  # "Shape",
                  # "ShapeN",
                  # "Slice",
                  # "Split",
                  # "SplitV",
                  # "Squeeze",
                  # "StopGradient",
                  # "Switch",
                  # "Transpose",
                  # "ZerosLike",
                  ]

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

    def _bf16_convert(self, bf16_node_name):
        self.converted_ops.append(bf16_node_name)
        bf16_node_detail = self.cur_graph.node_name_details[bf16_node_name]
        bf16_node = bf16_node_detail.node
        bf16_node_inputs = list(bf16_node.input)

        if 'T' in bf16_node.attr and bf16_node.attr['T'] != \
                attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum) and \
                    bf16_node.op != 'Dequantize':
            return
        for each_input in bf16_node_inputs:
            each_input_detail = self.cur_graph.node_name_details[Helper.node_name_from_input(
                each_input)]
            each_input_node = each_input_detail.node
            # Const + Cast => Const optimization
            if each_input_node.op == "Const":
                if each_input_node.attr["dtype"] == attr_value_pb2.AttrValue(
                        type=dtypes.float32.as_datatype_enum):
                    fp32_value = tensor_util.MakeNdarray(each_input_node.attr.get('value').tensor)
                    Helper.set_attr_dtype(each_input_node, "dtype", dtypes.bfloat16)
                    each_input_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(
                        tensor=tensor_util.make_tensor_proto(
                            fp32_value, dtypes.bfloat16, fp32_value.shape)))
                    self.converted_ops.append(each_input)
            elif 'T' in each_input_node.attr and each_input_node.attr['T'] != \
                attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum) and \
                    each_input_node.op != 'Dequantize':
                continue
            # Cast + Cast => O optimization
            elif (each_input_node.op == "Cast" and
                  each_input_node.attr["SrcT"] == attr_value_pb2.AttrValue(
                      type=dtypes.bfloat16.as_datatype_enum)):
                cast_input_name = each_input_node.input[0]
                for index, input_name in enumerate(bf16_node.input):
                    if input_name == each_input_node.name:
                        bf16_node.input[index] = cast_input_name
                self.cur_graph.node_name_details[cast_input_name].outputs.append(bf16_node_name)
                if len(each_input_detail.outputs) == 1:
                    self.cur_graph.remove_node(each_input)
                    self.cur_graph.node_name_details[cast_input_name].outputs.remove(each_input)
            elif (each_input not in self.fp32_ops + self.converted_ops and
                    each_input_node.op in BF16Convert.WHITE_LIST + \
                    BF16Convert.GRAY_LIST + BF16Convert.CLEAR_LIST and
                    len(each_input_detail.outputs) == 1):
                self._bf16_convert(each_input)
                # TODO: Consider multi-output case
            elif each_input in self.converted_ops:
                pass
            else:
                if each_input + "_FP32toBF16" not in list(self.cur_graph.node_name_details.keys()):
                    input_cast_node = Helper.create_node(
                        "Cast", each_input.replace(':', '__') + "_FP32toBF16", [each_input])
                    Helper.set_attr_dtype(input_cast_node, "DstT", dtypes.bfloat16)
                    Helper.set_attr_dtype(input_cast_node, "SrcT", dtypes.float32)
                    Helper.set_attr_bool(input_cast_node, "Truncate", False)
                    self.cur_graph.add_node(input_cast_node, each_input, [bf16_node_name])
                else:
                    input_cast_node = self.cur_graph.node_name_details[each_input +
                                                                       "_FP32toBF16"].node
                    for index, input_name in enumerate(bf16_node.input):
                        if Helper.node_name_from_input(input_name) == each_input:
                            bf16_node.input[index] = input_cast_node.name
                    self.cur_graph.node_name_details[input_cast_node.name].outputs.append(
                        bf16_node_name)

        # TODO: Need consider different op type
        Helper.set_attr_dtype(bf16_node, "T", dtypes.bfloat16)

        bf16_node_outputs = copy.deepcopy(bf16_node_detail.outputs)
        for each_output in bf16_node_outputs:
            each_output_detail = self.cur_graph.node_name_details[each_output]
            each_output_node = each_output_detail.node
            # Need consider output node op type

            if (each_output_node.op == "Cast" and
                    each_output_node.attr["DstT"] == attr_value_pb2.AttrValue(
                    type=dtypes.bfloat16.as_datatype_enum)):
                for cast_output in each_output_detail.outputs:
                    cast_output_node = self.cur_graph.node_name_details[cast_output].node
                    for index, input_name in enumerate(cast_output_node.input):
                        if each_output == input_name:
                            cast_output_node.input[index] = bf16_node.name
                bf16_node_detail.outputs.remove(each_output)
                bf16_node_detail.outputs.extend(each_output_detail.outputs)
                self.cur_graph.remove_node(each_output)
            elif (each_output not in self.fp32_ops + self.converted_ops and
                    each_output_node.op in BF16Convert.WHITE_LIST + \
                    BF16Convert.GRAY_LIST + BF16Convert.CLEAR_LIST):
                # TODO: Consider multi node inputs case, check others inputs whether
                # converted to BF16
                self._bf16_convert(each_output)
            elif each_output in self.converted_ops:
                pass
            else:
                if bf16_node_name + \
                        "_BF16toFP32" not in list(self.cur_graph.node_name_details.keys()):
                    output_cast_node = Helper.create_node(
                        "Cast", bf16_node_name + "_BF16toFP32", [bf16_node_name])
                    Helper.set_attr_dtype(output_cast_node, "DstT", dtypes.float32)
                    Helper.set_attr_dtype(output_cast_node, "SrcT", dtypes.bfloat16)
                    Helper.set_attr_bool(output_cast_node, "Truncate", False)
                    self.cur_graph.add_node(output_cast_node, bf16_node_name, [each_output])
                else:
                    output_cast_node = self.cur_graph.node_name_details[bf16_node_name +
                                                                        "_BF16toFP32"].node
                    for index, input_name in enumerate(each_output_node.input):
                        if bf16_node_name == input_name:
                            each_output_node.input[index] = output_cast_node.name
                    self.cur_graph.node_name_details[bf16_node_name +
                                                     "_BF16toFP32"].outputs.append(each_output)

        return

    def _model_bf16_convert(self):
        logging.debug("start convert bf16 graph")
        self.cur_graph.parse_graph()
        for bf16_node_name in set(self.bf16_ops):
            if bf16_node_name not in self.converted_ops:
                self._bf16_convert(bf16_node_name)

    def do_transformation(self):
        """
        Execute BF16 convert.
        :return: Transformed graph
        """
        self._model_bf16_convert()

        return self.cur_graph.dump_graph()
