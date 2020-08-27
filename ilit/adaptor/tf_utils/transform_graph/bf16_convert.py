#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2019 Intel Corporation
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

from collections import OrderedDict
from collections import namedtuple
import logging

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from .graph_transform_base import GraphTransformBase
from ..quantize_graph.quantize_graph_common import QuantizeGraphHelper as helper
from ..quantize_graph.quantize_graph_for_intel_cpu import QuantizeGraphForIntel

class BF16Convert(GraphTransformBase):
    """
    BF16 node convert transformation.
    """

    node_details = namedtuple('node_details', ['node', 'input_node', 'output'])
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
                  #"Select",
                  #"SelectV2",
                  #"Shape",
                  #"ShapeN",
                  #"Slice",
                  #"Split",
                  #"SplitV",
                  #"Squeeze",
                  #"StopGradient",
                  #"Switch",
                  #"Transpose",
                  #"ZerosLike",
                 ]

    def __init__(self,
                 input_pb,
                 device,
                 outputs,
                 fp32_ops=[],
                 bf16_ops=[]):
        super(BF16Convert, self).__init__(input_pb)

        self.parse_input_pb()
        self.device = device
        self.outputs = outputs
        self.fp32_ops = fp32_ops
        self.bf16_ops = bf16_ops
        self.converted_ops = []
        self.expand_fp32_ops = []
        self.expand_bf16_ops = []
        self._expand_op_lists()

    def _expand_op_lists(self):
        self._parse_graph()
        intel_quantizer = QuantizeGraphForIntel(self.input_graph, self.outputs, {}, self.device)
        for fp32_op in self.fp32_ops:
            fp32_node = self.node_name_mapping[fp32_op].node
            self.expand_fp32_ops.append(fp32_op)
            self.expand_fp32_ops.extend(intel_quantizer.can_fused_ops(fp32_node))
        for bf16_op in self.bf16_ops:
            bf16_node = self.node_name_mapping[bf16_op].node
            if bf16_node.op not in BF16Convert.BLACK_LIST:
                self.expand_bf16_ops.append(bf16_op)
                self.expand_bf16_ops.extend(intel_quantizer.can_fused_ops(bf16_node))

    def _parse_graph(self, input_graph=None):
        """
        Parse the graph and get the input node and output node name details.
        """
        logging.debug("start parsing graph")
        self.node_name_mapping = OrderedDict()

        graph = self.input_graph if input_graph is None else input_graph
        for node in graph.node:
            each_node = self.node_details(node=node, input_node=[], output=[])

            if node.name in self.node_name_mapping:
                raise ValueError(
                    "Duplicate Node Found when _parse_graph, the node name is {}" .format(
                        node.name))

            self.node_name_mapping[node.name] = each_node

        for node in graph.node:
            for input in node.input:
                self.node_name_mapping[helper.node_name_from_input(
                    input)].output.append(node.name)

    def _bf16_convert(self, bf16_node_name):
        self._parse_graph()
        self.converted_ops.append(bf16_node_name)
        bf16_node_detail = self.node_name_mapping[bf16_node_name]
        bf16_node = bf16_node_detail.node
        bf16_node_inputs = list(bf16_node.input)
        for each_input in bf16_node_inputs:
            each_input_detail = self.node_name_mapping[each_input]
            each_input_node = each_input_detail.node
            # Const + Cast => Const optimization
            if each_input_node.op == "Const":
                if each_input_node.attr["dtype"] == attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum):
                    fp32_value = tensor_util.MakeNdarray(each_input_node.attr.get('value').tensor)
                    helper.set_attr_dtype(each_input_node, "dtype", dtypes.bfloat16)
                    each_input_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(
                        tensor=tensor_util.make_tensor_proto(
                        fp32_value, dtypes.bfloat16, fp32_value.shape)))
                self.converted_ops.append(each_input)
            # Cast + Cast => O optimization
            elif (each_input_node.op == "Cast" and 
                    each_input_node.attr["SrcT"] == attr_value_pb2.AttrValue(type=dtypes.bfloat16.as_datatype_enum)):
                cast_input_name = each_input_node.input[0]
                for index, input_name in enumerate(bf16_node.input):
                    if input_name == each_input_node.name:
                        bf16_node.input[index] = cast_input_name
                if len(each_input_detail.output) == 1:
                    self.input_graph.node.remove(each_input_node)
                    del each_input_node
            elif (each_input not in self.expand_fp32_ops + self.converted_ops and 
                    each_input_node.op in BF16Convert.WHITE_LIST + BF16Convert.GRAY_LIST + BF16Convert.CLEAR_LIST):
                if len(each_input_detail.output) == 1:
                    self._bf16_convert(each_input)
                # TODO: Consider multi-output case
            elif each_input in self.converted_ops:
                pass
            else:
                if each_input + "_FP32toBF16" not in list(self.node_name_mapping.keys()):
                    input_cast_node = helper.create_node("Cast", each_input + "_FP32toBF16", [each_input])
                    helper.set_attr_dtype(input_cast_node, "DstT", dtypes.bfloat16)
                    helper.set_attr_dtype(input_cast_node, "SrcT", dtypes.float32)
                    helper.set_attr_bool(input_cast_node, "Truncate", False)
                else:
                    input_cast_node = self.node_name_mapping[each_input + "_FP32toBF16"].node
                for index, input_name in enumerate(bf16_node.input):
                    if input_name == each_input:
                        bf16_node.input[index] = input_cast_node.name
                self.input_graph.node.extend([input_cast_node])

        # TODO: Need consider different op type
        helper.set_attr_dtype(bf16_node, "T", dtypes.bfloat16)

        bf16_node_outputs = bf16_node_detail.output
        for each_output in bf16_node_outputs:
            each_output_detail = self.node_name_mapping[each_output]
            each_output_node = each_output_detail.node
            # Need consider output node op type
            
            if (each_output_node.op == "Cast" and 
                    each_output_node.attr["DstT"] == attr_value_pb2.AttrValue(type=dtypes.bfloat16.as_datatype_enum)):
                for cast_output in each_output_detail.output:
                    cast_output_node = self.node_name_mapping[cast_output].node
                    for index, input_name in enumerate(cast_output_node.input):
                        if each_output == input_name:
                            cast_output_node.input[index] = bf16_node.name
                del each_output_node
            elif (each_output not in self.expand_fp32_ops + self.converted_ops and
                    each_output_node.op in BF16Convert.WHITE_LIST + BF16Convert.GRAY_LIST + BF16Convert.CLEAR_LIST):
                # TODO: Consider multi node inputs case, check others inputs whether converted to BF16
                self._bf16_convert(each_output)
            elif each_output in self.converted_ops:
                pass
            else:
                if bf16_node_name + "_BF16toFP32" not in list(self.node_name_mapping.keys()):
                    output_cast_node = helper.create_node("Cast", bf16_node_name + "_BF16toFP32", [bf16_node_name])
                    helper.set_attr_dtype(output_cast_node, "DstT", dtypes.float32)
                    helper.set_attr_dtype(output_cast_node, "SrcT", dtypes.bfloat16)
                    helper.set_attr_bool(output_cast_node, "Truncate", False)
                else:
                    output_cast_node = self.node_name_mapping[bf16_node_name + "_BF16toFP32"].node

                for index, input_name in enumerate(each_output_node.input):
                    if bf16_node_name == input_name:
                        each_output_node.input[index] = output_cast_node.name
                self.input_graph.node.extend([output_cast_node])
        return
    
    def _model_bf16_convert(self):
        logging.debug("start convert bf16 graph")
        for bf16_node_name in set(self.expand_bf16_ops):
            if bf16_node_name not in self.converted_ops:
                self._bf16_convert(bf16_node_name)

    def do_transformation(self):
        """
        Execute BF16 convert.
        :return: Transformed graph
        """
        if len(self.bf16_ops) > 0:
            self._model_bf16_convert()

        return self.input_graph
