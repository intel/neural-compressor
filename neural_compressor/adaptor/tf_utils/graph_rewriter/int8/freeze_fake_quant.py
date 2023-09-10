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
"""Freeze FakeQuant op Graph Rewriter."""

from tensorflow.python.framework import dtypes

from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer, GraphRewriterHelper
from neural_compressor.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase


class FreezeFakeQuantOpOptimizer(GraphRewriterBase):  # pragma: no cover
    """Freeze fake_quant op to the following Quantize op and prioring Dequantize op."""

    def __init__(self, model):
        """Initialization."""
        super().__init__(model)

        self.graph_analyzer = GraphAnalyzer()
        self.graph_analyzer.graph = self.model

        self.graph_info = self.graph_analyzer.parse_graph()

        self.freeze_patterns = {
            str(
                [["Requantize", "RequantizePerChannel"], ["Dequantize"], ["FakeQuantWithMinMaxVars"]]
            ): self._freeze_requant_dequant_fakequant,
            str([["FakeQuantWithMinMaxVars"], ["QuantizeV2"]]): self._freeze_fakequant_quant,
            str(
                [["FakeQuantWithMinMaxVars"], ["Shape"], ["StridedSlice"], ["Pack"], ["Reshape"], ["QuantizeV2"]]
            ): self._freeze_fakequant_metaop_quant,
        }

    def _freeze_requant_dequant_fakequant(self, pattern_nodes):
        """Freeze Requantize Dequantize FakeQuant fusion."""
        requant_node_name = pattern_nodes[0]
        requant_node = self.graph_info[requant_node_name].node
        fake_quant_node_name = pattern_nodes[2]
        fake_quant_node = self.graph_info[fake_quant_node_name].node

        # set the third input "requested_output_min" of RequantizePerChannel or Requantize op.
        requested_output_min_node = self.graph_info[requant_node.input[3]].node
        min_node = self.graph_info[fake_quant_node.input[1]].node
        GraphRewriterHelper.set_attr_tensor(
            requested_output_min_node, "value", min_node.attr["value"].tensor, dtypes.float32
        )
        # set the fourth input "requested_output_max" of RequantizePerChannel or Requantize op.
        requested_output_max_node = self.graph_info[requant_node.input[4]].node
        max_node = self.graph_info[fake_quant_node.input[2]].node
        GraphRewriterHelper.set_attr_tensor(
            requested_output_max_node, "value", max_node.attr["value"].tensor, dtypes.float32
        )

    def _freeze_fakequant_quant(self, pattern_nodes):
        """Freeze FakeQuant QuantizeV2 fusion."""
        fake_quant_node_name = pattern_nodes[0]
        fake_quant_node = self.graph_info[fake_quant_node_name].node
        quant_node_name = pattern_nodes[1]
        quant_node = self.graph_info[quant_node_name].node

        # set the second input "min_range" of QuantizeV2 op.
        min_node = self.graph_info[fake_quant_node.input[1]].node
        min_range_node = self.graph_info[quant_node.input[1]].node
        GraphRewriterHelper.set_attr_tensor(min_range_node, "value", min_node.attr["value"].tensor, dtypes.float32)
        # set the third input "max_range" of QuantizeV2 op.
        max_node = self.graph_info[fake_quant_node.input[2]].node
        max_range_node = self.graph_info[quant_node.input[2]].node
        GraphRewriterHelper.set_attr_tensor(max_range_node, "value", max_node.attr["value"].tensor, dtypes.float32)

    def _freeze_fakequant_metaop_quant(self, pattern_nodes):
        """Freeze FakeQuant Meta ops QuantizeV2 fusion."""
        fake_quant_node_name = pattern_nodes[0]
        fake_quant_node = self.graph_info[fake_quant_node_name].node
        quant_node_name = pattern_nodes[5]
        quant_node = self.graph_info[quant_node_name].node

        # set the second input "min_range" of QuantizeV2 op.
        min_node = self.graph_info[fake_quant_node.input[1]].node
        min_range_node = self.graph_info[quant_node.input[1]].node
        GraphRewriterHelper.set_attr_tensor(min_range_node, "value", min_node.attr["value"].tensor, dtypes.float32)
        # set the third input "max_range" of QuantizeV2 op.
        max_node = self.graph_info[fake_quant_node.input[2]].node
        max_range_node = self.graph_info[quant_node.input[2]].node
        GraphRewriterHelper.set_attr_tensor(max_range_node, "value", max_node.attr["value"].tensor, dtypes.float32)

    def _remove_all_fake_quants(self):
        """Remove all the fake quants."""
        _const_node = []

        for node_name in list(self.graph_info.keys()):
            node = self.graph_info[node_name].node
            if node.op == "FakeQuantWithMinMaxVars":
                origin_outputs = list(self.graph_info[node_name].outputs)
                min_node_name = self.graph_info[node.input[1]].node.name
                max_node_name = self.graph_info[node.input[2]].node.name
                _const_node.append(min_node_name)
                _const_node.append(max_node_name)

                self.graph_analyzer.remove_node_with_single_input_output(node_name)

                for j in origin_outputs[1:]:
                    output_node = self.graph_info[j].node
                    if (
                        len(output_node.input) == 1
                        and output_node.op == "Const"
                        and output_node.input[0] == "^" + node.name
                    ):
                        self.graph_info[j].node.ClearField("input")
                    elif output_node.op == "NoOp":
                        new_noop_input = [
                            noop_input for noop_input in output_node.input if noop_input != "^" + node.name
                        ]
                        output_node.ClearField("input")
                        output_node.input.extend(new_noop_input)

        # remove those left const nodes used by FakeQuantWithMinMaxVars
        for node_name in list(self.graph_info.keys()):
            if node_name in _const_node:
                self.graph_analyzer.remove_node(node_name)

    @dump_elapsed_time("Pass FreezeFakeQuantOpOptimizer")
    def do_transformation(self):
        """Execute freeze FakeQuant optimization."""
        for _pattern, _handler in self.freeze_patterns.items():
            _match_pattern_nodes = self.graph_analyzer.query_fusion_pattern_nodes(eval(_pattern))
            if len(_match_pattern_nodes) == 0:
                continue

            for _pattern_nodes in _match_pattern_nodes:
                _handler(_pattern_nodes)

        self._remove_all_fake_quants()

        return GraphAnalyzer().dump_graph()
