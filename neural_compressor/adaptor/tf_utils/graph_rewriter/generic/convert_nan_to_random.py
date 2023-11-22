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
"""Convert NAN to random Graph Rewriter."""

import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes, tensor_util

from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer

from ..graph_base import GraphRewriterBase


class ConvertNanToRandom(GraphRewriterBase):
    """Convert Const node which value consists of NAN to random data."""

    def do_transformation(self):
        """Execute convert NAN to random."""
        cur_graph = GraphAnalyzer()
        cur_graph.graph = self.model

        graph_info = cur_graph.parse_graph()

        target_nodes = cur_graph.query_fusion_pattern_nodes([["Const"]])

        for i in target_nodes:
            const_node = graph_info[i[0]].node
            const_content = tensor_util.MakeNdarray(const_node.attr["value"].tensor)
            if const_content.dtype == np.float32 and np.any(np.isnan(const_content)):
                const_node.attr["value"].CopyFrom(
                    attr_value_pb2.AttrValue(
                        tensor=tensor_util.make_tensor_proto(
                            np.random.rand(*const_content.shape), dtypes.float32, const_content.shape
                        )
                    )
                )

        return cur_graph.dump_graph()
