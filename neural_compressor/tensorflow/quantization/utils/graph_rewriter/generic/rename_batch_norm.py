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
"""Rename FusedBatchNorm op to FusedBatchNormV2 Graph Rewriter."""

import math

import numpy as np
from tensorflow.core.framework import attr_value_pb2, node_def_pb2
from tensorflow.python.framework import tensor_util

from neural_compressor.tensorflow.quantization.utils.graph_util import GraphAnalyzer
from neural_compressor.tensorflow.quantization.utils.graph_util import GraphRewriterHelper as Helper
from neural_compressor.tensorflow.utils import dump_elapsed_time

from ..graph_base import GraphRewriterBase


class RenameBatchNormOptimizer(GraphRewriterBase):
    """Rename FusedBatchNorm op to FusedBatchNormV2."""

    @dump_elapsed_time("Pass RenameBatchNormOptimizer")
    def do_transformation(self):
        """Rename FusedBatchNorm op to FusedBatchNormV2.

        This pass is needed for bf16 conversion. Due to TensorFlow historical reason,
        FusedBatchNorm is not a bf16 op but FusedBatchNormV2 is. As the latter is compatible
        with the former, changing FusedBatchNorm op to FusedBatchNormV2 op will be able to
        convert to bf16 op on the platforms supporting VNNI_BF16 and AMX instructions.

        Returns:
          Modified graph with BN ops renamed.

        Raises:
          ValueError: If the graph is badly formed.
        """
        cur_graph = GraphAnalyzer()
        cur_graph.graph = self.model
        graph_details = cur_graph.parse_graph()

        for _, v in graph_details.items():
            # for node in cur_graph.graph.node:
            if v.node.op == "FusedBatchNorm" or v.node.op == "FusedBatchNormV2":
                v.node.op = "FusedBatchNormV3"
                v.node.attr["U"].CopyFrom(v.node.attr["T"])

        return cur_graph.dump_graph()
