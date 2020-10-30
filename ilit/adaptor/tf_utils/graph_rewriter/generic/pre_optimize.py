#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
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


import logging
from ilit.adaptor.tf_utils.util import get_graph_def
from ilit.adaptor.tf_utils.graph_rewriter.graph_util import GraphAnalyzer

from .fuse_column_wise_mul import FuseColumnWiseMulOptimizer
from .remove_training_nodes import RemoveTrainingNodesOptimizer
from .split_shared_input import SplitSharedInputOptimizer
from .strip_unused_nodes import StripUnusedNodesOptimizer
from .graph_cse_optimizer import GraphCseOptimizer
from .fold_constant import GraphFoldConstantOptimizer
from .fold_batch_norm import FoldBatchNormNodesOptimizer
from .update_enter import UpdateEnterOptimizer


class PreOptimization(object):
    def __init__(self, model, inputs_name, outputs_name):
        self.input_graph = get_graph_def(model, outputs_name)

        self.analyzer = GraphAnalyzer()
        self.analyzer.graph = self.input_graph
        self.analyzer.parse_graph()
        self.inputs = inputs_name
        self.outputs = outputs_name
        self.logger = logging.getLogger()
        self._tmp_graph_def = None
        # self.tf_version = tf.version.VERSION
        self._excluded_node_names = []
        if not self.inputs or not self.outputs:
            self.inputs, self.outputs = self.analyzer.get_graph_input_output()

    def get_excluded_node_names(self):
        """Get the excluded node name

        Returns:
            string list: the excluded ops' name
        """
        return self._excluded_node_names

    def get_optimized_graphdef(self):
        """Executed the non-precision dependant graph optimization.
        The input graph will be optimized with following passes:
        1. Remove the training nodes like Identity Op.
        2. Split the shared nodes like weights node for multi-Conv2d.
        3. Fold Constant Nodes as less as possible.
        4. Fuse the Mul node into the previous Conv2D/MatMul if possible.
        5. Strip the useless nodes.
        6. Do the Common sequence elimation optimization on the graph.
        7. Fold the BN node into the previous Conv2D if possible.

        Returns:
            [graphdef]: the optimized graphdef object.
        """
        self.logger.debug("Start to pre optimize input model...")

        self.logger.debug("Pre Optimize RemoveTrainingNodesOptimizer is working...")
        self._tmp_graph_def = RemoveTrainingNodesOptimizer(
            self.input_graph, protected_nodes=self.outputs).do_transformation()

        self.logger.debug("Pre Optimize SplitSharedInputOptimizer is working...")
        self._tmp_graph_def = SplitSharedInputOptimizer(self._tmp_graph_def).do_transformation()

        self.logger.debug("Pre Optimize GraphFoldConstantOptimizer is working...")
        self._tmp_graph_def = GraphFoldConstantOptimizer(self._tmp_graph_def).do_transformation()

        self.logger.debug("Pre Optimize FuseColumnWiseMulOptimizer is working...")
        self._tmp_graph_def = FuseColumnWiseMulOptimizer(self._tmp_graph_def).do_transformation()

        self.logger.debug("Pre Optimize StripUnusedNodesOptimizer is working...")
        self._tmp_graph_def = StripUnusedNodesOptimizer(self._tmp_graph_def, self.inputs,
                                                        self.outputs).do_transformation()

        self.logger.debug("Pre Optimize GraphCseOptimizer is working...")
        self._tmp_graph_def = GraphCseOptimizer(self._tmp_graph_def).do_transformation()

        self.logger.debug("Pre Optimize FoldBatchNormNodesOptimizer is working...")
        self._tmp_graph_def = FoldBatchNormNodesOptimizer(self._tmp_graph_def).do_transformation()

        #TODO we should handle all control ops elegantly not bypass it.
        self._tmp_graph_def, excluded_node_names = UpdateEnterOptimizer(
            self._tmp_graph_def).do_transformation()
        self._excluded_node_names.extend(excluded_node_names)
        self._tmp_graph_def.library.CopyFrom(self.input_graph.library)

        return self._tmp_graph_def

    def get_matched_nodes(self, patterns):
        """Searche the matched nodes with the specified patterns

        Args:
            patterns ([string list]): The patterns shouid be illustrated as below.
                [['MatMul'], ("BiasAdd"), ("Relu")]
        Returns:
            [string list]: It will return the list that contains the matched nodes name
                and pattern. ['matched_node_a_name', 'matched_node_a_name',['MatMul','BiasAdd']]
        """
        self.analyzer.graph = self._tmp_graph_def
        self.analyzer.parse_graph()
        res = []

        for sub_pattern in patterns:
            res.extend(self.analyzer.query_fusion_pattern_nodes(sub_pattern))

        return res

    def has_positive_input(self, node_name):
        """Check the specified node has the positive input or not.

        Args:
            node_name ([string]): node's name

        Returns:
            [bool]: True if the node has the positive input data,
                    False if the node has the negative input data.
        """
        return self.analyzer.has_positive_input(node_name)
