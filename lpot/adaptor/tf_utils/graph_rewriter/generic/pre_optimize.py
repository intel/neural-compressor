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


import logging
from lpot.adaptor.tf_utils.graph_rewriter.graph_util import GraphAnalyzer
from lpot.utils.utility import dump_elapsed_time
from lpot.model.model import TensorflowModel

from .fuse_column_wise_mul import FuseColumnWiseMulOptimizer
from .remove_training_nodes import RemoveTrainingNodesOptimizer
from .split_shared_input import SplitSharedInputOptimizer
from .strip_unused_nodes import StripUnusedNodesOptimizer
from .graph_cse_optimizer import GraphCseOptimizer
from .fold_constant import GraphFoldConstantOptimizer
from .fold_batch_norm import FoldBatchNormNodesOptimizer
from .update_enter import UpdateEnterOptimizer
from .convert_layout import ConvertLayoutOptimizer
from .fuse_gelu import FuseGeluOptimizer
from .grappler_pass import GrapplerOptimizer

class PreOptimization():
    def __init__(self, model):
        self.model = model
        self.output_node_names = model.output_node_names
        self.input_node_names = model.input_node_names
        if model.iter_op is not None:
            self.output_node_names.append('MakeIterator')

        self.analyzer = GraphAnalyzer()
        self.analyzer.graph = model.graph_def
        self.analyzer.parse_graph()
        self.logger = logging.getLogger()
        self._tmp_graph_def = None
        self._excluded_node_names = []

    def get_excluded_node_names(self):
        """Get the excluded node name

        Returns:
            string list: the excluded ops' name
        """
        return self._excluded_node_names

    @dump_elapsed_time("Pass Pre Optimization")
    def get_optimized_model(self):
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

        self._tmp_graph_def = ConvertLayoutOptimizer(
            self.model.graph_def, self.output_node_names).do_transformation()

        self._tmp_graph_def = GrapplerOptimizer(
            self._tmp_graph_def, self.output_node_names).do_transformation()

        self._tmp_graph_def = RemoveTrainingNodesOptimizer(
            self._tmp_graph_def, protected_nodes=self.output_node_names).do_transformation()

        self._tmp_graph_def = SplitSharedInputOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = GraphFoldConstantOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = FuseColumnWiseMulOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = StripUnusedNodesOptimizer(self._tmp_graph_def, self.input_node_names,
                                                        self.output_node_names).do_transformation()

        self._tmp_graph_def = FuseGeluOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = GraphCseOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = FoldBatchNormNodesOptimizer(self._tmp_graph_def).do_transformation()

        #TODO we should handle all control ops elegantly not bypass it.
        self._tmp_graph_def, excluded_node_names = UpdateEnterOptimizer(
            self._tmp_graph_def).do_transformation()
        self._excluded_node_names.extend(excluded_node_names)
        self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)

        optimized_model = TensorflowModel(self._tmp_graph_def, self.model.framework_specific_info)
        return optimized_model

    def get_matched_nodes(self, patterns):
        """Searche the matched nodes with the specified patterns

        Args:
            patterns ([string list]): The patterns should be illustrated as below.
                [['MatMul'], ("BiasAdd"), ("Relu",)]
        Returns:
            [string list]: It will return the list that contains the matched nodes name
                and pattern. ['matched_node_a_name', 'matched_node_a_name',['MatMul','BiasAdd']]
        """
        self.analyzer.graph = self._tmp_graph_def
        self.analyzer.parse_graph()
        res = []

        for sub_pattern in patterns:
            res.extend([i for i in self.analyzer.query_fusion_pattern_nodes(
                sub_pattern) if i not in res])
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
