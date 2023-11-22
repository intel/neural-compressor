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
"""Strip Equivalent Nodes Graph Rewriter."""

from neural_compressor.adaptor.tf_utils.util import fix_ref_type_of_graph_def, strip_equivalent_nodes
from neural_compressor.utils import logger
from neural_compressor.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase


class StripEquivalentNodesOptimizer(GraphRewriterBase):
    """Remove the equivalent nodes which have the same inputs and attributes."""

    def __init__(self, model, output_node_names):
        """Initialization."""
        super().__init__(model)
        self.output_node_names = output_node_names

    @dump_elapsed_time("Pass StripEquivalentNodesOptimizer")
    def do_transformation(self):
        """Strip the equivalent nodes in the graph."""
        self.model = fix_ref_type_of_graph_def(self.model)
        iter_num = 0
        replaced_nodes_type = True
        all_replaced_nodes_type = {}
        while replaced_nodes_type:
            self.model, replaced_nodes_type = strip_equivalent_nodes(self.model, self.output_node_names)
            for k, v in replaced_nodes_type.items():
                all_replaced_nodes_type[k] = all_replaced_nodes_type.get(k, 0) + v
            iter_num += 1
            logger.debug(
                f"StripEquivalentNodes[Iter-{iter_num}]-Replaced equivalent node types are {replaced_nodes_type}"
            )
        logger.warning("All replaced equivalent node types are {}".format(all_replaced_nodes_type))
        return self.model
