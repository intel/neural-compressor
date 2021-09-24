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


from neural_compressor.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase

class StripUnusedNodesOptimizer(GraphRewriterBase):
    def __init__(self, model, input_node_names, output_node_names):
        super().__init__(model)
        self.input_node_names = input_node_names
        self.output_node_names = output_node_names

    @dump_elapsed_time("Pass StripUnusedNodesOptimizer")
    def do_transformation(self):
        from neural_compressor.adaptor.tf_utils.util import fix_ref_type_of_graph_def
        from neural_compressor.adaptor.tf_utils.util import strip_unused_nodes
        self.model = fix_ref_type_of_graph_def(self.model)
        return strip_unused_nodes(self.model,
                                  self.input_node_names,
                                  self.output_node_names)
