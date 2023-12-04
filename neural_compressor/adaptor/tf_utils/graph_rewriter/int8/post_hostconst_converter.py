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
"""Post HostConst Graph Rewriter."""

import os

from tensorflow.core.framework import graph_pb2, node_def_pb2

from neural_compressor.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase


class PostHostConstConverter(GraphRewriterBase):
    """Support HostConst as default for all devices, not just for GPU."""

    @dump_elapsed_time("Pass PostHostConstConverter")
    def do_transformation(self):
        """Convert Const to HostConst as default."""
        if os.environ.get("DISABLE_HOSTCONST") == "1":
            return self.model
        output_graph_def = graph_pb2.GraphDef()
        for node in self.model.node:
            new_node = node_def_pb2.NodeDef()
            new_node.CopyFrom(node)
            new_node.device = ""
            if (
                node.op == "Const"
                and node.attr["dtype"].type in [1, 3]
                and (
                    node.name.endswith("_min")
                    or node.name.endswith("_max")
                    or node.name.endswith("_max_only")
                    or node.name.endswith("_min_only")
                )
            ):
                new_node.op = "HostConst"
            output_graph_def.node.extend([new_node])
        return output_graph_def
