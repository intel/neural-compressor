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
"""Convert Layout Graph Rewriter."""

import tensorflow as tf
from tensorflow.core.protobuf import config_pb2, meta_graph_pb2, rewriter_config_pb2
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training import saver as saver_lib

from neural_compressor.tensorflow.quantization.utils.graph_rewriter.graph_base import GraphRewriterBase
from neural_compressor.tensorflow.utils import dump_elapsed_time, version1_gt_version2


class ConvertLayoutOptimizer(GraphRewriterBase):
    """The layout conversion optimizer, convert NCHW to NHWC format.

    It is executed only when NCHW node exists and tensorflow version is 2.4.0 and above.

    Args: model: input graph_def
          outputs: output name list

    Return: converted graph_def
    """

    def __init__(self, model, outputs):
        """Initialization."""
        super().__init__(model)
        self.outputs = outputs

    @dump_elapsed_time("ConvertLayoutOptimizer")
    def do_transformation(self):
        """Execute converting layout."""
        convert = False
        for node in self.model.node:
            if "Conv" in node.op and "data_format" in node.attr and node.attr["data_format"].s == b"NCHW":
                convert = True
                break
        if convert and version1_gt_version2(tf.version.VERSION, "2.3.0"):
            g = tf.Graph()
            with g.as_default():  # pylint: disable=not-context-manager
                g = tf.compat.v1.import_graph_def(self.model, name="")
                meta_graph = saver_lib.export_meta_graph(graph_def=self.model, graph=g, clear_devices=False)
                fetch_collection = meta_graph_pb2.CollectionDef()
                for fetch in self.outputs:
                    fetch_collection.node_list.value.append(fetch)  # pylint: disable=no-member
                meta_graph.collection_def["train_op"].CopyFrom(  # pylint: disable=no-member
                    fetch_collection
                )  # pylint: disable=no-member

            config = config_pb2.ConfigProto()
            convert = rewriter_config_pb2.RewriterConfig.NCHW_TO_NHWC  # pylint: disable=no-member
            config.graph_options.rewrite_options.CopyFrom(  # pylint: disable=no-member
                rewriter_config_pb2.RewriterConfig(
                    disable_model_pruning=True,
                    constant_folding=rewriter_config_pb2.RewriterConfig.OFF,
                    dependency_optimization=rewriter_config_pb2.RewriterConfig.OFF,
                    memory_optimization=rewriter_config_pb2.RewriterConfig.NO_MEM_OPT,
                    arithmetic_optimization=rewriter_config_pb2.RewriterConfig.OFF,
                    shape_optimization=rewriter_config_pb2.RewriterConfig.OFF,
                    loop_optimization=rewriter_config_pb2.RewriterConfig.OFF,
                    function_optimization=rewriter_config_pb2.RewriterConfig.OFF,
                    remapping=rewriter_config_pb2.RewriterConfig.OFF,
                    implementation_selector=rewriter_config_pb2.RewriterConfig.OFF,
                    cpu_layout_conversion=convert,
                )
            )

            optimized_graph = tf_optimizer.OptimizeGraph(config, meta_graph)
            return optimized_graph

        return self.model
