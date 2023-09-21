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
"""Tensorflow Grappler Graph Rewriter."""

import tensorflow as tf
from tensorflow.core.protobuf import config_pb2, meta_graph_pb2
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training import saver

from neural_compressor.adaptor.tf_utils.util import version1_gt_version2
from neural_compressor.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase


class GrapplerOptimizer(GraphRewriterBase):
    """A python wrapper that leverages the built-in tensorflow grappler API to optimize the graph."""

    def __init__(self, model, input_output_names, opt_cfg):
        """Initialization."""
        super().__init__(model)
        self.input_output_names = input_output_names
        self.opt_cfg = opt_cfg
        self.generic_optimizer = ("pruning", "shape", "dependency", "debug_stripper", "loop")
        self.tf_2_optimizer = ("constfold", "arithmetic", "min_graph_nodes")

    @dump_elapsed_time("Pass GrapplerOptimizer")
    def do_transformation(self):
        """Apply tensorflow Grappler optimization."""
        try:
            g = tf.Graph()
            with g.as_default():
                g = tf.compat.v1.import_graph_def(self.model, name="")
                meta_graph = saver.export_meta_graph(graph_def=self.model, graph=g, clear_devices=True)
                fetch_collection = meta_graph_pb2.CollectionDef()
                for fetch in self.input_output_names:
                    fetch_collection.node_list.value.append(fetch)
                meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)
                config = config_pb2.ConfigProto()
                rewriter_config = config.graph_options.rewrite_options
                for optimizer in self.generic_optimizer:
                    if optimizer in self.opt_cfg and self.opt_cfg[optimizer]:
                        rewriter_config.optimizers.append(optimizer)

                if version1_gt_version2(tf.version.VERSION, "2.2.0"):
                    for optimizer in self.tf_2_optimizer:
                        if optimizer in self.opt_cfg and self.opt_cfg[optimizer]:
                            rewriter_config.optimizers.append(optimizer)

                rewriter_config.min_graph_nodes = -1

                optimized_graph = tf_optimizer.OptimizeGraph(config, meta_graph)

            return optimized_graph
        except Exception as e:
            self.logger.warning("Fail to run grappler pass due to {}.".format(str(e)))
            return self.model
