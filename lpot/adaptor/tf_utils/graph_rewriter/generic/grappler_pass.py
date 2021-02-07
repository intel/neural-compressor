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

from lpot.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase

from tensorflow.python.training import saver
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.grappler import tf_optimizer
from tensorflow.core.protobuf import meta_graph_pb2
import tensorflow as tf


class GrapplerOptimizer(GraphRewriterBase):
    """A python wrapper that leverages the built-in tensorflow grappler API to optimize the graph.
    """
    def __init__(self, model, outputs):
        super().__init__(model)
        self.outputs = outputs

    @dump_elapsed_time("Pass GrapplerOptimizer")
    def do_transformation(self):
        try:
            g = tf.Graph()
            with g.as_default():
                g = tf.compat.v1.import_graph_def(self.model, name='')
                meta_graph = saver.export_meta_graph(
                    graph_def=self.model, graph=g, clear_devices=True)
                fetch_collection = meta_graph_pb2.CollectionDef()
                for fetch in self.outputs:
                    fetch_collection.node_list.value.append(fetch)
                meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)
                config = config_pb2.ConfigProto()
                rewriter_config = config.graph_options.rewrite_options
                rewriter_config.optimizers.append('pruning')
                rewriter_config.optimizers.append('dependency')
                rewriter_config.optimizers.append('debug_stripper')
                rewriter_config.optimizers.append('loop')
                rewriter_config.min_graph_nodes = -1

                optimized_graph = tf_optimizer.OptimizeGraph(config, meta_graph)

            return optimized_graph
        except Exception as e:
            self.logger.warning("Failed to run grappler pass due to {}".format(str(e)))
            return self.model
