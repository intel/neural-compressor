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
from tensorflow.python.training import saver as saver_lib
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.grappler import tf_optimizer
from tensorflow.core.protobuf import meta_graph_pb2
import tensorflow as tf

class ConvertLayoutOptimizer(GraphRewriterBase):
    """ The layout convertion optimizer, convert NCHW to NHWC format.
        It is executed only when NCHW node exists and tensorflow version is 2.4.0 and above.

    Args: model: input graph_def
          outputs: output name list

    Return: converted graph_def
    """

    def __init__(self, model, outputs):
        super().__init__(model)
        self.outputs = outputs

    @dump_elapsed_time("ConvertLayoutOptimizer")
    def do_transformation(self):
        convert = False
        for node in self.model.node:
            if 'Conv' in node.op and \
               'data_format' in node.attr and \
               node.attr['data_format'].s == b'NCHW':
                convert = True
                break
        if convert:
            assert tf.version.VERSION >= '2.4.0', 'layout convert is only supported by \
                                                            tensorflow 2.4.0 and above'

            g = tf.Graph()
            with g.as_default(): # pylint: disable=not-context-manager
                g = tf.compat.v1.import_graph_def(self.model, name='')
                meta_graph = saver_lib.export_meta_graph(
                    graph_def=self.model, graph=g, clear_devices=True)
                fetch_collection = meta_graph_pb2.CollectionDef()
                for fetch in self.outputs:
                    fetch_collection.node_list.value.append(fetch) # pylint: disable=no-member
                meta_graph.collection_def["train_op"].CopyFrom( # pylint: disable=no-member
                                                    fetch_collection) # pylint: disable=no-member
            config = config_pb2.ConfigProto()
            convert = rewriter_config_pb2.RewriterConfig.NCHW_TO_NHWC # pylint: disable=no-member
            config.graph_options.rewrite_options.CopyFrom( # pylint: disable=no-member
                rewriter_config_pb2.RewriterConfig(
                    cpu_layout_conversion=convert))
            optimized_graph = tf_optimizer.OptimizeGraph(config, meta_graph)
            return optimized_graph
        else:
            return self.model
