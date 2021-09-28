#!/usr/bin/env python

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

import tensorflow as tf
import numpy as np

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    'input_model', None, 'Run inference with specified pb graph.')

flags.DEFINE_string(
    'output_model', None, 'The output model of the quantized model.')

flags.DEFINE_string(
    'mode', 'performance', 'define benchmark mode for accuracy or performance')

flags.DEFINE_bool(
    'tune', False, 'whether to tune the model')

flags.DEFINE_bool(
    'benchmark', False, 'whether to benchmark the model')

flags.DEFINE_string(
    'config', 'bert.yaml', 'yaml configuration of the model')

flags.DEFINE_bool(
    'strip_iterator', False, 'whether to strip the iterator of the model')

def strip_iterator(graph_def):
    from lpot.adaptor.tf_utils.util import strip_unused_nodes
    input_node_names = ['input_ids', 'input_mask', 'segment_ids']
    output_node_names = ['unstack']
    # create the placeholder and merge with the graph
    with tf.compat.v1.Graph().as_default() as g: 
        input_ids = tf.compat.v1.placeholder(tf.int32, shape=(None,384), name="input_ids")
        input_mask = tf.compat.v1.placeholder(tf.int32, shape=(None,384), name="input_mask")
        segment_ids = tf.compat.v1.placeholder(tf.int32, shape=(None,384), name="segment_ids")
        tf.import_graph_def(graph_def, name='')

    graph_def = g.as_graph_def()
    # change the input from iterator to placeholder
    for node in graph_def.node:
        for idx, in_tensor in enumerate(node.input):
            if 'IteratorGetNext:0' == in_tensor or 'IteratorGetNext' == in_tensor:
                node.input[idx] = 'input_ids'
            if 'IteratorGetNext:1' in in_tensor:
                node.input[idx] = 'input_mask'
            if 'IteratorGetNext:2' in in_tensor:
                node.input[idx] = 'segment_ids'

    graph_def = strip_unused_nodes(graph_def, input_node_names, output_node_names)
    return graph_def

def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    if FLAGS.benchmark:
        from lpot.experimental import Benchmark
        evaluator = Benchmark(FLAGS.config)
        evaluator.model = FLAGS.input_model
        evaluator(FLAGS.mode)

    elif FLAGS.tune:
        from lpot.experimental import Quantization
        quantizer = Quantization(FLAGS.config)
        quantizer.model = FLAGS.input_model
        q_model = quantizer()
        if FLAGS.strip_iterator:
            q_model.graph_def = strip_iterator(q_model.graph_def)
        q_model.save(FLAGS.output_model)

if __name__ == "__main__":
    tf.compat.v1.app.run()
