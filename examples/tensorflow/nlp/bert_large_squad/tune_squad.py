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
        q_model.save(FLAGS.output_model)

if __name__ == "__main__":
    tf.compat.v1.app.run()
