#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#

import numpy as np
from argparse import ArgumentParser
import tensorflow as tf
from neural_compressor.model.nets_factory import TFSlimNetsFactory
import copy

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_eager_execution()
from inception_v4 import inception_v4, inception_v4_arg_scope

def save(model, path):
    from tensorflow.python.platform import gfile
    f = gfile.GFile(path, 'wb')
    try:
        f.write(model.as_graph_def().SerializeToString())
    except AttributeError as no_model:
        print("None of the quantized models fits the \
               accuracy criteria: {0}".format(no_model))
    except Exception as exc:
        print("Unexpected error while saving the model: {0}".format(exc))

def main(_):
  arg_parser = ArgumentParser(description='Parse args')

  arg_parser.add_argument("--input-graph",
                          help='Specify the slim model',
                          dest='input_graph')

  arg_parser.add_argument("--output-graph",
                          help='Specify tune result model save dir',
                          dest='output_graph')

  arg_parser.add_argument("--config", default=None, help="tuning config")

  arg_parser.add_argument('--benchmark', dest='benchmark', action='store_true', help='run benchmark')

  arg_parser.add_argument('--mode', dest='mode', default='performance', help='benchmark mode')

  arg_parser.add_argument('--tune', dest='tune', action='store_true', help='use neural_compressor to tune.')

  args = arg_parser.parse_args()

  factory = TFSlimNetsFactory()
  # user specific model can register to slim net factory
  input_shape = [None, 299, 299, 3]
  factory.register('inception_v4', inception_v4, input_shape, inception_v4_arg_scope)

  if args.tune:
      from neural_compressor.experimental import Quantization
      quantizer = Quantization(args.config)
      quantizer.model = args.input_graph
      q_model = quantizer.fit()
      q_model.save(args.output_graph)

  if args.benchmark:
      from neural_compressor.experimental import Benchmark
      evaluator = Benchmark(args.config)
      evaluator.model = args.input_graph
      evaluator(args.mode)
  
if __name__ == '__main__':
  tf.compat.v1.app.run()
