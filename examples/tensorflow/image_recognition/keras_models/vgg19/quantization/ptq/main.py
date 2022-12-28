#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
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

import time
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class eval_classifier_optimized_graph:
  """Evaluate image classifier with optimized TensorFlow graph"""

  def __init__(self):

    arg_parser = ArgumentParser(description='Parse args')

    arg_parser.add_argument('-g', "--input-graph",
                            help='Specify the input graph for the transform tool',
                            dest='input_graph')

    arg_parser.add_argument("--output-graph",
                            help='Specify tune result model save dir',
                            dest='output_graph')

    arg_parser.add_argument("--config", default=None, help="tuning config")

    arg_parser.add_argument('--benchmark', dest='benchmark', action='store_true', help='run benchmark')

    arg_parser.add_argument('--tune', dest='tune', action='store_true', help='use neural_compressor to tune.')
    arg_parser.add_argument('--mode', dest='mode', default='performance', help='benchmark mode, support performance and accuracy')

    self.args = arg_parser.parse_args()

  def run(self):
      """ This is neural_compressor function include tuning and benchmark option """

      if self.args.tune:
          from neural_compressor.experimental import Quantization, common
          quantizer = Quantization(self.args.config)
          quantizer.model = common.Model(self.args.input_graph)
          q_model = quantizer.fit()
          q_model.save(self.args.output_graph)

      if self.args.benchmark:
          from neural_compressor.experimental import Benchmark, common
          evaluator = Benchmark(self.args.config)
          evaluator.model = common.Model(self.args.input_graph)
          evaluator(self.args.mode)

if __name__ == "__main__":

  evaluate_opt_graph = eval_classifier_optimized_graph()
  evaluate_opt_graph.run()
