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
tf.compat.v1.disable_eager_execution()

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

    arg_parser.add_argument('--tune', dest='tune', action='store_true', help='use lpot to tune.')

    self.args = arg_parser.parse_args()

  def run(self):
      """ This is lpot function include tuning and benchmark option """

      if self.args.tune:
          from lpot import Quantization
          quantizer = Quantization(self.args.config)
          q_model = quantizer(self.args.input_graph)

          def save(model, path):
              from tensorflow.python.platform import gfile
              f = gfile.GFile(path, 'wb')
              f.write(model.as_graph_def().SerializeToString())

          try:
            save(q_model, evaluate_opt_graph.args.output_graph)
          except AttributeError as no_model:
              print("None of the quantized models fits the \
                     accuracy criteria: {0}".format(no_model))
          except Exception as exc:
              print("Unexpected error while saving the model: {0}".format(exc))

      if self.args.benchmark:
          from lpot import Benchmark
          evaluator = Benchmark(self.args.config)
          results = evaluator(model=self.args.input_graph)
          for mode, result in results.items():
              acc, batch_size, result_list = result
              latency = np.array(result_list).mean() / batch_size

              print('\n{} mode benchmark result:'.format(mode))
              print('Accuracy is {:.3f}'.format(acc))
              print('Batch size = {}'.format(batch_size))
              print('Latency: {:.3f} ms'.format(latency * 1000))
              print('Throughput: {:.3f} images/sec'.format(1./ latency))

if __name__ == "__main__":

  evaluate_opt_graph = eval_classifier_optimized_graph()
  evaluate_opt_graph.run()
