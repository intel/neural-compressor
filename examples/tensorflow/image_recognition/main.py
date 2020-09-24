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

    arg_parser.add_argument("--int8-input", type=bool, default=False,
                            help="use int8 input",
                            dest='int8_input')

    arg_parser.add_argument("--warmup_steps", type=int, default=10,
                            help="skip number of steps")

    arg_parser.add_argument("--config", default=None, help="tuning config")

    arg_parser.add_argument('--benchmark', dest='benchmark', action='store_true', help='run benchmark')

    arg_parser.add_argument('--tune', dest='tune', action='store_true', help='use ilit to tune.')

    self.args = arg_parser.parse_args()

  def run(self):
      """ This is ilit function include tuning and benchmark option """

      if self.args.tune:
          import ilit
          tuner = ilit.Tuner(self.args.config)
          q_model = tuner.tune(self.args.input_graph)

          if self.args.int8_input:
              from ilit.adaptor.tf_utils.util import remove_quantize_op
              q_model = remove_quantize_op(q_model)

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
          import ilit 
          evaluator = ilit.Benchmark(self.args.config)
          acc, batch_size, measurer = \
              evaluator.benchmark(model=self.args.input_graph)

          print('Accuracy is {:.3f}'.format(acc))
          print('Batch size = {}'.format(batch_size))

          latency = measurer.result(self.args.warmup_steps) / batch_size

          print('Latency: {:.3f} ms'.format(latency * 1000))
          print('Throughput: {:.3f} images/sec'.format(1./ latency))

if __name__ == "__main__":

  evaluate_opt_graph = eval_classifier_optimized_graph()
  evaluate_opt_graph.run()
