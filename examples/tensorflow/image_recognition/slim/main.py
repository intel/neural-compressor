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
# from lpot.adaptor.tf_utils.util import write_graph
from nets_factory import TFSlimNetsFactory
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

  arg_parser.add_argument('--tune', dest='tune', action='store_true', help='use lpot to tune.')

  args = arg_parser.parse_args()

  factory = TFSlimNetsFactory()
  # user specific model can register to slim net factory
  input_shape = [None, 299, 299, 3]
  factory.register('inception_v4', inception_v4, input_shape, inception_v4_arg_scope)
  if args.input_graph.endswith('.ckpt'):
      # directly get the topology name from input_graph 
      topology = args.input_graph.rsplit('/', 1)[-1].split('.', 1)[0]
      # get the model func from net factory 
      assert topology in factory.default_slim_models, \
          'only support topology {}'.format(factory.default_slim_models)
      net = copy.deepcopy(factory.networks_map[topology])
      model_func = net.pop('model')
      arg_scope = net.pop('arg_scope')()
      inputs_shape = net.pop('input_shape')
      kwargs = net
      images = tf.compat.v1.placeholder(name='input', dtype=tf.float32, \
                                    shape=inputs_shape)
      from lpot.adaptor.tf_utils.util import get_slim_graph
      model = get_slim_graph(args.input_graph, model_func, arg_scope, images, **kwargs)
  else:
      model = args.input_graph

  if args.tune:

      from lpot import Quantization
      quantizer = Quantization(args.config)
      q_model = quantizer(model)
      save(q_model, args.output_graph)

  if args.benchmark:
      from lpot import Benchmark
      evaluator = Benchmark(args.config)
      results = evaluator(model=model)
      for mode, result in results.items():
          acc, batch_size, result_list = result
          latency = np.array(result_list).mean() / batch_size

          print('\n{} mode benchmark result:'.format(mode))
          print('Accuracy is {:.3f}'.format(acc))
          print('Batch size = {}'.format(batch_size))
          print('Latency: {:.3f} ms'.format(latency * 1000))
          print('Throughput: {:.3f} images/sec'.format(1./ latency))
  
if __name__ == '__main__':
  tf.compat.v1.app.run()
