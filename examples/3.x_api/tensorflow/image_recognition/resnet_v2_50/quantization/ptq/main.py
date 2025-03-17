#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

import tensorflow as tf
import numpy as np

from argparse import ArgumentParser
from data_process import (
    ImageRecordDataset, 
    ComposeTransform, 
    BilinearImagenetTransform, 
    TFDataLoader,
    TopKMetric,
)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

arg_parser = ArgumentParser(description='Parse args')
arg_parser.add_argument('-g', "--input-graph",
                        help='Specify the input graph for the transform tool',
                        dest='input_graph')
arg_parser.add_argument("--output-graph",
                        help='Specify tune result model save dir',
                        dest='output_graph')
arg_parser.add_argument('--benchmark', dest='benchmark', action='store_true', help='run benchmark')
arg_parser.add_argument('--mode', dest='mode', default='performance', help='benchmark mode')
arg_parser.add_argument('--tune', dest='tune', action='store_true', help='use neural_compressor to tune.')
arg_parser.add_argument('--diagnose', dest='diagnose', action='store_true', help='use Neural Insights to diagnose tuning and benchmark.')
arg_parser.add_argument('--dataset_location', dest='dataset_location',
                          help='location of calibration dataset and evaluate dataset')
arg_parser.add_argument('--batch_size', type=int, default=32, dest='batch_size', help='batch_size of benchmark')
arg_parser.add_argument('--iters', type=int, default=100, dest='iters', help='interations')
args = arg_parser.parse_args()

def evaluate(model, eval_dataloader, metric, postprocess=None):
    """Custom evaluate function to estimate the accuracy of the model.

    Args:
        model (tf.Graph_def): The input model graph
        
    Returns:
        accuracy (float): evaluation result, the larger is better.
    """
    from neural_compressor.tensorflow import Model
    model = Model(model)
    input_tensor = model.input_tensor
    output_tensor = model.output_tensor if len(model.output_tensor)>1 else \
                        model.output_tensor[0]
    iteration = -1
    if args.benchmark and args.mode == 'performance':
        iteration = args.iters

    def eval_func(dataloader):
        latency_list = []
        for idx, (inputs, labels) in enumerate(dataloader):
            # dataloader should keep the order and len of inputs same with input_tensor
            inputs = np.array([inputs])
            feed_dict = dict(zip(input_tensor, inputs))

            start = time.time()
            predictions = model.sess.run(output_tensor, feed_dict)
            end = time.time()

            metric.update(predictions, labels)
            latency_list.append(end-start)
            if idx + 1 == iteration:
                break
        latency = np.array(latency_list).mean() / args.batch_size
        return latency

    latency = eval_func(eval_dataloader)
    if args.benchmark and args.mode == 'performance':
        print("Batch size = {}".format(args.batch_size))
        print("Latency: {:.3f} ms".format(latency * 1000))
        print("Throughput: {:.3f} images/sec".format(1. / latency))
    acc = metric.result()
    return acc

class eval_classifier_optimized_graph:
    """Evaluate image classifier with optimized TensorFlow graph."""

    def run(self):
        """This is neural_compressor function include tuning, export and benchmark option."""
        from neural_compressor.common import set_random_seed
        set_random_seed(9527)

        if args.tune:
            from neural_compressor.tensorflow import StaticQuantConfig, quantize_model

            dataset = ImageRecordDataset(
                root=args.dataset_location, 
                transform=ComposeTransform(transform_list= [
                    BilinearImagenetTransform(height=224, width=224),
                    ]
                )
            )
            calib_dataloader = TFDataLoader(dataset=dataset, batch_size=10)

            quant_config = StaticQuantConfig()
            q_model = quantize_model(args.input_graph, quant_config, calib_dataloader)
            q_model.save(args.output_graph)

        if args.benchmark:
            dataset = ImageRecordDataset(
                root=args.dataset_location, 
                transform=ComposeTransform(transform_list= [
                    BilinearImagenetTransform(height=224, width=224),
                    ]
                )
            )
            dataloader = TFDataLoader(dataset=dataset, batch_size=args.batch_size)

            def eval(model):
                top1 = TopKMetric(k=1)
                return evaluate(model, dataloader, top1)

            if args.mode == 'performance':
                eval(args.input_graph)
            elif args.mode == 'accuracy':
                acc_result = eval(args.input_graph)
                print("Batch size = %d" % dataloader.batch_size)
                print("Accuracy: %.5f" % acc_result)

if __name__ == "__main__":
    evaluate_opt_graph = eval_classifier_optimized_graph()
    evaluate_opt_graph.run()
