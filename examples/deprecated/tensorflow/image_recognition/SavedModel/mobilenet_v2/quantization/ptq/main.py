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
#

from __future__ import division
import time
import tensorflow as tf
import numpy as np
from argparse import ArgumentParser

arg_parser = ArgumentParser(description='Parse args')
arg_parser.add_argument('-g', "--input-graph",
                        help='Specify the input graph for the transform tool',
                        dest='input_graph')
arg_parser.add_argument("--output-graph",
                        help='Specify tune result model save dir',
                        dest='output_graph')
arg_parser.add_argument('--benchmark', dest='benchmark', action='store_true', help='run benchmark')
arg_parser.add_argument('--mode', dest='mode', default='performance', help='benchmark mode')
arg_parser.add_argument('--export', dest='export', action='store_true', help='use neural_compressor to export.')
arg_parser.add_argument('--tune', dest='tune', action='store_true', help='use neural_compressor to tune.')
arg_parser.add_argument('--dataset_location', dest='dataset_location',
                            help='location of calibration dataset and evaluate dataset')
arg_parser.add_argument('--batch_size', type=int, default=32, dest='batch_size', help='batch_size of evaluation')
arg_parser.add_argument('--iters', type=int, default=100, dest='iters', help='interations')
args = arg_parser.parse_args()

def evaluate(model):
    """Custom evaluate function to estimate the accuracy of the model.
    Args:
        model (tf.Graph_def): The input model graph
        
    Returns:
        accuracy (float): evaluation result, the larger is better.
    """
    infer = model.signatures["serving_default"]
    output_dict_keys = infer.structured_outputs.keys()
    output_name = list(output_dict_keys )[0]
    from neural_compressor import METRICS
    metrics = METRICS('tensorflow')
    metric = metrics['topk']()

    def eval_func(dataloader, metric):
        warmup = 5
        iteration = None
        latency_list = []
        if args.benchmark and args.mode == 'performance':
            iteration = args.iters
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = np.array(inputs)
            input_tensor = tf.constant(inputs)
            start = time.time()
            predictions = infer(input_tensor)[output_name]
            end = time.time()
            predictions = predictions.numpy()
            metric.update(predictions, labels)
            latency_list.append(end - start)
            if iteration and idx >= iteration:
                break
        latency = np.array(latency_list[warmup:]).mean() / eval_dataloader.batch_size
        return latency

    from neural_compressor.utils.create_obj_from_config import create_dataloader
    dataloader_args = {
        'batch_size': args.batch_size,
        'dataset': {"ImageRecord": {'root': args.dataset_location}},
        'transform': {'BilinearImagenet': {'height': 224, 'width': 224}},
        'filter': None
    }
    eval_dataloader = create_dataloader('tensorflow', dataloader_args)
    latency = eval_func(eval_dataloader, metric)
    if args.benchmark and args.mode == 'performance':
        print("Batch size = {}".format(eval_dataloader.batch_size))
        print("Latency: {:.3f} ms".format(latency * 1000))
        print("Throughput: {:.3f} images/sec".format(1. / latency))
    acc = metric.result()
    return acc

class eval_object_detection_optimized_graph(object):
    def run(self):
        from neural_compressor import set_random_seed
        set_random_seed(9527)
        if args.tune:
            from neural_compressor import quantization
            from neural_compressor.config import PostTrainingQuantConfig
            from neural_compressor.utils.create_obj_from_config import create_dataloader
            calib_dataloader_args = {
                'batch_size': 10,
                'dataset': {"ImageRecord": {'root':args.dataset_location}},
                'transform': {'BilinearImagenet':
                     {'height': 224, 'width': 224}},
                'filter': None
            }
            calib_dataloader = create_dataloader('tensorflow', calib_dataloader_args)
            conf = PostTrainingQuantConfig(calibration_sampling_size=[20, 50])
            q_model = quantization.fit(model=args.input_graph, conf=conf,
                                       calib_dataloader=calib_dataloader, eval_func=evaluate)
            q_model.save(args.output_graph)

        if args.benchmark:
            from neural_compressor.benchmark import fit
            from neural_compressor.config import BenchmarkConfig
            if args.mode == 'performance':
                conf = BenchmarkConfig(cores_per_instance=4, num_of_instance=1)
                from neural_compressor.utils.create_obj_from_config import create_dataloader
                dataloader_args = {
                    'batch_size': args.batch_size,
                    'dataset': {"ImageRecord": {'root': args.dataset_location}},
                    'transform': {'BilinearImagenet': {'height': 224, 'width': 224}},
                    'filter': None
                }
                eval_dataloader = create_dataloader('tensorflow', dataloader_args)
                fit(model=args.input_graph, conf=conf, b_dataloader=eval_dataloader)
            else:
                from neural_compressor.model import Model
                model = Model(args.input_graph).model
                accuracy = evaluate(model)
                print('Batch size = %d' % args.batch_size)
                print("Accuracy: %.5f" % accuracy)

if __name__ == "__main__":
    evaluate_opt_graph = eval_object_detection_optimized_graph()
    evaluate_opt_graph.run()
