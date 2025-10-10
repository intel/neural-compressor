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

from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import time

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
    from neural_compressor.model import Model
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

            if postprocess:
                predictions, labels = postprocess((predictions, labels))

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
        from neural_compressor import set_random_seed
        set_random_seed(9527)

        if args.tune:
            from neural_compressor import quantization
            from neural_compressor.config import PostTrainingQuantConfig
            from neural_compressor.utils.create_obj_from_config import create_dataloader
            calib_dataloader_args = {
                'batch_size': 10,
                'dataset': {"ImageRecord": {'root':args.dataset_location}},
                'transform': {'ResizeCropImagenet':
                     {'height': 224, 'width': 224, 'scale': 0.017,
                      'mean_value': [123.68, 116.78, 103.94]}},
                'filter': None
            }
            calib_dataloader = create_dataloader('tensorflow', calib_dataloader_args)
            eval_dataloader_args = {
                'batch_size': 32,
                'dataset': {"ImageRecord": {'root':args.dataset_location}},
                'transform': {'ResizeCropImagenet':
                     {'height': 224, 'width': 224, 'scale': 0.017,
                      'mean_value': [123.68, 116.78, 103.94]}},
                'filter': None
            }
            eval_dataloader = create_dataloader('tensorflow', eval_dataloader_args)
            conf = PostTrainingQuantConfig(calibration_sampling_size=[50, 100])
            from neural_compressor import METRICS
            metrics = METRICS('tensorflow')
            top1 = metrics['topk']()
            from neural_compressor.data import LabelShift
            postprocess = LabelShift(label_shift=1)
            def eval(model):
                return evaluate(model, eval_dataloader, top1, postprocess)
            q_model = quantization.fit(args.input_graph, conf=conf, calib_dataloader=calib_dataloader,
                        eval_func=eval)
            q_model.save(args.output_graph)

        if args.benchmark:
            from neural_compressor.utils.create_obj_from_config import create_dataloader
            dataloader_args = {
                'batch_size': args.batch_size,
                'dataset': {"ImageRecord": {'root':args.dataset_location}},
                'transform': {'ResizeCropImagenet': {'height': 224, 'width': 224, 'scale': 0.017,
                                                     'mean_value': [123.68, 116.78, 103.94]}},
                'filter': None
            }
            dataloader = create_dataloader('tensorflow', dataloader_args)
            from neural_compressor import METRICS
            metrics = METRICS('tensorflow')
            top1 = metrics['topk']()
            from neural_compressor.data import LabelShift
            postprocess = LabelShift(label_shift=1)
            def eval(model):
                return evaluate(model, dataloader, top1, postprocess)

            if args.mode == 'performance':
                from neural_compressor.benchmark import fit
                from neural_compressor.config import BenchmarkConfig
                conf = BenchmarkConfig(warmup=10, iteration=100, cores_per_instance=4, num_of_instance=1)
                fit(args.input_graph, conf, b_dataloader=dataloader)
            elif args.mode == 'accuracy':
                acc_result = eval(args.input_graph)
                print("Batch size = %d" % dataloader.batch_size)
                print("Accuracy: %.5f" % acc_result)

if __name__ == "__main__":
    evaluate_opt_graph = eval_classifier_optimized_graph()
    evaluate_opt_graph.run()
