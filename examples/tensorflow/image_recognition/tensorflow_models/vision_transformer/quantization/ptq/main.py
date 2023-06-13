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
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes
from tensorflow.core.protobuf import saved_model_pb2

import datasets
import numpy as np

INPUTS = 'inputs'
OUTPUTS = 'Identity'

RESNET_IMAGE_SIZE = 224
IMAGENET_VALIDATION_IMAGES = 50000

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

class eval_classifier_optimized_graph:
    """Evaluate image classifier with optimized TensorFlow graph."""

    def run(self):
        """This is neural_compressor function include tuning, export and benchmark option."""
        from neural_compressor import set_random_seed
        set_random_seed(9527)

        if args.tune:
            from neural_compressor import quantization
            from neural_compressor.config import PostTrainingQuantConfig, AccuracyCriterion
            from neural_compressor.utils.create_obj_from_config import create_dataloader
            calib_dataloader_args = {
                'batch_size': 10,
                'dataset': {"ImageRecord": {'root':args.dataset_location}},
                'transform': {'ResizeCropImagenet':
                     {'height': 224, 'width': 224}, 'TransposeShiftRescale':{}},
                'filter': None
            }
            calib_dataloader = create_dataloader('tensorflow', calib_dataloader_args)
            eval_dataloader_args = {
                'batch_size': 32,
                'dataset': {"ImageRecord": {'root':args.dataset_location}},
                'transform': {'ResizeCropImagenet':
                     {'height': 224, 'width': 224}, 'TransposeShiftRescale':{}},
                'filter': None
            }
            eval_dataloader = create_dataloader('tensorflow', eval_dataloader_args)

            conf = PostTrainingQuantConfig(calibration_sampling_size=[50, 100],
                                           accuracy_criterion = AccuracyCriterion(tolerable_loss=0.01),
                                           backend='itex',
                                           op_type_dict={'conv2d':{ 'weight':{'dtype':['fp32']}, 'activation':{'dtype':['fp32']} }})
            from neural_compressor import Metric
            top1 = Metric(name="topk", k=1)
            from tensorflow.core.protobuf import saved_model_pb2
            sm = saved_model_pb2.SavedModel()
            with tf.io.gfile.GFile(args.input_graph, "rb") as f:
                sm.ParseFromString(f.read())
            graph_def = sm.meta_graphs[0].graph_def
            q_model = quantization.fit(graph_def, conf=conf, calib_dataloader=calib_dataloader,
                        eval_dataloader=eval_dataloader, eval_metric=top1)
            q_model.save(args.output_graph)

        # TODO let itex check
        # if args.benchmark:
        #     from neural_compressor.utils.create_obj_from_config import create_dataloader
        #     dataloader_args = {
        #         'batch_size': args.batch_size,
        #         'dataset': {"ImageRecord": {'root':args.dataset_location}},
        #         'transform': {'ResizeCropImagenet': {'height': 224, 'width': 224}},
        #         'filter': None
        #     }
        #     dataloader = create_dataloader('tensorflow', dataloader_args)
        #     from neural_compressor import METRICS
        #     metrics = METRICS('tensorflow')
        #     top1 = metrics['topk']()
        #     def eval(model):
        #         return evaluate(model, dataloader, top1)

        #     if args.mode == 'performance':
        #         from neural_compressor.benchmark import fit
        #         from neural_compressor.config import BenchmarkConfig
        #         conf = BenchmarkConfig(warmup=10, iteration=100, cores_per_instance=4, num_of_instance=1)
        #         fit(args.input_graph, conf, b_dataloader=dataloader)
        #     elif args.mode == 'accuracy':
        #         acc_result = eval(args.input_graph)
        #         print("Batch size = %d" % dataloader.batch_size)
        #         print("Accuracy: %.5f" % acc_result)

if __name__ == "__main__":
    evaluate_opt_graph = eval_classifier_optimized_graph()
    evaluate_opt_graph.run()