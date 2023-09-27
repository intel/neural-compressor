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
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from neural_compressor.data import DataLoader
from neural_compressor.metric import COCOmAPv2
from neural_compressor.data import COCORecordDataset
from neural_compressor.data import ComposeTransform, ResizeTFTransform

arg_parser = ArgumentParser(description='Parse args')

arg_parser.add_argument('-g',
                    "--input-graph",
                    help='Specify the input graph.',
                    dest='input_graph')
arg_parser.add_argument('--config', type=str, default='')
arg_parser.add_argument('--dataset_location', type=str, default='')
arg_parser.add_argument('--output_model', type=str, default='')
arg_parser.add_argument('--mode', type=str, default='performance')
arg_parser.add_argument('--batch_size', type=int, default=10)
arg_parser.add_argument('--iters', type=int, default=100, dest='iters', help='iterations')
arg_parser.add_argument('--tune', action='store_true', default=False)
arg_parser.add_argument('--benchmark', dest='benchmark',
                    action='store_true', help='run benchmark')
args = arg_parser.parse_args()

def evaluate(model):
    """Custom evaluate function to estimate the accuracy of the model.

    Args:
        model (tf.Graph or string or INC.model.TensorflowCheckpointModel): The input model.
        
    Returns:
        accuracy (float): evaluation result, the larger is better.
    """
    from neural_compressor.model import Model
    if isinstance(model, str) or isinstance(model, tf.compat.v1.Graph):
        model = Model(model)
        model.input_tensor_names = ["image_tensor:0"]
        model.output_tensor_names = ["num_detections:0", "detection_boxes:0", \
                                        "detection_scores:0", "detection_classes:0"]
    input_tensor = model.input_tensor
    output_tensor = model.output_tensor if len(model.output_tensor)>1 else \
                        model.output_tensor[0]
    warmup = 5
    iteration = -1
    if args.benchmark and args.mode == 'performance':
        iteration = args.iters
    metric = COCOmAPv2(output_index_mapping={'num_detections':0, 'boxes':1, 'scores':2, 'classes':3})

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
        latency = np.array(latency_list[warmup:]).mean() / args.batch_size
        return latency

    eval_dataset = COCORecordDataset(root=args.dataset_location, filter=None, \
            transform=ComposeTransform(transform_list=[ResizeTFTransform(size=640)]))
    eval_dataloader=DataLoader(framework='tensorflow', dataset=eval_dataset, batch_size=args.batch_size)
    latency = eval_func(eval_dataloader)
    if args.benchmark and args.mode == 'performance':
        print("Batch size = {}".format(args.batch_size))
        print("Latency: {:.3f} ms".format(latency * 1000))
        print("Throughput: {:.3f} images/sec".format(1. / latency))
    acc = metric.result()
    return acc 

def main(_):
    calib_dataset = COCORecordDataset(root=args.dataset_location, filter=None, \
            transform=ComposeTransform(transform_list=[ResizeTFTransform(size=640)]))
    calib_dataloader = DataLoader(framework='tensorflow', dataset=calib_dataset)

    if args.tune:
        from neural_compressor import quantization
        from neural_compressor.config import PostTrainingQuantConfig
        op_name_dict = {
                'FeatureExtractor/resnet_v1_50/fpn/bottom_up_block5/Conv2D': {
                    'activation':  {'dtype': ['fp32']},
                },
                'WeightSharedConvolutionalBoxPredictor_2/BoxPredictionTower/conv2d_0/Conv2D': {
                    'activation':  {'dtype': ['fp32']},
                },
                'WeightSharedConvolutionalBoxPredictor_2/ClassPredictionTower/conv2d_0/Conv2D': {
                    'activation':  {'dtype': ['fp32']},
                },
            }
        # only for TF newAPI
        if tf.version.VERSION in ['2.11.0202242', '2.11.0202250', '2.11.0202317', '2.11.0202323']:
            config = PostTrainingQuantConfig(
                inputs=["image_tensor"],
                outputs=["num_detections", "detection_boxes", "detection_scores", "detection_classes"],
                calibration_sampling_size=[10, 50, 100, 200],
                op_name_dict=op_name_dict)
        else:
            config = PostTrainingQuantConfig(
                inputs=["image_tensor"],
                outputs=["num_detections", "detection_boxes", "detection_scores", "detection_classes"],
                calibration_sampling_size=[10, 50, 100, 200])
        q_model = quantization.fit(model=args.input_graph, conf=config, 
                                    calib_dataloader=calib_dataloader, eval_func=evaluate)
        q_model.save(args.output_model)
            
    if args.benchmark:
        from neural_compressor.benchmark import fit
        from neural_compressor.config import BenchmarkConfig
        if args.mode == 'performance':
            conf = BenchmarkConfig(
                inputs=["image_tensor"],
                outputs=["num_detections", "detection_boxes", "detection_scores", "detection_classes"],
                cores_per_instance=4,
                num_of_instance=1)
            fit(args.input_graph, conf, b_func=evaluate)
        else:
            accuracy = evaluate(args.input_graph)
            print('Batch size = %d' % args.batch_size)
            print("Accuracy: %.5f" % accuracy)

if __name__ == "__main__":
	tf.compat.v1.app.run()
