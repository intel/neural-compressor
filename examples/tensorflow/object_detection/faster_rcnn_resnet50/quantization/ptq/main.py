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
from data_process import(
    COCOmAPv2,
    COCORecordDataset,
    ComposeTransform,
    ResizeTFTransform,
    TFDataLoader,
)

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
        model (tf.Graph): The input model graph.
        
    Returns:
        accuracy (float): evaluation result, the larger is better.
    """
    from neural_compressor.tensorflow import Model
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
            transform=ComposeTransform(transform_list=[ResizeTFTransform(size=600)]))
    eval_dataloader=TFDataLoader(dataset=eval_dataset, batch_size=args.batch_size)
    latency = eval_func(eval_dataloader)
    if args.benchmark and args.mode == 'performance':
        print("Batch size = {}".format(args.batch_size))
        print("Latency: {:.3f} ms".format(latency * 1000))
        print("Throughput: {:.3f} images/sec".format(1. / latency))
    acc = metric.result()
    return acc 

def main(_):
    calib_dataset = COCORecordDataset(root=args.dataset_location, filter=None, \
            transform=ComposeTransform(transform_list=[ResizeTFTransform(size=600)]))
    calib_dataloader = TFDataLoader(dataset=calib_dataset, batch_size=args.batch_size)

    if args.tune:
        from neural_compressor.tensorflow import StaticQuantConfig, quantize_model, Model

        quant_config = StaticQuantConfig(weight_granularity="per_channel")
        model = Model(args.input_graph)
        model.input_tensor_names = ['image_tensor']
        model.output_tensor_names = ["num_detections", "detection_boxes", "detection_scores", "detection_classes"]
        q_model = quantize_model(model, quant_config, calib_dataloader)
        q_model.save(args.output_model)
            
    if args.benchmark:
        if args.mode == 'performance':
            evaluate(args.input_graph)
        else:
            accuracy = evaluate(args.input_graph)
            print('Batch size = %d' % args.batch_size)
            print("Accuracy: %.5f" % accuracy)

if __name__ == "__main__":
	tf.compat.v1.app.run()
