#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2020 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import preprocessing
import datasets
from tensorflow.core.framework import graph_pb2

from google.protobuf import text_format
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))

from ilit import tuner as iLit


def load_graph(model_file):

    graph = tf.Graph()
    graph_def = tf.GraphDef()

    if not isinstance(model_file, graph_pb2.GraphDef):
        file_ext = os.path.splitext(model_file)[1]

        with open(model_file, "rb") as f:
            if file_ext == '.pbtxt':
                text_format.Merge(f.read(), graph_def)
            else:
                graph_def.ParseFromString(f.read())

        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
    else:
        with graph.as_default():
            tf.import_graph_def(model_file, name='')

    return graph





class Dataloader(object):
    '''
    This is a example class that wrapped the model specified parameters,
    such as dataset path, batch size.
    And more importantly, it provides the ability to iterate the dataset.
    '''
    def __init__(self, data_location, subset, input_height, input_width,
                 batch_size):
        self.batch_size = batch_size
        self.subset = subset
        self.dataset = datasets.ImagenetData(data_location)
        self.total_image = self.dataset.num_examples_per_epoch(self.subset)
        self.preprocessor = preprocessing.ImagePreprocessor(
            input_height,
            input_width,
            batch_size,
            1,
            tf.float32,
            train=False,
            resize_method='crop')
        self.n = int(self.total_image / self.batch_size)

    def __iter__(self):
        images, labels = self.preprocessor.minibatch(self.dataset, self.subset)
        with tf.compat.v1.Session() as sess:
            for i in range(self.n):
                yield sess.run([images[0], labels[0]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tensorflow Resnet50-v1.0 demo for iLit')
    parser.add_argument('--input_graph', type=str, default='')
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--inputs', type=str, default='', help='input tensor')
    parser.add_argument('--outputs',
                        type=str,
                        required='',
                        help='output tensor')
    parser.add_argument('--data_location',
                        type=str,
                        required='',
                        help='dataset location')
    parser.add_argument('--input_height',
                        type=int,
                        default=224,
                        help='input height')
    parser.add_argument('--input_width',
                        type=int,
                        default=224,
                        help='input width')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size per iteration')
    parser.add_argument('--num_batches', type=int, default=100, help='iteration')
    parser.add_argument('--num_inter_threads', type=int, default=2)
    parser.add_argument('--num_intra_threads', type=int, default=28)

    args = parser.parse_args()


    def eval_inference(graph):
        '''
        This function is a example that user provided to execute the inference.
        '''
        input_layer = args.inputs
        output_layer = args.outputs
        num_inter_threads = args.num_inter_threads
        num_intra_threads = args.num_intra_threads
        num_batches = args.batch_size

        input_tensor = graph.get_tensor_by_name(input_layer + ":0")
        output_tensor = graph.get_tensor_by_name(output_layer + ":0")

        config = tf.ConfigProto()
        config.inter_op_parallelism_threads = num_inter_threads
        config.intra_op_parallelism_threads = num_intra_threads

        num_processed_images = 0
        batch_size = args.batch_size

        if num_batches > 0:
            num_remaining_images = batch_size * num_batches

        total_accuracy1, total_accuracy5 = (0.0, 0.0)
        dataset = datasets.ImagenetData(args.data_location)
        preprocessor = preprocessing.ImagePreprocessor(
            args.input_height,
            args.input_width,
            args.batch_size,
            1,  # device count
            tf.float32,  # data_type for input fed to the graph
            train=False,  # doing inference
            resize_method='crop')
        images, labels = preprocessor.minibatch(dataset, subset='validation')

        with tf.Session() as sess:
            sess_graph = tf.Session(graph=graph, config=config)
            while num_remaining_images >= batch_size:
                # Reads and preprocess data
                np_images, np_labels = sess.run([images[0], labels[0]])
                num_processed_images += batch_size
                num_remaining_images -= batch_size
                # Compute inference on the preprocessed data
                predictions = sess_graph.run(output_tensor,
                                            {input_tensor: np_images})
                accuracy1 = tf.reduce_sum(
                    tf.cast(
                        tf.nn.in_top_k(tf.constant(predictions),
                                    tf.constant(np_labels), 1), tf.float32))

                accuracy5 = tf.reduce_sum(
                    tf.cast(
                        tf.nn.in_top_k(tf.constant(predictions),
                                    tf.constant(np_labels), 5), tf.float32))

                np_accuracy1, np_accuracy5 = sess.run([accuracy1, accuracy5])
                total_accuracy1 += np_accuracy1
                total_accuracy5 += np_accuracy5
                print(
                    "Processed %d images. (Top1 accuracy, Top5 accuracy) = (%0.4f, %0.4f)"
                    %
                    (num_processed_images, total_accuracy1 / num_processed_images,
                    total_accuracy5 / num_processed_images))

        return total_accuracy1 / num_processed_images


    fp32_graph = load_graph(args.input_graph)
    at = iLit.Tuner(args.config)

    dataloader = Dataloader(args.data_location, 'validation',
                            args.input_height, args.input_width,
                            args.batch_size)

    rn50_input_output = {
        "inputs": args.inputs.split(' '),
        "outputs": args.outputs.split(' ')}

    at.tune(
        fp32_graph,
        q_dataloader=dataloader,
        # eval_func=eval_inference, model_specific_cfg=rn50_input_output)
        eval_func=None,
        eval_dataloader=dataloader,
        model_specific_cfg=rn50_input_output)
