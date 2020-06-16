#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
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
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

import argparse
import sys
import os
import time
import numpy as np

from google.protobuf import text_format
import tensorflow as tf
import accuracy_preprocessing as preprocessing
import accuracy_datasets as datasets

NUM_TEST_IMAGES = 50000


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()

    import os
    file_ext = os.path.splitext(model_file)[1]

    with open(model_file, "rb") as f:
        if file_ext == '.pbtxt':
            text_format.Merge(f.read(), graph_def)
        else:
            graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')

    return graph

def prepare_dataloader(data_location, input_height, input_width, batch_size):
    dataset = datasets.ImagenetData(data_location)
    preprocessor = dataset.get_image_preprocessor()(
        input_height, input_width, batch_size,
        1,  # device count
        tf.float32,  # data_type for input fed to the graph
        train=False,  # doing inference
        resize_method='bilinear')
    with tf.compat.v1.get_default_graph().as_default():
        images, labels = preprocessor.minibatch(dataset, subset='validation',
                         use_datasets=True, cache_data=False)
    return images

def inference(graph):
    input_layer = "input"
    output_layer = "MobilenetV1/Predictions/Reshape_1"
    num_inter_threads = 1
    num_intra_threads = 28
    input_height = 224
    input_width = 224
    data_location = "/lustre/dataset/tensorflow/imagenet"
    dataset = datasets.ImagenetData(data_location)
    preprocessor = dataset.get_image_preprocessor()(
        input_height, input_width, batch_size,
        1,  # device count
        tf.float32,  # data_type for input fed to the graph
        train=False,  # doing inference
        resize_method='bilinear')

    with tf.compat.v1.get_default_graph().as_default():
        images, labels = preprocessor.minibatch(dataset, subset='validation',
                         use_datasets=True, cache_data=False)
    input_tensor = graph.get_tensor_by_name(input_layer + ":0")
    output_tensor = graph.get_tensor_by_name(output_layer + ":0")

    config = tf.compat.v1.ConfigProto()
    config.inter_op_parallelism_threads = num_inter_threads
    config.intra_op_parallelism_threads = num_intra_threads

    total_accuracy1, total_accuracy5 = (0.0, 0.0)
    num_processed_images = 0
    num_remaining_images = dataset.num_examples_per_epoch(subset='validation') \
                           - num_processed_images
    if num_batches > 0:
        num_remaining_images = batch_size * num_batches
    with tf.compat.v1.Session() as sess:
        sess_graph = tf.compat.v1.Session(graph=graph, config=config)
        while num_remaining_images >= batch_size:
            # Reads and preprocess data
            np_images, np_labels = sess.run([images[0], labels[0]])
            num_processed_images += batch_size
            num_remaining_images -= batch_size
            start_time = time.time()
            # Compute inference on the preprocessed data
            predictions = sess_graph.run(output_tensor,
                                         {input_tensor: np_images})
            elapsed_time = time.time() - start_time
            accuracy1 = tf.reduce_sum(
                input_tensor=tf.cast(tf.nn.in_top_k(predictions=tf.constant(predictions),
                                       targets=tf.constant(np_labels), k=1), tf.float32))

            accuracy5 = tf.reduce_sum(
                input_tensor=tf.cast(tf.nn.in_top_k(predictions=tf.constant(predictions),
                                       targets=tf.constant(np_labels), k=5), tf.float32))
            np_accuracy1, np_accuracy5 = sess.run([accuracy1, accuracy5])
            total_accuracy1 += np_accuracy1
            total_accuracy5 += np_accuracy5
            print("Iteration time: %0.4f ms" % elapsed_time)
            print(
                "Processed %d images. (Top1 accuracy, Top5 accuracy) = (%0.4f, %0.4f)" \
                % (
                num_processed_images, total_accuracy1 / num_processed_images,
                total_accuracy5 / num_processed_images))
    return total_accuracy1 / num_processed_images

if __name__ == "__main__":
    fp32_graph = load_graph('/lustre/models/mobilenet_fp32_.pb')
    at = ilit.Tuner("tf.yaml")
    mob_input_output = {"inputs": ['input'], "outputs": ["MobilenetV1/Predictions/Reshape_1"]}
    dataloader = prepare_dataloader(data_location="/lustre/dataset/tensorflow/imagenet", input_height=224, input_width=224, batch_size=32)
    at.tune(fp32_graph, q_dataloader=dataloader,
            eval_func=inference, dicts=mob_input_output)
    

