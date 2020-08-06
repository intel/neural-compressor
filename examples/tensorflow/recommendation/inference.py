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

#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import numpy as np
import argparse
import collections
import time
import math
import json
import datetime

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.core.framework import graph_pb2
from google.protobuf import text_format
from argparse import ArgumentParser
from ilit.adaptor.tf_utils.util import write_graph

tf.compat.v1.disable_eager_execution()

def load_graph(model_file):
    """This is a function to load TF graph from pb file

    Args:
        model_file (string): TF pb file local path

    Returns:
        graph: TF graph object
  """
    graph = tf.Graph()
    #graph_def = tf.compat.v1.GraphDef()
    graph_def = graph_pb2.GraphDef()

    file_ext = os.path.splitext(model_file)[1]

    with open(model_file, "rb") as f:
        if file_ext == '.pbtxt':
            text_format.Merge(f.read(), graph_def)
        else:
            graph_def.ParseFromString(f.read())

    with graph.as_default():
        tf.import_graph_def(graph_def, name='')

    return graph


numeric_feature_names = ["numeric_1"]
string_feature_names = ["string_1"]

def get_feature_name(compute_accuracy):

    if compute_accuracy:
        full_features_names = numeric_feature_names + string_feature_names + ["label"]
        feature_datatypes = [tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True)]+[tf.io.FixedLenSequenceFeature(
                    [], tf.int64, default_value=0, allow_missing=True)]+[tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True)]
    else:
        full_features_names = numeric_feature_names + string_feature_names
        feature_datatypes = [tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True)]+[tf.io.FixedLenSequenceFeature(
                    [], tf.int64, default_value=0, allow_missing=True)]
    return full_features_names, feature_datatypes

def input_fn(data_file, num_epochs, shuffle, batch_size, compute_accuracy=True):
    """Generate an input function for the Estimator."""
    full_features_names, feature_datatypes = get_feature_name(compute_accuracy)
    def _parse_function(proto):
        f = collections.OrderedDict(
                zip(full_features_names, feature_datatypes))
        parsed_features = tf.io.parse_example(proto, f)
        parsed_feature_vals_num = [tf.reshape(
                                    parsed_features["numeric_1"], shape=[-1, 13])]
        parsed_feature_vals_str = [tf.reshape(
                                    parsed_features["string_1"], shape=[-1, 2]) for i in string_feature_names]
        parsed_feature_vals = parsed_feature_vals_num + parsed_feature_vals_str
        if compute_accuracy:
            parsed_feature_vals_label = [tf.reshape(parsed_features[i], shape=[-1]) for i in ["label"]]
            parsed_feature_vals = parsed_feature_vals + parsed_feature_vals_label
        return parsed_feature_vals

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TFRecordDataset([data_file])
    if shuffle:
        dataset = dataset.shuffle(buffer_size=20000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(_parse_function, num_parallel_calls=28)
    dataset = dataset.prefetch(batch_size*10)
    return dataset

class eval_classifier_optimized_graph:
    """Evaluate image classifier with optimized TensorFlow graph"""

    def __init__(self):
        arg_parser = ArgumentParser(description='Parse args')
        arg_parser.add_argument('-i', '--input_graph', type=str,
                                help='Specify the input of the model',
                                dest='input_graph',
                                required=True)
        arg_parser.add_argument('--calibration_data_location', type=str,
                                help='full path of calibration data file',
                                dest='calib_data',
                                required=True)
        arg_parser.add_argument('--evaluation_data_location', type=str,
                                help='full path of validation data file',
                                dest='eval_data',
                                required=True)
        arg_parser.add_argument('--evaluation_data_location', type=str,
                                help='full path of validation data file',
                                dest='eval_data',
        arg_parser.add_argument('--batch_size', type=int,
                                help='batch size for inference.Default is 512',
                                default=512,
                                dest='batch_size')
        arg_parser.add_argument('--num_intra_threads', type=int,
                                help='number of threads for an operator',
                                required=False,
                                default=28,
                                dest='num_intra_threads')
        arg_parser.add_argument('--num_inter_threads', type=int,
                                help='number of threads across operators',
                                required=False,
                                default=2,
                                dest='num_inter_threads')
        arg_parser.add_argument('--num_omp_threads', type=str,
                                help='number of threads to use',
                                required=False,
                                default=None,
                                dest='num_omp_threads')
        arg_parser.add_argument('--kmp_blocktime', type=str,
                                help='KMP_BLOCKTIME value',
                                required=False,
                                default=None,
                                dest='kmp_blocktime')
        arg_parser.add_argument('-r', "--accuracy_only",
                                help='For accuracy measurement only.',
                                dest='accuracy_only', action='store_true')
        arg_parser.add_argument("--config", default=None,
                                help="tuning config")

        self.args = arg_parser.parse_args()

    def auto_tune(self):
        """This is iLiT tuning part to generate a quantized pb
        Returns:
            graph: it will return a quantized pb
        """
        import ilit

        fp32_graph = load_graph(self.args.input_graph)
        tuner = ilit.Tuner(self.args.config)
        calib_dataloader = Dataloader(self.args.calib_data, self.args.batch_size)
        q_model = tuner.tune(
                            fp32_graph,
                            q_dataloader=calib_dataloader,
                            eval_func=self.eval_inference,
                            eval_dataloader=None)
        return q_model

    def eval_inference(self, infer_graph):
        """run benchmark with optimized graph"""

        print("Run inference")

        data_config = tf.compat.v1.ConfigProto()
        data_config.intra_op_parallelism_threads = self.args.num_intra_threads
        data_config.inter_op_parallelism_threads = self.args.num_inter_threads
        data_config.use_per_session_threads = 1

        no_of_test_samples = sum(1 for _ in tf.compat.v1.python_io.tf_record_iterator(self.args.eval_data))
        no_of_batches = math.ceil(float(no_of_test_samples)/self.args.batch_size)
        placeholder_list = ['new_numeric_placeholder','new_categorical_placeholder']
        input_tensor = [infer_graph.get_tensor_by_name(name + ":0") for name in placeholder_list]
        output_name = "import/head/predictions/probabilities"
        output_tensor = infer_graph.get_tensor_by_name(output_name + ":0" )
        correctly_predicted = 0
        total_infer_consume = 0.0

        eval_dataloader = Dataloader(self.args.eval_data, self.args.batch_size)

        with tf.compat.v1.Session(config=data_config, graph=infer_graph) as sess:
            i=0
            for content in eval_dataloader:
                if i >= no_of_batches:
                    break

                inference_start = time.time()
                logistic = sess.run(output_tensor, dict(zip(input_tensor, content[0][0:2])))

                infer_time = time.time() - inference_start
                total_infer_consume += infer_time
                if self.args.accuracy_only:
                    predicted_labels = np.argmax(logistic,1)
                    correctly_predicted=correctly_predicted+np.sum(content[1] == predicted_labels)

                i=i+1

        if self.args.accuracy_only:
            accuracy = float(correctly_predicted)/float(no_of_test_samples)

        evaluate_duration = total_infer_consume
        latency = (1000 * self.args.batch_size * float(evaluate_duration) / float(no_of_test_samples))
        throughput = no_of_test_samples / evaluate_duration

        print('--------------------------------------------------')
        print('Total test records           : ', no_of_test_samples)
        print('Batch size is                : ', self.args.batch_size)
        print('Number of batches            : ', int(no_of_batches))
        if self.args.accuracy_only:
            print('Classification accuracy (%)  : ', round((accuracy * 100), 4))
            print('No of correct predictions    : ', int(correctly_predicted))
        print('Inference duration (seconds) : ', round(evaluate_duration, 4))
        print('Average Latency (ms/batch)   : ', round(latency,4))
        print('Throughput is (records/sec)  : ', round(throughput, 3))
        print('--------------------------------------------------')

        if self.args.accuracy_only:
            return accuracy


class Dataloader(object):
    def __init__(self, data_location, batch_size):
        """dataloader generator

        Args:
            data_location (str): tf recorder local path
            batch_size (int): dataloader batch size
        """
        self.batch_size = batch_size
        self.data_file = data_location
        self.total_samples = sum(1 for _ in tf.compat.v1.python_io.tf_record_iterator(data_location))
        self.n = math.ceil(float(self.total_samples) / batch_size)
        print("batch size is " + str(self.batch_size) + "," + str(self.n) + " iteration")

    def __iter__(self):
        data_graph = tf.Graph()
        with data_graph.as_default():
            self.dataset = input_fn(self.data_file, 1, False, self.batch_size)
            self.dataset_iterator = tf.compat.v1.data.make_one_shot_iterator(self.dataset)
            next_element = self.dataset_iterator.get_next()

        with tf.compat.v1.Session(graph=data_graph) as sess:
            for i in range(self.n):
                batch = sess.run(next_element)
                yield (batch[0:2], batch[2])


if __name__ == "__main__":
    evaluate_opt_graph = eval_classifier_optimized_graph()
    q_model = evaluate_opt_graph.auto_tune()
    evaluate_opt_graph.eval_inference(q_model)
    write_graph(q_model.as_graph_def(), "./output_int8.pb")

