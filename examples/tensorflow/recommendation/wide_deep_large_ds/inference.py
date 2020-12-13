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
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

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
        arg_parser.add_argument('-o', '--output_graph', type=str,
                                help='Specify the output of the model',
                                dest='output_graph')
        arg_parser.add_argument('--calibration_data_location', type=str,
                                help='full path of calibration data file',
                                dest='calib_data')
        arg_parser.add_argument('--evaluation_data_location', type=str,
                                help='full path of validation data file',
                                dest='eval_data',
                                required=True)
        arg_parser.add_argument('--batch_size', type=int,
                                help='batch size for inference.Default is 512',
                                default=512,
                                dest='batch_size')
        arg_parser.add_argument('--num_intra_threads', type=int,
                                help='number of threads for an operator',
                                required=False,
                                default=0,
                                dest='num_intra_threads')
        arg_parser.add_argument('--num_inter_threads', type=int,
                                help='number of threads across operators',
                                required=False,
                                default=0,
                                dest='num_inter_threads')
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
        arg_parser.add_argument('--benchmark',
                                dest='benchmark',
                                action='store_true',
                                help='run benchmark')
        arg_parser.add_argument('--tune',
                                dest='tune',
                                action='store_true',
                                help='use lpot to tune.')
        arg_parser.add_argument("--warmup-steps",
                                type=int, default=50,
                                help="number of warmup steps")
        arg_parser.add_argument("--steps",
                                type=int, default=2000,
                                help="number of iterations")

        arg_parser.add_argument('--env',
                                dest='env',
                                help='specific Tensorflow env',
                                default='mkl')


        self.args = arg_parser.parse_args()

    def auto_tune(self):
        """This is lpot tuning part to generate a quantized pb
        Returns:
            graph: it will return a quantized pb
        """
        from lpot import Quantization

        fp32_graph = load_graph(self.args.input_graph)
        quantizer = Quantization(self.args.config)
        if self.args.calib_data:
            calib_dataloader = Dataloader(self.args.calib_data, self.args.batch_size)
            q_model = quantizer(
                                fp32_graph,
                                q_dataloader=calib_dataloader,
                                eval_func=self.eval_inference,
                                eval_dataloader=None)
            return q_model
        else:
            print("Please provide calibration dataset!")

    def eval_inference(self, infer_graph):
        print("Run inference")

        data_config = tf.compat.v1.ConfigProto()
        data_config.intra_op_parallelism_threads = self.args.num_intra_threads
        data_config.inter_op_parallelism_threads = self.args.num_inter_threads
        data_config.use_per_session_threads = 1

        infer_config = tf.compat.v1.ConfigProto()
        if self.args.env == 'mkl':
            print("Set inter and intra for mkl: ")
            print("intra_op_parallelism_threads = ", self.args.num_intra_threads)
            print("inter_op_parallelism_threads = ", self.args.num_inter_threads)
            infer_config.intra_op_parallelism_threads = self.args.num_intra_threads
            infer_config.inter_op_parallelism_threads = self.args.num_inter_threads
        infer_config.use_per_session_threads = 1

        total_test_samples = sum(1 for _ in tf.compat.v1.python_io.tf_record_iterator(self.args.eval_data))
        total_batches = math.ceil(float(total_test_samples)/self.args.batch_size)
        placeholder_list = ['new_numeric_placeholder','new_categorical_placeholder']
        input_tensor = [infer_graph.get_tensor_by_name(name + ":0") for name in placeholder_list]
        output_name = "import/head/predictions/probabilities"
        output_tensor = infer_graph.get_tensor_by_name(output_name + ":0" )
        correctly_predicted = 0
        evaluate_duration = 0.0

        features_list = []
        data_graph = tf.Graph()
        with data_graph.as_default():
            res_dataset = input_fn(self.args.eval_data, 1, False, self.args.batch_size)
            iterator = tf.compat.v1.data.make_one_shot_iterator(res_dataset)
            next_element = iterator.get_next()
            with tf.compat.v1.Session(config=data_config, graph=data_graph) as data_sess:
                for i in range(int(total_batches)):
                    batch = data_sess.run(next_element)
                    features=batch[0:3]
                    features_list.append(features)

        if (not self.args.accuracy_only):
            iteration = 0
            warm_up_iteration = self.args.warmup_steps
            total_run = self.args.steps

            if total_run > total_batches:
                total_run = total_batches

            with tf.compat.v1.Session(config=infer_config, graph=infer_graph) as infer_sess:
                i = 0
                for i in range(int(total_run)):
                    start_time = time.time()
                    logistic = infer_sess.run(output_tensor, dict(zip(input_tensor, features_list[iteration][0:2])))
                    time_consume = time.time() - start_time

                    if iteration > warm_up_iteration:
                        evaluate_duration += time_consume

                    iteration += 1
                    if iteration > total_batches:
                        iteration = 0
                test_batches = total_run - warm_up_iteration
        else:
            with tf.compat.v1.Session(config=infer_config, graph=infer_graph) as infer_sess:
                i = 0
                for i in range(int(total_batches)):
                    start_time = time.time()
                    logistic = infer_sess.run(output_tensor, dict(zip(input_tensor, features_list[i][0:2])))
                    time_consume = time.time() - start_time
                    evaluate_duration += time_consume

                    predicted_labels = np.argmax(logistic,1)
                    correctly_predicted=correctly_predicted+np.sum(features_list[i][2] == predicted_labels)

                    i=i+1

                accuracy = float(correctly_predicted) / float(total_test_samples)
                test_batches = total_batches

        no_of_test_samples = test_batches * self.args.batch_size
        latency = 1000 * float(evaluate_duration) / float(test_batches)
        throughput = no_of_test_samples / evaluate_duration

        print('--------------------------------------------------')
        print('Total test records: %d' % no_of_test_samples)
        print('Number of batches: %d' % test_batches)
        print('Batch size = %d' % self.args.batch_size)
        print('Latency: %.3f ms' % latency)
        print('Throughput: %.3f records/sec' % throughput)
        if self.args.accuracy_only:
            print("Accuracy: %.5f" % accuracy)
        print('--------------------------------------------------')

        if self.args.accuracy_only:
            return accuracy

    def run(self):
        """ This is lpot function include tuning and benchmark option """

        if self.args.tune:
            q_model = evaluate_opt_graph.auto_tune()
            def save(model, path):
                from tensorflow.python.platform import gfile
                f = gfile.GFile(path, 'wb')
                f.write(model.as_graph_def().SerializeToString())

            save(q_model, self.args.output_graph)

        if self.args.benchmark:
            infer_graph = load_graph(self.args.input_graph)

            self.eval_inference(infer_graph)



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
    evaluate_opt_graph.run()

