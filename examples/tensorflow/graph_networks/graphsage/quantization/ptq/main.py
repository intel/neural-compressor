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

import os
import time
import utils
import dataloader
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import tf_logging
from tensorflow.core.protobuf import rewriter_config_pb2

from argparse import ArgumentParser

np.random.seed(123)

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
arg_parser.add_argument('-e', "--num-inter-threads",
                        help='The number of inter-thread.',
                        dest='num_inter_threads', type=int, default=0)

arg_parser.add_argument('-a', "--num-intra-threads",
                        help='The number of intra-thread.',
                        dest='num_intra_threads', type=int, default=0)
arg_parser.add_argument('--batch_size', type=int, default=1000, dest='batch_size', help='batch_size of benchmark')
arg_parser.add_argument('--iters', type=int, default=100, dest='iters', help='interations')
args = arg_parser.parse_args()

def prepare_Dataset():
    data_location = args.dataset_location
    pretrained_model = args.input_graph
    data = dataloader.load_data(prefix=data_location+'/ppi')
    G = data[0]
    features = data[1]
    id_map = data[2]
    class_map  = data[4]
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))

    context_pairs = data[3]
    placeholders = utils.construct_placeholders(num_classes)
    minibatch = utils.NodeMinibatchIterator(G, 
            id_map,
            placeholders, 
            class_map,
            num_classes,
            batch_size=args.batch_size,
            context_pairs = context_pairs)
    return minibatch

class CustomDataset(object):
    def __init__(self):
        self.batch1 = []
        self.batch_labels = []
        minibatch = prepare_Dataset() 
        self.parse_minibatch(minibatch)

    def parse_minibatch(self, minibatch):
        iter_num = 0
        finished = False
        while not finished:
            feed_dict_val, batch_labels, finished, _ = minibatch.incremental_node_val_feed_dict(args.batch_size, iter_num, test=True)
            self.batch1.append(feed_dict_val['batch1:0'])
            self.batch_labels.append(batch_labels)
            iter_num += 1

    def __getitem__(self, index):
        return (self.batch1[index], len(self.batch1[index])), self.batch_labels[index]

    def __len__(self):
        return len(self.batch1)

def evaluate(model):
    """Custom evaluate function to estimate the accuracy of the model.

    Args:
        model (tf.Graph_def): The input model graph
        
    Returns:
        accuracy (float): evaluation result, the larger is better.
    """
    from neural_compressor.model import Model
    model = Model(model)
    output_tensor = model.output_tensor if len(model.output_tensor)>1 else \
                        model.output_tensor[0]
    iteration = -1              
    minibatch = prepare_Dataset()      
    if args.benchmark and args.mode == 'performance':
        iteration = args.iters

    #output_tensor = model.sess.graph.get_tensor_by_name('Sigmoid:0')
    def eval_func(size, output_tensor, minibatch, test):
        t_test = time.time()
        val_losses = []
        val_preds = []
        labels = []
        iter_num = 0
        finished = False
        total_time = 0
        while not finished:
            feed_dict_val, batch_labels, finished, _ = minibatch.incremental_node_val_feed_dict(size, iter_num, test=True)
            tf_logging.warn('\n---> Start iteration {0}'.format(str(iter_num)))
            start_time = time.time()
            node_outs_val = model.sess.run([output_tensor],feed_dict=feed_dict_val)
            time_consume = time.time() - start_time
            val_preds.append(node_outs_val[0].astype(float))
            labels.append(batch_labels)
            iter_num += 1
            total_time += time_consume
            if iteration != -1 and iter_num >= iteration:
                break
        tf_logging.warn('\n---> Stop iteration {0}'.format(str(iter_num)))
        val_preds = np.vstack(val_preds)
        labels = np.vstack(labels)
        f1_scores = utils.calc_f1(labels, val_preds)
        time_average = total_time / iter_num
        return f1_scores, (time.time() - t_test)/iter_num, time_average

    test_f1_micro, duration, time_average = eval_func(args.batch_size, output_tensor, minibatch, test=True)
    if args.benchmark and args.mode == 'performance':
        latency = time_average / args.batch_size
        print("Batch size = {}".format(args.batch_size))
        print("Latency: {:.3f} ms".format(latency * 1000))
        print("Throughput: {:.3f} images/sec".format(1. / latency))
    return test_f1_micro

def collate_function(batch):
    return (batch[0][0][0], batch[0][0][1]), batch[0][1]

class eval_graphsage_optimized_graph:
    """Evaluate image classifier with optimized TensorFlow graph."""

    def run(self):
        """This is neural_compressor function include tuning, export and benchmark option."""
        from neural_compressor import set_random_seed
        set_random_seed(9527)
        
        if args.tune:
            from neural_compressor import quantization
            from neural_compressor.data import DataLoader
            from neural_compressor.config import PostTrainingQuantConfig  
            dataset = CustomDataset()
            calib_dataloader=DataLoader(framework='tensorflow', dataset=dataset, \
                                        batch_size=1, collate_fn = collate_function)          
            conf = PostTrainingQuantConfig()
            q_model = quantization.fit(args.input_graph, conf=conf, \
                                       calib_dataloader=calib_dataloader, eval_func=evaluate)
            q_model.save(args.output_graph)

        if args.benchmark:
            if args.mode == 'performance':
                from neural_compressor.benchmark import fit
                from neural_compressor.config import BenchmarkConfig
                conf = BenchmarkConfig()
                fit(args.input_graph, conf, b_func=evaluate)
            elif args.mode == 'accuracy':
                acc_result = evaluate(args.input_graph)
                print("Batch size = %d" % args.batch_size)
                print("Accuracy: %.5f" % acc_result)

if __name__ == "__main__":
    evaluate_opt_graph = eval_graphsage_optimized_graph()
    evaluate_opt_graph.run()
