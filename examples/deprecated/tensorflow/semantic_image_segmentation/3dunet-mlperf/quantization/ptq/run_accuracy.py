#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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

import time
from argparse import ArgumentParser
import os
import pickle
import sys
import math
import array

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from nnunet.evaluation.region_based_evaluation import evaluate_regions, get_brats_regions

from nnUNet.setup import setup
from nnUNet.postprocess import postprocess_output

INPUTS = 'input'
OUTPUTS = 'Identity'

if __name__ == "__main__":
    """Evaluate 3d_unet with optimized TensorFlow graph"""
    def get_args():
        arg_parser = ArgumentParser(description='Parse args')

        arg_parser.add_argument('-m', "--mode",
                                help="One of three options: 'benchmark'/'accuracy'/'tune'.")
        arg_parser.add_argument('-n', "--iters",
                                help='The number of iteration. shall > warmup num(10)',
                                type=int, default=20)
        arg_parser.add_argument('-e', "--num-inter-threads",
                                help='The number of inter-thread.',
                                dest='num_inter_threads', type=int, default=0)
        arg_parser.add_argument('-a', "--num-intra-threads",
                                help='The number of intra-thread.',
                                dest='num_intra_threads', type=int, default=0)
        arg_parser.add_argument('-i', "--input-model",
                                help='Specify the input graph.',
                                dest='input_model')
        arg_parser.add_argument('-o', "--output-model",
                                help='Specify the output graph.',
                                dest='output_model')
        arg_parser.add_argument('-c', "--calib-preprocess",
                                help='Specify calibration preprocess dir.',
                                dest='calib_preprocess')
        arg_parser.add_argument('-d', "--data-location",
                                help='Specify the location of the data.',
                                dest="data_location", default=None)
        arg_parser.add_argument("--batch-size", dest="batch_size", type=int, default=1)
        arg_parser.add_argument("--bfloat16", type=int, default=0)

        args = arg_parser.parse_args()
        print(args)
        return args

    def eval_func(graph):
        print("Run inference for accuracy")
        args = get_args()
        #setup(args.data_location, args.input_model)

        output_graph = optimize_for_inference(graph.as_graph_def(), [INPUTS], [OUTPUTS],
                            dtypes.float32.as_datatype_enum, False)
        tf.import_graph_def(output_graph, name="")

        input_tensor = graph.get_tensor_by_name('input:0')
        output_tensor = graph.get_tensor_by_name('Identity:0')

        config = tf.compat.v1.ConfigProto()
        config.intra_op_parallelism_threads=args.num_intra_threads
        config.inter_op_parallelism_threads=args.num_inter_threads
        if args.bfloat16:
            config.graph_options.rewrite_options.auto_mixed_precision_mkl = rewriter_config_pb2.RewriterConfig.ON

        sess = tf.compat.v1.Session(graph=graph, config=config)
        if args.mode:
            print("Inference with real data")
            preprocessed_data_dir = os.path.join(args.data_location, "preprocessed_data")
            with open(os.path.join(preprocessed_data_dir, "preprocessed_files.pkl"), "rb") as f:
                preprocessed_files = pickle.load(f)

            dictionaries = []
            for preprocessed_file in preprocessed_files:
                with open(os.path.join(preprocessed_data_dir, preprocessed_file + ".pkl"), "rb") as f:
                    dct = pickle.load(f)[1]
                    dictionaries.append(dct)

            count = len(preprocessed_files)
            predictions = [None] * count
            validation_indices = list(range(0,count))
            print("Found {:d} preprocessed files".format(count))
            loaded_files = {}
            batch_size = args.batch_size
  
            # Get the number of steps based on batch size
            steps = count#math.ceil(count/batch_size)
            warmup = 10
            assert args.iters >= warmup, 'iteration must be larger than warmup'
            time_list=[]
            for i in range(steps):
                print("Iteration {} ...".format(i))
                test_data_index = validation_indices[i]#validation_indices[i * batch_size:(i + 1) * batch_size]
                file_name = preprocessed_files[test_data_index]
                with open(os.path.join(preprocessed_data_dir, "{:}.pkl".format(file_name)), "rb") as f:
                    data = pickle.load(f)[0]
                if args.mode == 'performance' and i < args.iters:
                    time_start = time.time()
                    predictions[i] = sess.run(output_tensor, feed_dict={input_tensor: data[np.newaxis, ...]})[0].astype(np.float32)
                    duration = time.time() - time_start
                    time_list.append(duration)
                else:
                    predictions[i] = sess.run(output_tensor, feed_dict={input_tensor: data[np.newaxis, ...]})[0].astype(np.float32)
            if args.mode == 'performance':
                latency = np.array(time_list[warmup: ]).mean() / args.batch_size
                print('Batch size = {}'.format(args.batch_size))
                print('Latency: {:.3f} ms'.format(latency * 1000))
                print('Throughput: {:.3f} items/sec'.format(1./ latency))
            else:
                output_folder = os.path.join(args.data_location, "postprocessed_data")
                output_files = preprocessed_files
                # Post Process
                postprocess_output(predictions, dictionaries, validation_indices, output_folder, output_files)

                ground_truths = os.path.join(args.data_location, \
                     "raw_data/nnUNet_raw_data/Task043_BraTS2019/labelsTr")
                # Run evaluation
                print("Running evaluation...")
                evaluate_regions(output_folder, ground_truths, get_brats_regions())
                # Load evaluation summary
                print("Loading evaluation summary...")
                accuracy=0.0
                with open(os.path.join(output_folder, "summary.csv")) as f:
                    for line in f:
                        words = line.split(",")
                        if words[0] == "mean":
                            whole = float(words[1])
                            core = float(words[2])
                            enhancing = float(words[3])
                            mean = (whole + core + enhancing) / 3
                            accuracy=mean
                            print("Batch size =", args.batch_size)
                            print("Accuracy is {:.5f}".format(mean))
                            break
                print("Done!")
                return accuracy

    def load_graph(file_name):
        tf.compat.v1.logging.info('Loading graph from: ' + file_name)
        with tf.io.gfile.GFile(file_name, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
        return graph

    class CalibrationDL():
        def __init__(self):
            path = os.path.abspath(os.path.expanduser(
                './brats_cal_images_list.txt'))
            with open(path, 'r') as f:
                self.preprocess_files = [line.rstrip() for line in f]

            self.loaded_files = {}
            self.batch_size = 1

        def __getitem__(self, sample_id):
            file_name = self.preprocess_files[sample_id]
            print("Loading file {:}".format(file_name))
            with open(os.path.join(args.calib_preprocess, "{:}.pkl".format(file_name)), "rb") as f:
                self.loaded_files[sample_id] = pickle.load(f)[0]
            # note that calibration phase does not care label, here we return 0 for label free case.
            return self.loaded_files[sample_id], 0

        def __len__(self):
            self.count = len(self.preprocess_files)
            return self.count


    args = get_args()
    print(args)
    graph = load_graph(args.input_model)
    if args.mode == 'tune':
        from neural_compressor.data import DataLoader
        from neural_compressor.quantization import fit
        from neural_compressor.config import PostTrainingQuantConfig
        from neural_compressor import set_random_seed
        set_random_seed(9527)
        config = PostTrainingQuantConfig(calibration_sampling_size=[40])

        q_model = fit(
            model=graph,
            conf=config,
            calib_dataloader=DataLoader(framework='tensorflow', dataset=CalibrationDL()),
            eval_func=eval_func)
        try:
            q_model.save(args.output_model)
        except Exception as e:
            print("Failed to save model due to {}".format(str(e)))
    else:
        from neural_compressor.data import DataLoader
        from neural_compressor.benchmark import fit
        from neural_compressor.config import BenchmarkConfig
        conf = BenchmarkConfig(cores_per_instance=4, num_of_instance=1)
        fit(graph, conf,
            b_dataloader=DataLoader(framework='tensorflow', dataset=CalibrationDL()),
            b_func=eval_func)
