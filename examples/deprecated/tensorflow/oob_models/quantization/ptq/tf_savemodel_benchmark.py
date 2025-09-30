#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.compat.v1 import graph_util

def get_dynamic_inputshape(model_dir,dshape):
    # judge object_detection model
    path = model_dir.split('/')
    is_detection = False
    for item in path:
        if 'detection' in item or 'mask' in item or 'rcnn' in item:
            is_detection = True
            break
    fix_dynamic_shape = 600 if is_detection else 300
    for dim,val in enumerate(dshape[1:]):
        if val==-1:
            dshape[dim+1]=fix_dynamic_shape
    return dshape

def generate_data(batch_size, input_shape, input_dtype):
    np.random.seed(1024)
    if input_dtype=='float32':
       dummy_input = np.random.randn(*input_shape[1:]).astype(input_dtype)
    elif input_dtype=='uint8':
       dummy_input = np.random.randint(-127,128,input_shape[1:]).astype(input_dtype)
    return np.repeat(dummy_input[np.newaxis, :], batch_size, axis=0)

def metrics_generator(array, tolerance):
    max_diff = np.max(array)
    mean_diff = np.mean(array)
    median_diff = np.median(array)
    success_rate = np.sum(array < tolerance) / array.size
    return max_diff, mean_diff, median_diff, success_rate

def create_tf_config(precision):
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    # config.intra_op_parallelism_threads = 1
    # config.inter_op_parallelism_threads = 1
    if precision == 'bfloat16':
        config.graph_options.rewrite_options.auto_mixed_precision_mkl = rewriter_config_pb2.RewriterConfig.ON
        print("auto_mixed_precision_mkl ON.")
    return config

def initialize_graph(args):
    tf_config = create_tf_config(args.precision)
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        with tf.compat.v1.Session(config=tf_config) as sess:
            meta_graph=tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], args.model_path)
            assert savemodel_valid(meta_graph),"savemodel is invalid"
            model_graph_signature = list(meta_graph.signature_def.items())[0][1]
            input_tensor_names = []
            dummy_inputs=[]
            for input_item in model_graph_signature.inputs.items():
                input_tensor_name = input_item[1].name
                input_tensor_names.append(input_tensor_name)
                if input_item[1].dtype==1:
                    dtype='float32'
                else:
                    dtype='uint8'
                dshape=[int(item.size) for item in input_item[1].tensor_shape.dim]
                if -1 in dshape[1:]:
                    dshape=get_dynamic_inputshape(args.model_path, dshape)
                dummy_inputs.append(generate_data(args.batch_size, dshape, dtype))
            output_tensor_names = []
            for output_item in model_graph_signature.outputs.items():
                output_tensor_name = output_item[1].name
                output_tensor_names.append(output_tensor_name)
            freeze_graph_def = graph_util.convert_variables_to_constants(  
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[output_name.split(":")[0] for output_name in output_tensor_names])
            if args.disable_optimize:
                freeze_graph_def = optimize_for_inference_lib.optimize_for_inference(
                                        freeze_graph_def, #inputGraph,
                                        [input_name.split(":")[0] for input_name in input_tensor_names], # an array of the input node(s)
                                        [output_name.split(":")[0] for output_name in output_tensor_names], # an array of output nodes
                                        tf.float32.as_datatype_enum)
            input_variables = {in_name : tf.Variable(dummy_inputs[i])
                            for i,in_name in enumerate(input_tensor_names)}
            tf.import_graph_def(freeze_graph_def,name='g',input_map=input_variables)
    return graph,output_tensor_names

def savemodel_valid(meta_graph):
    valid_op=["Conv2D","DepthwiseConv2dNative","MaxPool","AvgPool","FusedBatchNorm","FusedBatchNormV3","BatchNormWithGlobalNormalization",
                 "Relu","Relu6","Softmax","BiasAdd","Add","AddV2"]
    all_op_types = []
    for i in meta_graph.graph_def.node:
        all_op_types.append(i.op)
    print (set(all_op_types))
    flag=False
    for op in set(all_op_types):
        if op in valid_op:
            flag=True
    return flag

def run_benchmark(args):
    tf_config = create_tf_config(args.precision)
    graph,output_tensor_names=initialize_graph(args)
    run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()
    with tf.compat.v1.Session(config=tf_config,graph=graph) as sess:
        output_dict = {out_name: graph.get_tensor_by_name("g/" + out_name )
                for out_name in output_tensor_names}
        total_time = 0.0
        reps_done = 0
        sess.run(tf.compat.v1.global_variables_initializer())
        for rep in range(args.num_iter):
            if rep < args.num_warmup:
                sess.run(output_dict)
                continue
            start = time.time()
            if args.profile:
                sess.run(output_dict, options=run_options, run_metadata=run_metadata)
            else:
                sess.run(output_dict)
            end = time.time()
            delta = end - start
            total_time += delta
            reps_done += 1
            if rep % 10 == 0:
                print("Iteration: {}, inference time: {:.6f} sec.".format(rep, delta))

        # save profiling file
        if args.profile:
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            # model_dir = os.path.dirname(os.path.abspath(model_detail['model_dir']))
            model_dir = str(os.path.dirname(os.path.realpath(__file__))) + '/timeline'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            profiling_file = model_dir + '/timeline-' + str(rep) + '-' + str(os.getpid()) + '.json'
            with open(profiling_file, 'w') as trace_file:
                trace_file.write(
                    trace.generate_chrome_trace_format(show_memory=False))

        avg_time = total_time / reps_done
        latency = avg_time * 1000
        throughput = 1.0 / avg_time * args.batch_size
        print("Latency: {:.0f} ms".format(latency))
        print("Throughput: {:.2f} fps".format(throughput))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", help="path of savemodel", required=True)
    parser.add_argument("-t", "--precision", type=str, default='float32', 
                    help="float32, int8 or bfloat16")
    parser.add_argument("-n", "--num_iter", type=int, default=500,
                        help="numbers of inference iteration, default is 500")
    parser.add_argument("-w","--num_warmup", type=int, default=10,
                        help="numbers of warmup iteration, default is 10")
    parser.add_argument("--disable_optimize", action='store_false',
                        help="use this to disable optimize_for_inference")
    parser.add_argument("-b", "--batch_size", type=int, default=1, 
                    help="batch size")
    parser.add_argument("--profile", action='store_true', help="profile")
    args = parser.parse_args()

    run_benchmark(args)

