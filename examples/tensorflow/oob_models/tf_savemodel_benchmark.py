import os
import cv2
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.framework import graph_util
if "PRINT_LATENCY" in os.environ and int(os.environ["PRINT_LATENCY"]) == 1:
    PRINT_LATENCY = True
else:
    PRINT_LATENCY = False

if "CHECK_ACCURACY" in os.environ and os.environ["CHECK_ACCURACY"] == "1":
    CHECK_ACCURACY = True
else:
    CHECK_ACCURACY = False

if "BS" in os.environ:
    BATCH_SIZE = int(os.environ["BS"])
else:
    BATCH_SIZE = 1

if "NUM_INSTANCES" in os.environ:
    NUM_INSTANCES = int(os.environ["NUM_INSTANCES"])
else:
    NUM_INSTANCES = 1

if "OMP_NUM_THREADS" in os.environ:
    NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])

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
def generate_data(input_shape, input_dtype):
    np.random.seed(1024)
    if input_dtype=='float32':
       dummy_input = np.random.randn(*input_shape[1:]).astype(input_dtype)
    elif input_dtype=='uint8':
       dummy_input = np.random.randint(-127,128,input_shape[1:]).astype(input_dtype)
    return np.repeat(dummy_input[np.newaxis, :], BATCH_SIZE, axis=0)
def metrics_generator(array, tolerance):
    max_diff = np.max(array)
    mean_diff = np.mean(array)
    median_diff = np.median(array)
    success_rate = np.sum(array < tolerance) / array.size
    return max_diff, mean_diff, median_diff, success_rate
def create_tf_config():
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 1
    return config
def initialize_graph(model_dir, enable_optimize):
    tf_config = create_tf_config()
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        with tf.compat.v1.Session(config=tf_config) as sess:
            meta_graph=tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING],model_dir)
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
                    dshape=get_dynamic_inputshape(model_dir,dshape)
                dummy_inputs.append(generate_data(dshape, dtype))
            output_tensor_names = []
            for output_item in model_graph_signature.outputs.items():
                output_tensor_name = output_item[1].name
                output_tensor_names.append(output_tensor_name)
            freeze_graph_def = graph_util.convert_variables_to_constants(  
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[output_name.split(":")[0] for output_name in output_tensor_names])
            if enable_optimize:
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
def run_benchmark(model_dir, max_reps, num_warmup,enable_optimize):
    tf_config = create_tf_config()
    graph,output_tensor_names=initialize_graph(model_dir, enable_optimize)
    with tf.compat.v1.Session(config=tf_config,graph=graph) as sess:
        output_dict = {out_name: graph.get_tensor_by_name("g/" + out_name )
                for out_name in output_tensor_names}
        total_time = 0.0
        reps_done = 0
        sess.run(tf.compat.v1.global_variables_initializer())
        for rep in range(max_reps):
            if rep < num_warmup:
                sess.run(output_dict)
                continue
            start = time.time()
            sess.run(output_dict)
            end = time.time()
            delta = end - start
            total_time += delta
            reps_done += 1
        avg_time = total_time / reps_done
        latency = avg_time * 1000
        throughput = 1.0 / avg_time * BATCH_SIZE * NUM_INSTANCES
        if PRINT_LATENCY:
            print("Latency: {:.0f} ms".format(latency))
        else:
            print("Throughput: {:.2f} fps".format(throughput))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", help="path of savemodel", required=True)
    parser.add_argument("-n", "--num_iter", type=int, default=500,
                        help="numbers of inference iteration, default is 500")
    parser.add_argument("-w","--num_warmup", type=int, default=10,
                        help="numbers of warmup iteration, default is 10")
    parser.add_argument("--disable_optimize", action='store_false',
                        help="use this to disable optimize_for_inference")
    args = parser.parse_args()

    run_benchmark(args.model_path,args.num_iter,args.num_warmup,args.disable_optimize)

