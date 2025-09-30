
import tensorflow as tf
print("Tensorflow version {}".format(tf.__version__))

import numpy as np
import time
import argparse
import os
import json


import mnist_dataset
import alexnet


def val_data():
    x_train, y_train, label_train, x_test, y_test, label_test = mnist_dataset.read_data()
    return x_test, y_test, label_test


def calc_accuracy(predictions, labels):
    predictions = np.argmax(predictions, axis=1)
    same = 0
    for i, x in enumerate(predictions):
        if x == labels[i]:
            same += 1
    if len(predictions) == 0:
        return 0
    else:
        return same / len(predictions)


def get_concrete_function(graph_def, inputs, outputs, print_graph=False):
    def imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrap_function = tf.compat.v1.wrap_function(imports_graph_def, [])
    graph = wrap_function.graph

    return wrap_function.prune(
        tf.nest.map_structure(graph.as_graph_element, inputs),
        tf.nest.map_structure(graph.as_graph_element, outputs))


def infer_perf_pb(pb_model_file, val_data, inputs=["x:0"], outputs=["Identity:0"]):
    x_test, y_test, label_test = val_data
    q_model = alexnet.load_pb(pb_model_file)
    concrete_function = get_concrete_function(graph_def=q_model.as_graph_def(),
                                              inputs=inputs,
                                              outputs=outputs,
                                              print_graph=True)

    bt = time.time()
    _frozen_graph_predictions = concrete_function(x=tf.constant(x_test))
    et = time.time()

    accuracy = calc_accuracy(_frozen_graph_predictions[0], label_test)
    print('accuracy:', accuracy)
    throughput = x_test.shape[0] / (et - bt)
    print('max throughput(fps):', throughput)

    # latency when BS=1
    times = 1000
    single_test = x_test[:1]

    bt = 0
    warmup = 20
    for i in range(times):
        if i == warmup:
            bt = time.time()
        _frozen_graph_predictions = concrete_function(x=tf.constant(single_test))
    et = time.time()

    latency = (et - bt) * 1000 / (times - warmup)
    print('latency(ms):', latency)

    return accuracy, throughput, latency


def save_res(result):
    accuracy, throughput, latency = result
    res = {}
    res['accuracy'] = accuracy
    res['throughput'] = throughput
    res['latency'] = latency

    outfile = args.index + ".json"
    with open(outfile, 'w') as f:
        json.dump(res, f)
        print("Save result to {}".format(outfile))

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=str, help='file name of output', required=True)

parser.add_argument('--input-graph', type=str, help='file name for graph', required=True)

parser.add_argument('--num-intra-threads', type=str, help='number of threads for an operator', required=False,
                    default="24" )
parser.add_argument('--num-inter-threads', type=str, help='number of threads across operators', required=False,
                    default="1")
parser.add_argument('--omp-num-threads', type=str, help='number of threads to use', required=False,
                    default="24")

args = parser.parse_args()
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "0"
os.environ["OMP_NUM_THREADS"] = args.omp_num_threads
os.environ["TF_NUM_INTEROP_THREADS"] = args.num_inter_threads
os.environ["TF_NUM_INTRAOP_THREADS"] = args.num_intra_threads

save_res(infer_perf_pb(args.input_graph, val_data()))
