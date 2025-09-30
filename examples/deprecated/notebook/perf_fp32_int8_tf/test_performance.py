#this .py assumes that both models are correctedly downloaded, and corresponding environment is set up and activated

#import needed packages

import tensorflow as tf
print("Tensorflow version {}".format(tf.__version__))
 
import numpy as np
import time
import argparse
import os, sys
import json
        
#necessary functions
def get_concrete_function(graph_def, inputs, outputs, print_graph=False):
    def imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrap_function = tf.compat.v1.wrap_function(imports_graph_def, [])
    graph = wrap_function.graph

    return wrap_function.prune(
        tf.nest.map_structure(graph.as_graph_element, inputs),
        tf.nest.map_structure(graph.as_graph_element, outputs))

def save_res(result, dtype):
    throughput, latency = result
    res = {}
    res['throughput'] = throughput
    res['latency'] = latency

    outfile = (str)(dtype)+".json"
    with open(outfile, 'w') as f:
        json.dump(res, f)
        print("Save result to {}".format(outfile))

#run inference using the model and x_test, and return throughput and latency
def run_infer(batchsize, dtype, x_test, model_file):
    print("Running %s model..."%(dtype))

    with tf.compat.v1.Session() as sess:
        print("load graph")
    with tf.io.gfile.GFile(model_file,'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    graph_nodes=[n for n in graph_def.node]
    names = []

    with tf.Graph().as_default() as graph:
        for op in graph.get_operations():
            print("Operation Name :" + op.name)
            print("Tensor Stats :" + str(op.values()))

    concrete_function = get_concrete_function(graph_def=graph_def,
                                                  inputs=["input:0"],
                                                  outputs=["predict:0"],
                                                  print_graph=True)
    
    batches = x_test.reshape((x_test.shape[0]//batchsize,batchsize,224,224,3))
    
    totaltime = 0
    
    for batch in batches:
        bt = time.time()
        _frozen_graph_predictions = concrete_function(input=tf.constant(batch))
        et = time.time()
    
        totaltime = totaltime + (et-bt)

    throughput = x_test.shape[0] / (et - bt)

    print('max throughput(fps):', throughput)

    # latency when BS=1
    times = 100
    single_test = x_test[:1]

    bt = 0
    warmup = 20
    for i in range(times):
        if i == warmup:
            bt = time.time()
        _frozen_graph_predictions = concrete_function(input=tf.constant(single_test))
    et = time.time()

    latency = (et - bt) * 1000 / (times - warmup)
    print('latency(ms):', latency)

    return throughput,latency

def main(fp32_model, int8_model):
    #the batch size for estimating throughput
    batchsize = 64

    x_test = np.random.random((640,224,224,3))
    x_test = x_test.astype('float32')

    save_res(run_infer(batchsize,"int8",x_test, int8_model),"int8")
    save_res(run_infer(batchsize,"fp32",x_test, fp32_model),"fp32")

if __name__ == "__main__":
    if len(sys.argv)<3:
        print("Miss parameters!\n{} fp32_model int8_model".format(sys.argv[0]))
        sys.exit(1)
    fp32_model = sys.argv[1]
    int8_model = sys.argv[2]
    main(fp32_model, int8_model)
