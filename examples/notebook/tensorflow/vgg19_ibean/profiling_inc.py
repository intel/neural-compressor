
import tensorflow as tf
print("Tensorflow version {}".format(tf.__version__))
tf.config.run_functions_eagerly(False)

import numpy as np
import time
import argparse
import os
import json
import tensorflow_hub as hub
import tensorflow_datasets as tfds

w=h=32
class_num=3
def scale(image, label):
    w=224
    h=224
    class_num=3
    image = tf.cast(image, tf.float32)
    image /= 255.0
    
    return tf.image.resize(image, [w, h]), tf.one_hot(label, class_num)

def val_data():
    datasets , info = tfds.load(name = 'beans', with_info = True, as_supervised = True, split = ['train'])
    valdataset = [scale(v, l) for v,l in datasets[-1]]    
    return valdataset

def load_raw_dataset():
    raw_datasets, _raw_info = tfds.load(name = 'beans', with_info = True, as_supervised = True, split = ['train'],
                                       batch_size=-1)
    ds_numpy = tfds.as_numpy(raw_datasets)
    return ds_numpy

def preprocss(dataset):
    [images, labels]= dataset
    inputs = []
    res = []
    for image in images:        
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image /= 255.0
        image = tf.image.resize(image, [w, h])
        inputs.append(image)
        
    for label in labels:        
        res.append(tf.one_hot(label, class_num))
        
    return np.array(inputs), np.array(res)
        
def load_dataset():
    return [preprocss(dataset) for dataset in load_raw_dataset()]

def calc_accuracy(predictions, labels):  
    predictions = np.argmax(predictions.numpy(), axis=1)
    labels = np.argmax(labels, axis=1)
    
    same = 0
    for i, x in enumerate(predictions):
        if x == labels[i]:
            same += 1
    if len(predictions) == 0:
        return 0
    else:
        return same / len(predictions)

def test_perf(pb_model_file, val_data):
    [x_test_np, label_test] = val_data
    q_model = tf.saved_model.load(pb_model_file)
    x_test = tf.convert_to_tensor(x_test_np)
    infer = q_model.signatures["serving_default"]
    
    times = 10

    bt = 0
    warmup = int(times*0.2)
    for i in range(times):
        if i == warmup:
            bt = time.time()
        res = infer(x_test)
    et = time.time()
    
    res = list(res.values())[0]
    accuracy = calc_accuracy(res, label_test)
    print('accuracy:', accuracy)

    throughput = len(x_test)*(times-warmup)/(et-bt)
    print('max throughput(fps):', throughput)
   
    
     # latency when BS=1
    times = 1

    bt = 0
    warmup = int(times*0.2)
    for i in range(times):
        if i == warmup:
            bt = time.time()
        for i in range(len(x_test)):
           res = infer(tf.convert_to_tensor([x_test_np[i]])) 
        #q_model.test_on_batch(val_data, verbose=0)
    et = time.time()

    latency = (et - bt) * 1000 / (times - warmup)/len(x_test)
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
#os.environ["DNNL_VERBOSE"] = "1"
datasets = load_dataset()
print(len(datasets[-1]))
save_res(test_perf(args.input_graph, datasets[-1]))
