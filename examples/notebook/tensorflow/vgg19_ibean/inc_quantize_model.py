"""
Environment Setting

Enable Intel Optimized TensorFlow 2.6.0 and newer by setting environment variable TF_ENABLE_ONEDNN_OPTS=1
That will accelerate training and inference, and  it's mandatory requirement of running Intel® Neural Compressor quantize Fp32 model or deploying the quantized model.
"""

import neural_compressor as inc
print("neural_compressor version {}".format(inc.__version__))

import tensorflow as tf
print("tensorflow {}".format(tf.__version__))

from neural_compressor.experimental import Quantization, common

import numpy as np
import tensorflow_datasets as tfds



# define class number
class_num=3

# define input image size and class number
w=h=32


def preprocess(image, label):
    image = tf.cast(image, tf.float32)/255.0
    return tf.image.resize(image, [w, h]), tf.one_hot(label, class_num)


def load_raw_dataset():
    raw_datasets, raw_info = tfds.load(name = 'beans', with_info = True,
                                       as_supervised = True, 
                                       split = ['train', 'test'])
    return raw_datasets, raw_info
    
class Dataset(object):
    def __init__(self):
        datasets , info = load_raw_dataset()        
        self.test_dataset = [preprocess(v, l) for v,l in datasets[-1]]
    
    def __getitem__(self, index):
        return self.test_dataset[index]

    def __len__(self):
        return len(list(self.test_dataset))

def auto_tune(input_graph_path, yaml_config, batch_size, int8_pb_file):
    quantizer = Quantization(yaml_config)
    dataset = Dataset()
    quantizer.calib_dataloader = common.DataLoader(dataset, batch_size=batch_size)
    quantizer.eval_dataloader = common.DataLoader(dataset, batch_size=batch_size)
    quantizer.model = common.Model(input_graph_path)
    q_model = quantizer.fit()

    return q_model


yaml_file = "vgg19.yaml"
batch_size = 32
model_fp32_path="model_keras.fp32"
int8_pb_file = "model_pb.int8"
q_model = auto_tune(model_fp32_path, yaml_file, batch_size, int8_pb_file)
q_model.save(int8_pb_file)
print("Save quantized model to {}".format(int8_pb_file))

