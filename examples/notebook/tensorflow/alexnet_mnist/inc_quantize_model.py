"""
Environment Setting

Enable Intel Optimized TensorFlow 2.6.0 and newer by setting environment variable TF_ENABLE_ONEDNN_OPTS=1
That will accelerate training and inference, and  it's mandatory requirement of running Intel® Neural Compressor quantize Fp32 model or deploying the quantized model.
"""

import neural_compressor as inc
print("neural_compressor version {}".format(inc.__version__))

import tensorflow as tf
print("tensorflow {}".format(tf.__version__))

from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.data import DataLoader
from neural_compressor.quantization import fit

import mnist_dataset


class Dataset(object):
    def __init__(self):
        _x_train, _y_train, label_train, x_test, y_test, label_test = mnist_dataset.read_data()

        self.test_images = x_test
        self.labels = label_test

    def __getitem__(self, index):
        return self.test_images[index], self.labels[index]

    def __len__(self):
        return len(self.test_images)

def ver2int(ver):
    s_vers = ver.split(".")
    res = 0
    for i, s in enumerate(s_vers):
        res += int(s)*(100**(2-i))

    return res

def compare_ver(src, dst):
    src_ver = ver2int(src)
    dst_ver = ver2int(dst)
    if src_ver>dst_ver:
        return 1
    if src_ver<dst_ver:
        return -1
    return 0

def auto_tune(input_graph_path, config, batch_size, int8_pb_file):
    dataset = Dataset()
    dataloader = DataLoader(framework='tensorflow', dataset=dataset, batch_size = batch_size)
    q_model = fit(
        model=input_graph_path,
        conf=config,
        calib_dataloader=dataloader,
        eval_dataloader=dataloader)

    return q_model


config = PostTrainingQuantConfig()
batch_size = 200
fp32_frozen_pb_file = "fp32_frozen.pb"
int8_pb_file = "alexnet_int8_model.pb"

q_model = auto_tune(fp32_frozen_pb_file, config, batch_size, int8_pb_file)
q_model.save(int8_pb_file)
