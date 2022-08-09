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

def auto_tune(input_graph_path, yaml_config, batch_size, int8_pb_file):
    quantizer = Quantization(yaml_config)
    dataset = Dataset()
    quantizer.calib_dataloader = common.DataLoader(dataset, batch_size=batch_size)
    quantizer.eval_dataloader = common.DataLoader(dataset, batch_size=batch_size)
    quantizer.model = common.Model(input_graph_path)
    if compare_ver(inc.__version__, "1.9")>=0:
        q_model = quantizer.fit()
    else:
        q_model = quantizer()

    return q_model


yaml_file = "alexnet.yaml"
batch_size = 200
fp32_frozen_pb_file = "fp32_frozen.pb"
int8_pb_file = "alexnet_int8_model.pb"

q_model = auto_tune(fp32_frozen_pb_file, yaml_file, batch_size, int8_pb_file)
q_model.save(int8_pb_file)
