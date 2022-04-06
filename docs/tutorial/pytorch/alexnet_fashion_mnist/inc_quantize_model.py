"""
Environment Setting

Enable Intel Optimized TensorFlow 2.6.0 and newer by setting environment variable TF_ENABLE_ONEDNN_OPTS=1
That will accelerate training and inference, and  it's mandatory requirement of running Intel® Neural Compressor quantize Fp32 model or deploying the quantized model.
"""

import neural_compressor as inc
print("neural_compressor version {}".format(inc.__version__))

import torch
print("torch {}".format(torch.__version__))

from neural_compressor.experimental import Quantization, common
import fashion_mnist
import alexnet


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

def auto_tune(input_graph_path, yaml_config, batch_size):
    quantizer = Quantization(yaml_config)
    train_dataset, test_dataset = fashion_mnist.download_dataset()

    quantizer.calib_dataloader = common.DataLoader(test_dataset, batch_size=batch_size)
    quantizer.eval_dataloader = common.DataLoader(test_dataset, batch_size=batch_size)
    quantizer.model = alexnet.load_mod(input_graph_path)
    if compare_ver(inc.__version__, "1.9")>=0:
        q_model = quantizer.fit()
    else:
        q_model = quantizer()

    return q_model

def main():
    yaml_file = "alexnet.yaml"
    batch_size = 200
    fp32_model_file = "alexnet_mnist_fp32_mod.pth"
    int8_model = "alexnet_mnist_int8_mod"

    q_model = auto_tune(fp32_model_file, yaml_file, batch_size)
    q_model.save(int8_model)
    print("Save int8 model to {}".format(int8_model))

if __name__ == "__main__":
    main()
    