import tensorflow as tf
import time
import numpy as np
from tensorflow import keras
from lpot.experimental import Quantization,  common

tf.compat.v1.disable_eager_execution()

def main():

    quantizer = Quantization('./conf.yaml')
    dataset = quantizer.dataset('dummy_v2', \
        input_shape=(100, 100, 3), label_shape=(1, ))
    quantizer.model = common.Model('./model/public/rfcn-resnet101-coco-tf/rfcn_resnet101_coco_2018_01_28/')
    quantizer.calib_dataloader = common.DataLoader(dataset)
    quantized_model = quantizer()

 
if __name__ == "__main__":

    main()
