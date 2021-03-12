import tensorflow as tf
import time
import numpy as np
from tensorflow import keras
from lpot.data import DATASETS, DataLoader
from lpot import common

tf.compat.v1.disable_eager_execution()

def main():

    import lpot
    quantizer = lpot.Quantization('./conf.yaml')
    dataset = quantizer.dataset('dummy', shape=(100, 100, 100, 3), label=True)
    quantizer.model = common.Model('./model/public/rfcn-resnet101-coco-tf/model/public/rfcn-resnet101-coco-tf/rfcn_resnet101_coco_2018_01_28/')
    quantizer.calib_dataloader = common.DataLoader(dataset)
    quantized_model = quantizer()

 
if __name__ == "__main__":

    main()
