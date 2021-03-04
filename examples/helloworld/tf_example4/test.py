import tensorflow as tf
import time
import numpy as np
from tensorflow import keras
from lpot.data import DATASETS, DataLoader

tf.compat.v1.disable_eager_execution()

def main():

    import lpot
    quantizer = lpot.Quantization('./conf.yaml')
    dataset = quantizer.dataset('dummy', shape=(100, 100, 100, 3), label=True)
    data_loader = DataLoader('tensorflow', dataset)
    model = quantizer.model('./model/public/rfcn-resnet101-coco-tf/model/public/rfcn-resnet101-coco-tf/rfcn_resnet101_coco_2018_01_28/')
    quantized_model = quantizer(model, q_dataloader=data_loader )

 
if __name__ == "__main__":

    main()
