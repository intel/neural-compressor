import tensorflow as tf
import time
import numpy as np

def main():

    from neural_compressor.experimental import Quantization,  common
    quantizer = Quantization('./conf.yaml')

    # Do quantization
    quantizer.model = common.Model('./inception_v1.ckpt')
    quantized_model = quantizer.fit()
  
     
if __name__ == "__main__":

    main()
