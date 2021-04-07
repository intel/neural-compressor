import tensorflow as tf
import time
import numpy as np

tf.compat.v1.disable_eager_execution()

def main():

    from lpot.experimental import Quantization,  common
    quantizer = Quantization('./conf.yaml')

    # Do quantization
    quantizer.model = common.Model('./inception_v1.ckpt')
    quantized_model = quantizer()
  
     
if __name__ == "__main__":

    main()
