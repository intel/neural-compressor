import tensorflow as tf
import time
import numpy as np

tf.compat.v1.disable_eager_execution()

def main():

    import lpot
    quantizer = lpot.Quantization('./conf.yaml')
    quantized_model = quantizer('./inception_v1.ckpt')
  
     
if __name__ == "__main__":

    main()
