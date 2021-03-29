import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
def main():

    import lpot
    quantizer = lpot.Quantization('./conf.yaml')
    quantized_model = quantizer("./mobilenet_v1_1.0_224_frozen.pb")
      
if __name__ == "__main__":

    main()
