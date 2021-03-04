import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
def main():

    import lpot
    quantizer = lpot.Quantization('./conf.yaml')
    model = quantizer.model("./mobilenet_v1_1.0_224_frozen.pb")
    quantized_model = quantizer(model)
      
if __name__ == "__main__":

    main()
