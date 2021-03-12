import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
def main():

    import lpot
    from lpot import common
    quantizer = lpot.Quantization('./conf.yaml')
    quantizer.model = common.Model("./mobilenet_v1_1.0_224_frozen.pb")
    quantized_model = quantizer()
      
if __name__ == "__main__":

    main()
