import tensorflow as tf

import numpy as np
def main():

    from neural_compressor.experimental import Quantization, common
    quantizer = Quantization('./conf.yaml')
    quantizer.model = common.Model("./mobilenet_v1_1.0_224_frozen.pb")
    quantized_model = quantizer.fit()
      
if __name__ == "__main__":

    main()
