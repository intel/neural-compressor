import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
def main():

    from lpot.experimental import Quantization,  common
    quantizer = Quantization('./conf.yaml')
    quantizer.model = common.Model("./mobilenet_v1_1.0_224_frozen.pb")
    quantized_model = quantizer()

     # Optional, run benchmark 
    from lpot.experimental import Benchmark
    evaluator = Benchmark('./conf.yaml')
    evaluator.model = common.Model(quantized_model.graph_def)
    evaluator(mode='accuracy')

if __name__ == "__main__":

    main()
