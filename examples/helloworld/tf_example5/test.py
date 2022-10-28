import tensorflow as tf
from argparse import ArgumentParser

import numpy as np
def main():
    arg_parser = ArgumentParser(description='Parse args')
    arg_parser.add_argument('--benchmark', action='store_true', help='run benchmark')
    arg_parser.add_argument('--tune', action='store_true', help='run tuning')
    args = arg_parser.parse_args()

    if args.tune:
        from neural_compressor.experimental import Quantization, common
        quantizer = Quantization('./conf.yaml')
        quantizer.model = common.Model("./mobilenet_v1_1.0_224_frozen.pb")
        quantized_model = quantizer.fit()
        quantized_model.save('./int8.pb')

    if args.benchmark:
        from neural_compressor.experimental import Benchmark, common
        evaluator = Benchmark('./conf.yaml')
        evaluator.model = common.Model('int8.pb')
        evaluator(mode='accuracy')

if __name__ == "__main__":

    main()
