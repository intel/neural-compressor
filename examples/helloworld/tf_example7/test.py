import tensorflow as tf
from argparse import ArgumentParser
from neural_compressor import conf
from neural_compressor.experimental import common

def main():
    arg_parser = ArgumentParser(description='Parse args')
    arg_parser.add_argument('--benchmark', action='store_true', help='run benchmark')
    arg_parser.add_argument('--tune', action='store_true', help='run tuning')
    args = arg_parser.parse_args()
    
    dataloader = {
        'dataset': {'dummy_v2': {'input_shape': [28, 28]}}
    }
    conf.quantization.calibration.dataloader = dataloader
    conf.evaluation.accuracy.dataloader = dataloader
    conf.tuning.accuracy_criterion.absolute = 0.9
    conf.evaluation.performance.dataloader = dataloader
    if args.tune:
        from neural_compressor.experimental import Quantization
        quantizer = Quantization(conf)
        quantizer.model = common.Model("../models/frozen_graph.pb")
        quantizer.fit()

    if args.benchmark:
        from neural_compressor.experimental import Benchmark
        evaluator = Benchmark(conf)
        evaluator.model = common.Model("../models/frozen_graph.pb")
        evaluator('performance')
      
if __name__ == "__main__":

    main()
