import tensorflow as tf
from argparse import ArgumentParser
tf.compat.v1.disable_eager_execution()

import numpy as np
def main():
    arg_parser = ArgumentParser(description='Parse args')
    arg_parser.add_argument('--benchmark', action='store_true', help='run benchmark')
    arg_parser.add_argument('--tune', action='store_true', help='run tuning')
    args = arg_parser.parse_args()

    if args.tune:
        from neural_compressor import Quantization
        quantizer = Quantization('./conf.yaml')
        quantized_model = quantizer("./mobilenet_v1_1.0_224_frozen.pb")
        tf.io.write_graph(graph_or_graph_def=quantized_model,
                          logdir='./',
                          name='int8.pb',
                          as_text=False)

    if args.benchmark:
        from neural_compressor import Benchmark
        evaluator = Benchmark('./conf.yaml')
        results = evaluator('./int8.pb')
        batch_size = 1
        for mode, result in results.items():
           acc, batch_size, result_list = result
           latency = np.array(result_list).mean() / batch_size

           print('Accuracy is {:.3f}'.format(acc))
           print('Latency: {:.3f} ms'.format(latency * 1000))
      
if __name__ == "__main__":

    main()
