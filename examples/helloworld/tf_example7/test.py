import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from neural_compressor import conf
def main():

    from neural_compressor.experimental import Quantization, Benchmark, common
    dataloader = {
        'dataset': {'dummy_v2': {'input_shape': [28, 28]}}
    }
    conf.evaluation.performance.dataloader = dataloader
    evaluator = Benchmark(conf)
    evaluator.model = common.Model("../models/frozen_graph.pb")
    evaluator('performance')

    conf.quantization.calibration.dataloader = dataloader
    conf.evaluation.accuracy.dataloader = dataloader
    conf.tuning.accuracy_criterion.absolute = 0.9
    quantizer = Quantization(conf)
    quantizer.model = common.Model("../models/frozen_graph.pb")
    quantizer.fit()
      
if __name__ == "__main__":

    main()
