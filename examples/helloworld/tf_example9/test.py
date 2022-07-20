from neural_compressor.experimental import MixedPrecision, common
from neural_compressor import conf

def eval(model):
    return 0.5

def main():
    conf.tuning.exit_policy.max_trials = 10
    conf.tuning.exit_policy.timeout = 500
    conf.model.framework = 'tensorflow'
    converter = MixedPrecision(conf)
    converter.precisions = 'bf16'
    converter.input = 'input'
    converter.output = 'MobilenetV1/Predictions/Reshape_1'
    converter.model = common.Model('./mobilenet_v1_1.0_224_frozen.pb')
    converter.eval_func = eval
    output_model = converter.fit()

if __name__ == "__main__":

    main()