class FakeModel:
    pass

def get_fp32_model():
    return FakeModel()

# user usage
# auto-tune with default tuning config
def eval_fn(model):
    return 1.0


from neural_compressor.torch import TuningConfig, autotune

best_model = autotune(model=get_fp32_model(), tune_config=TuningConfig(), eval_fn=eval_fn)

# auto-tune with custom tuning config
from neural_compressor.torch import GPTQConfig, RTNWeightQuantConfig, TuningConfig, autotune

custom_tune_config = TuningConfig(
    tuning_order=[GPTQConfig(weight_dtype=[4, 6]), RTNWeightQuantConfig(weight_bits=[4, 6])], max_trials=3
)
best_model = autotune(model=get_fp32_model(), tune_config=custom_tune_config, eval_fn=eval_fn)
