class FakeModel:
    pass

class FakeTuneConfig:
    pass

def get_fp32_model():
    return FakeModel()

# user usage
# case 1. auto-tune with default tuning config
def eval_fn(model):
    return 1.0

from neural_compressor.torch import autotune, get_default_tune_config, register_tuning_target

best_model = autotune(model=get_fp32_model(), tune_config=get_default_tune_config())


# case 2. auto-tune with custom tuning config
from neural_compressor.torch import GPTQConfig, RTNWeightQuantConfig, TuningConfig, autotune, register_tuning_target

@register_tuning_target(weight = 0.5, name="accuracy")
def eval_acc_fn(model) -> float:
    return 1.0

@register_tuning_target(algo_name="rtn_weight_only_quant", weight = 0.5, mode = "min")
def eval_perf_fn(model) -> float:
    return 1.0

def calib_fn(model):
    pass

custom_tune_config = TuningConfig(
    tuning_order=[GPTQConfig(weight_bits=[4, 6]), RTNWeightQuantConfig(weight_bits=[4, 6])], max_trials=3
)
best_model = autotune(model=get_fp32_model(), tune_config=custom_tune_config, run_fn=calib_fn)
