# Introduction of `autotune` API in INC 3.X


# User scripts
from typing import Union

import torch


class UserModel(torch.nn.Module):
    def __init__(self, dims):
        self.fc1 = torch.nn.Linear(dims, dims)
        self.fc2 = torch.nn.Linear(dims, dims)
        self.fc3 = torch.nn.Linear(dims, dims)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def build_simple_model(dims=10):
    return UserModel(dims)


def eval_fn(model) -> Union[float, int]:
    # Define a function to evaluate the model and return a score(float or int).
    # ...
    return 1.0


def calib_fn(model):
    #  Define a function to calibrate the model.
    # ...
    pass


# 1. Use the default tuning space.
from neural_compressor.torch.quantization import GPTQConfig, RTNConfig, TuningConfig, autotune

float_model = build_simple_model()
rtn_default_config_set = RTNConfig.get_config_set_for_tuning()

q_model = autotune(model=float_model, tune_config=TuningConfig(rtn_default_config_set), eval_fns=eval_fn)

# 2.1 Customize the tuning space for single algorithm.
float_model = build_simple_model()
customize_config_set = TuningConfig(config_set=RTNConfig(bits=[4, 8]))
q_model = autotune(model=float_model, tune_config=TuningConfig(customize_config_set), eval_fns=eval_fn)

# 2.2 Combine multiple algorithms into the tuning space.
float_model = build_simple_model()
customize_config_set = [RTNConfig(bits=[4, 8]), GPTQConfig(bits=[4, 8])]
q_model = autotune(model=float_model, tune_config=TuningConfig(customize_config_set), run_fn=calib_fn, eval_fns=eval_fn)
