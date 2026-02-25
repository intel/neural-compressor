AutoTune
========================================

1. [Overview](#overview)
2. [How it Works](#how-it-works)
3. [Working with Autotune](#working-with-autotune) \
    3.1 [Working with PyTorch Model](#working-with-pytorch-model) \
    3.1 [Working with Tensorflow Model](#working-with-tensorflow-model)


## Overview

Intel® Neural Compressor aims to help users quickly deploy low-precision models by leveraging popular compression techniques, such as post-training quantization and weight-only quantization algorithms. Despite having a variety of these algorithms, finding the appropriate configuration for a model can be difficult and time-consuming. To address this, we built the `autotune` module based on the [strategy](tuning_strategies.md) in 2.x for accuracy-aware tuning, which identifies the best algorithm configuration for models to achieve optimal performance under the certain accuracy criteria. This module allows users to easily use predefined tuning recipes and customize the tuning space as needed.

## How it Works

The autotune module constructs the tuning space according to the pre-defined tuning set or users' tuning set. It iterates the tuning space and applies the configuration on given float model then records and compares its evaluation result with the baseline. The tuning process stops when meeting the exit policy. 


## Working with Autotune

The `autotune` API is used across all of frameworks supported by INC. It accepts three primary arguments: `model`, `tune_config`, and `eval_fn`.

The `TuningConfig` class defines the tuning process, including the tuning space, order, and exit policy.

- Define the tuning space

  User can define the tuning space by setting `config_set` with an algorithm configuration or a set of configurations.
  ```python
  # Use the default tuning space
  config_set = get_woq_tuning_config()

  # Customize the tuning space with one algorithm configurations
  config_set = RTNConfig(use_sym=False, group_size=[32, 64])

  # Customize the tuning space with two algorithm configurations
  config_set = ([RTNConfig(use_sym=False, group_size=32), GPTQConfig(group_size=128, use_sym=False)],)
  ```

- Define the tuning order

  The tuning order determines how the process traverses the tuning space and samples configurations. Users can customize it by configuring the `sampler`. Currently, we provide the `default_sampler`, which samples configurations sequentially, always in the same order.

- Define the exit policy

  The exit policy includes two components: accuracy goal (`tolerable_loss`) and the allowed number of trials (`max_trials`). The tuning process will stop when either condition is met.

### Working with PyTorch Model
The example below demonstrates how to autotune a PyTorch model on four `RTNConfig` configurations.

```python
from neural_compressor.torch.quantization import RTNConfig, TuningConfig, autotune


def eval_fn(model) -> float:
    return ...


tune_config = TuningConfig(
    config_set=RTNConfig(use_sym=[False, True], group_size=[32, 128]),
    tolerable_loss=0.2,
    max_trials=10,
)
q_model = autotune(model, tune_config=tune_config, eval_fn=eval_fn)
```

### Working with Tensorflow Model

The example below demonstrates how to autotune a TensorFlow model on two `StaticQuantConfig` configurations.

```python
from neural_compressor.tensorflow.quantization import StaticQuantConfig, autotune

calib_dataloader = MyDataloader(...)
custom_tune_config = TuningConfig(
    config_set=[
        StaticQuantConfig(weight_sym=True, act_sym=True),
        StaticQuantConfig(weight_sym=False, act_sym=False),
    ]
)


def eval_fn(model) -> float:
    return ...


best_model = autotune(
    model="baseline_model", tune_config=custom_tune_config, eval_fn=eval_fn, calib_dataloader=calib_dataloader
)
```
