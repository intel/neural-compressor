AutoTune
============

## Overview

Intel® Neural Compressor aims to help users quickly deploy low-precision models by leveraging popular compression techniques, such as post-training quantization and weight-only quantization algorithms. Despite having a variety of these algorithms, finding the appropriate configuration for a model can be difficult and time-consuming. To address this, we built the `autotune` module based on the [strategy](./tuning_strategies.md) in 2.x for accuracy-aware tuning, which identifies the best algorithm configuration for models to achieve optimal performance under the certain accuracy criteria. This module allows users to easily use predefined tuning recipes and customize the tuning space as needed.

## How it Works

The autotune module construct the tuning space according to the pre-defined tuning set or users' tuning set. It iterate the tuning space and apply the configuration on given float model then record and compare its evaluation result with the baseline. The tuning process stops when meeting the exit policy. The following figure provides a high level overview of the tuning process.

<FIG>



## Working with Autotune

The `autotune` API is used across all of frameworks supported by INC. It accepts three primary arguments: `model`, `tune_config`, and `eval_fn`.

The `TuningConfig` class defines the tuning process, including the tuning space, order, and exit policy.

- Define the tuning space

  User can define the tuning space by passing an algorithm configuration or a set of configurations.

- Define the tuning order

  The tuning order determines how the process traverses the tuning space and samples configurations. Users can customize it by configuring the `sampler`. Currently, we provide the `default_sampler`, which samples configurations sequentially, always in the same order.

- Define the exit policy

  The exit policy includes two components: accuracy goal (`tolerable_loss`) and the allowed number of trials (`max_trials`). The tuning process will stop when either condition is met.

### Example:

```python
def eval_acc(model) -> float:
    return ...


tune_config = TuningConfig(
    config_set=[RTNConfig(use_sym=False, group_size=32), GPTQConfig(group_size=128, use_sym=False)],
    tolerable_loss=0.2,
    max_trials=10,
)
q_model = autotune(model, tune_config=tune_config, eval_fn=eval_acc)
```
