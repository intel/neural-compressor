# Tuning Strategies

## Introduction

Intel® Low Precision Optimization Tool aims to help users quickly deploy
the low-precision inference solution on popular Deep Learning frameworks
such as TensorFlow, PyTorch, and MxNet. Using built-in strategies, it
automatically optimizes low-precision recipes for deep learning models to
achieve optimal product objectives, such as inference performance and memory
usage, with expected accuracy criteria. Currently, it supports `Basic`, `Bayesian`, `Exhaustive`, `MSE`, `Random`, and `TPE` strategies. `Basic` is
the default strategy.

## Strategy Design

Each strategy generates the next quantization configuration according to its
logic and the last quantization result. The function of strategies is shown
below:

<div align="left">
  <img src="imgs/strategy.png" width="700px" />
</div>

Strategies begin with an adaptor layer (Framework Adaptor) where the user
passes a framework-specific model to initialize an instance of the
`lpot.Quantization() class`; strategies call the `self.adaptor.query_fw_capability(model)` to get the framework and
model-specific quantization capabilities. From there, each strategy merges
model-specific configurations in a `yaml` configuration file to filter some
capability from the first step in order to generate the tuning space. Each
strategy then generates the quantization config according to its location
and logic with tuning strategy configurations from the `yaml` configuration
file. All strategies finish the tuning processing when the `timeout` or `max_trails` is reached. The default value of `timeout` is 0; if reached, the
tuning phase stops when the `accuracy` criteria is met.

## Configurations

Detailed configuration templates can be found in [`here`](../lpot/template).

### Model-specific configurations

For model-specific configurations, users can set the quantization approach.
For post-training static quantization, users can also set calibration and
quantization-related parameters for model-wise and op-wise:

```yaml
quantization:                                        # optional. tuning constraints on model-wise for advance user to reduce tuning space.
  approach: post_training_static_quant               # optional. default value is post_training_static_quant.
  calibration:
    sampling_size: 1000, 2000                        # optional. default value is the size of whole dataset. used to set how many portions of calibration dataset is used. exclusive with iterations field.
    dataloader:                                      # optional. if not specified, user need construct a q_dataloader in code for lpot.Quantization.
      dataset:
        TFRecordDataset:
          root: /path/to/tf_record
      transform:
        Resize:
          size: 256
        CenterCrop:
          size: 224
  model_wise:                                        # optional. tuning constraints on model-wise for advance user to reduce tuning space.
    weight:
      granularity: per_channel
      scheme: asym
      dtype: int8
      algorithm: minmax
    activation:
      granularity: per_tensor
      scheme: asym
      dtype: int8, fp32
      algorithm: minmax, kl
  op_wise: {                                         # optional. tuning constraints on op-wise for advance user to reduce tuning space. 
         'conv1': {
           'activation':  {'dtype': ['uint8', 'fp32'], 'algorithm': ['minmax', 'kl'], 'scheme':['sym']},
           'weight': {'dtype': ['int8', 'fp32'], 'algorithm': ['kl']}
         },
         'pool1': {
           'activation': {'dtype': ['int8'], 'scheme': ['sym'], 'granularity': ['per_tensor'], 'algorithm': ['minmax', 'kl']},
         },
         'conv2': {
           'activation':  {'dtype': ['fp32']},
           'weight': {'dtype': ['fp32']}
         }
       }
```

### Strategy tuning part-related configurations

In strategy tuning part-related configurations, users can choose a specific
tuning strategy and then set the accuracy criterion and optimization
objective for tuning. Users can also set the `stop` condition for the tuning
by changing the `exit_policy`:

```yaml
tuning:
  strategy:
    name: basic                                      # optional. default value is basic. other values are bayesian, mse...
  accuracy_criterion:
    relative:  0.01                                  # optional. default value is relative, other value is absolute. this example allows relative accuracy loss: 1%.
  objective: performance                             # optional. objective with accuracy constraint guaranteed. default value is performance. other values are modelsize and footprint.

  exit_policy:
    timeout: 0                                       # optional. tuning timeout (seconds). default value is 0 which means early stop. combine with max_trials field to decide when to exit.
    max_trials: 100                                  # optional. max tune times. default value is 100. combine with timeout field to decide when to exit.

  random_seed: 9527                                  # optional. random seed for deterministic tuning.
  tensorboard: True                                  # optional. dump tensor distribution in evaluation phase for debug purpose. default value is False.
```

### Basic

#### Design

`Basic` strategy is designed for most models to do quantization. It includes
three steps. First, `Basic` strategy tries all model-wise tuning configs to
get the best quantized model. If none of the model-wise tuning configs meet
the accuracy loss criteria, Basic applies the second step. In this step, it
performs high-precision `OP` (`FP32`, `BF16` ...) fallbacks one-by-one based
on the best model-wise tuning config, and records the impact of each `OP` on
accuracy and then sorts accordingly. In the final step, Basic tries to
incrementally fallback multiple `OPs` to high precision according to the
sorted `OP` list that is generated in the second step until the accuracy
goal is achieved.

#### Usage

`Basic` is the default strategy. It can be used by default if you don't add
the `strategy` field in your `yaml` configuration file. Classical settings in the configuration file are shown below:

```yaml
tuning:
  accuracy_criterion:
    relative:  0.01
  exit_policy:
    timeout: 0
  random_seed: 9527
```

### Bayesian

#### Design

`Bayesian` optimization is a sequential design strategy for the global
optimization of black-box functions. This strategy takes the [Bayesian
optimization](https://github.com/fmfn/BayesianOptimization) package and
changes it to a discrete version that complies with the strategy standard of
Intel® Low Precision Optimization Tool. It uses [Gaussian processes](https://en.wikipedia.org/wiki/Neural_network_Gaussian_process) to define
the prior/posterior distribution over the black-box function with the tuning
history, and then finds the tuning configuration that maximizes the expected
improvement. For now, the Bayesian strategy just tunes op-wise quantize
configs; it does not include fallback-datatype configs.

#### Usage

For the `Bayesian` strategy, set the `timeout` or `max_trials` to a non-zero
value as shown in the below example. This is because the param space for `bayesian` can be very small so the accuracy goal might not be reached which
can make the tuning never end. Additionally, if the log level is set to `debug` by `LOGLEVEL=DEBUG` in the environment, the message `[DEBUG] Tuning config was evaluated, skip!` will print endlessly. If the timeout is changed from 0 to an integer, `Bayesian` ends after the timeout is reached.


```yaml
tuning:
  strategy:
    name: bayesian
  accuracy_criterion:
    relative:  0.01
  objective: performance

  exit_policy:
    timeout: 0
    max_trials: 100
```

### MSE

#### Design

`MSE` and `Basic` strategies share similar ideas. The primary difference
between the two strategies is the way sorted op lists are generated in step
2. The `MSE` strategy needs to get the tensors for each operator of raw FP32
models and the quantized model based on the best model-wise tuning
configuration. It then calculates the MSE (Mean Squared Error) for each
operator, sorts those operators according to the MSE value, and performs
the op-wise fallback in this order.

#### Usage

`MSE` is similar to `Basic` but the specific strategy name of `mse` must be
included.

```yaml
tuning:
  strategy:
    name: mse
  accuracy_criterion:
    relative:  0.01
  exit_policy:
    timeout: 0
  random_seed: 9527
```

### TPE

#### Design

`TPE` uses sequential model-based optimization methods (SMBOs). **Sequential
** refers to running trials one after another and selecting a better
**hyperparameter** to evaluate based on previous trials. A hyperparameter is
a parameter whose value is set before the learning process begins; it
controls the learning process. SMBO apples Bayesian reasoning in that it
updates a **surrogate** model that represents an **objective** function
(objective functions are more expensive to compute). Specifically, it finds
hyperparameters that perform best on the surrogate and then applies them to
the objective function. The process is repeated and the surrogate is updated
with incorporated new results until the timeout or max trials is reached.

A surrogate model and selection criteria can be built in a variety of ways.
`TPE` builds a surrogate model by applying Bayesian reasoning. The TPE
algorithm consists of the following steps:

1. Define a domain of hyperparameter search space.
2. Create an objective function which takes in hyperparameters and outputs a
score (e.g., loss, RMSE, cross-entropy) that we want to minimize.
3. Collect a few observations (score) using a randomly selected set of
hyperparameters.
4. Sort the collected observations by score and divide them into two groups
based on some quantile. The first group (x1) contains observations that
gives the best scores and the second one (x2) contains all other
observations.
5. Model the two densities l(x1) and g(x2) using Parzen Estimators (also known as kernel density estimators) which are a simple average of kernels centered on existing data points.
6. Draw sample hyperparameters from l(x1). Evaluate them in terms of l(x1)/g(x2), and return the set that yields the minimum value under l(x1)/g(x1) that
corresponds to the greatest expected improvement. Evaluate these
hyperparameters on the objective function.
7. Update the observation list in step 3.
8. Repeat steps 4-7 with a fixed number of trials.

>Note: TPE requires many iterations in order to reach an optimal solution;
we recommend running at least 200 iterations. Because every iteration
requires evaluation of a generated model--which means accuracy measurements
on a dataset and latency measurements using a benchmark--this process can
take from 24 hours to few days to complete, depending on the model.

#### Usage

`TPE` usage is similar to `Bayesian`:

```yaml
tuning:
  strategy:
    name: bayesian
  accuracy_criterion:
    relative:  0.01
  objective: performance

  exit_policy:
    timeout: 0
    max_trials: 100
```

### Exhaustive

#### Design

`Exhaustive` strategy is used to sequentially traverse all possible tuning
configurations in a tuning space. From the perspective of the impact on
performance, we currently only traverse all possible quantize tuning
configs. Fallback datatypes are not included.

#### Usage

`Exhaustive` usage is similar to `Basic`:

```yaml
tuning:
  strategy:
    name: exhaustive
  accuracy_criterion:
    relative:  0.01
  exit_policy:
    timeout: 0
  random_seed: 9527
```

### Random

#### Design

`Random` strategy is used to randomly choose tuning configurations from the
tuning space. As with `Exhaustive` strategy, it also only considers quantize
tuning configs to generate a better-performance quantized model.

#### Usage

`Random` usage is similar to `Basic`:

```yaml
tuning:
  strategy:
    name: random 
  accuracy_criterion:
    relative:  0.01
  exit_policy:
    timeout: 0
  random_seed: 9527

```

Customize a New Tuning Strategy
======================

Intel® Low Precision Optimization Tool supports new strategy extension by implementing a subclass of `TuneStrategy` class in lpot.strategy package
 and registering this strategy by `strategy_registry` decorator.

for example, user can implement a `Abc` strategy like below:

```
@strategy_registry
class AbcTuneStrategy(TuneStrategy):
    def __init__(self, model, conf, q_dataloader, q_func=None,
                 eval_dataloader=None, eval_func=None, dicts=None):
        ...

    def next_tune_cfg(self):
        ...

```

The `next_tune_cfg` function is used to yield the next tune configuration according to some algorithm or strategy. `TuneStrategy` base class will traverse
 all the tuning space till a quantization configuration meets pre-defined accuray criterion.

If the traverse behavior of `TuneStrategy` base class does not meet new strategy requirement, it could re-implement `traverse` function with self own logic.
An example like this is under [TPE Strategy](../lpot/strategy/tpe.py).
