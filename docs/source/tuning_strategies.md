Tuning Strategies
============

1. [Introduction](#introduction)

2. [Strategy Design](#strategy-design)

    2.1. [Tuning Space](#tuning-space)

	2.2. [Exist Policy](#exist-policy)

	2.3. [Accuracy Criteria](#accuracy-criteria)
3. [Traverse Logics](#traverse-logics)

    3.1. [Basic](#basic)

    3.2. [Bayesian](#bayesian)

    3.3. [MSE](#mse)

    3.4. [MSE_v2](#mse_v2)

    3.5. [HAWQ_V2](#hawq_v2)

    3.6. [TPE](#tpe)

    3.7. [Exhaustive](#exhaustive)

    3.8. [Random](#random)

    3.9. [SigOpt](#sigOpt)


 4. [Customize a New Tuning Strategy](#customize-a-new-tuning-strategy)

## Introduction

Intel® Neural Compressor aims to help users quickly deploy
the low-precision inference solution on popular Deep Learning frameworks
such as TensorFlow, PyTorch, ONNX, and MXNet. Using built-in strategies, it
automatically optimizes low-precision recipes for deep learning models to
achieve optimal product objectives, such as inference performance and memory
usage, with expected accuracy criteria. Currently, several strategies, including
`Basic`, `Bayesian`, `Exhaustive`, `MSE`, `MSE_v2`, `Hawq_v2`, `Random`, `SigOpt`, `TPE`, 
 etc is supported. By default, `Basic` strategy is used for tuning.

## Strategy Design
Before the tuning, the `tuning space` was constructed according to the framework capability and user configuration. Then the selected strategy drive to generates the next quantization configuration according to its traverse logic and the previous tuning record. The tuning process stop when meet the exist policy. The function of strategies is shown
below:

![Tuning Strategy](./_static/imgs/strategy.png "Strategy Framework")

### Tuning Space

The `tuning space` include all tunable items and their options, for example calibration sampling size, quantization scheme(scheme/asymmetric), or select parts of model not quantize. 
Specifically, for one op, the tuning items and options are as following.


To incorporate the human experience and reduce the tuning time, user can reduce the tuning space by specifying the `op_name_list` and `op_type_list`. Before tuning, the strategy will merge these user configurations with framework capability to create the final tuning space.

### Exist policy
User can control the tuning process by setting the exist policy by specifying the `timeout`, and `max_trials`.

```python
from neural_compressor.config import TuningCriterion

tuning_criterion=TuningCriterion(
    timeout=0, # optional. tuning timeout (seconds). when set to 0, early stop is enabled.
    max_trials=100, # optional. max tuning times. combined with `timeout` field to decide when to exit tuning.
    strategy="basic", # optional. name of tuning strategy. 
    strategy_kwargs=None, # optional. see concrete tuning strategy for available settings.
)
```


### Accuracy Criteria
User can set the accuracy criteria by specifying the `higher_is_better`, `criterion`, and `tolerable_loss`.

``` python
from neural_compressor.config import AccuracyCriterion

accuracy_criterion=AccuracyCriterion(
    higher_is_better=True, # optional. 
    criterion='relative', # optional. available values are 'relative' and 'absolute'.
    tolerable_loss=0.01, # optional.
)
```

## Traverse Logics

### Basic

#### Design

`Basic` strategy is designed for most models to do quantization. It includes
three stages and each stage is executed sequentially, and the tuning process ends once the codition meets the exist policy. 
- Stage I. Op-type-wise tuning

    In this stage, it try to quantized the OPs as many as possible and traverse all op-type-wise tuning configs. Note that, we initial the op with difference quantization mode according to the quantization approach.
    
    a. `post_training_static_quant`: We quantized all OPs support Post Training Static Quantization(PTQ static).

    b. `post_training_dynamic_quant`: We quantized all OPs support Post Training Dynamic Quantization(PTQ Dynamic).
    
    c. `post_training_auto_quant`: it quantized all OPs support PTQ static or PTQ Dynamic. For the OPs support both PTQ static or PTQ Dynamic, we first try to do PTQ static for them, if none of the Op-type-wise tuning configs meet the accuracy loss criteria, we try do PTQ Dynamic for them.

- Stage II. Fallback op one-by-one
    In this stage, it performs high-precision OP (FP32, BF16 ...) fallbacks one-by-one based on the tuning config with best result in the previous stage, and records the impact of each OP. 

- Stage III.Fallback multiple operations accumulated
    In final stage, it first sorted OPs list according to the impact score in the stage II, and tries to incrementally fallback multiple OPs to high precision according to the sorted OP list.

#### Usage

`Basic` is the default strategy. It can be used by default if you don't change the
the `strategy` field in the `tuning_criterion` setting. Classical settings are shown below:

```python
from neural_compressor import config

conf = config.PostTrainingQuantConfig(
    tuning_criterion=config.TuningCriterion(
        strategy="basic", # optional. name of tuning strategy. 
    ),
)
```

### Bayesian

#### Design

`Bayesian` optimization is a sequential design strategy for the global
optimization of black-box functions. This strategy comes from the [Bayesian
optimization](https://github.com/fmfn/BayesianOptimization) package and
changed it to a discrete version that complied with the strategy standard of
Intel® Neural Compressor. It uses [Gaussian processes](https://en.wikipedia.org/wiki/Neural_network_Gaussian_process) to define
the prior/posterior distribution over the black-box function with the tuning
history, and then finds the tuning configuration that maximizes the expected
improvement. For now, `Bayesian` just focus on op-wise quantize configs tuning 
without fallback phase. In order to obtain a quantized model with good accuracy 
and better performance in a short time, we don't add datatype as a tuning 
parameter into `Bayesian`.

#### Usage

For the `Bayesian` strategy, set the `timeout` or `max_trials` to a non-zero
value as shown in the below example. This is because the param space for `bayesian` can be very small so the accuracy goal might not be reached which
can make the tuning never end. Additionally, if the log level is set to `debug` by `LOGLEVEL=DEBUG` in the environment, the message `[DEBUG] Tuning config was evaluated, skip!` will print endlessly. If the `timeout` is changed from 0 to an integer, `Bayesian` ends after the timeout is reached.

```python
from neural_compressor import config

conf = config.PostTrainingQuantConfig(
    tuning_criterion=config.TuningCriterion(
        timeout=0, # optional. tuning timeout (seconds). when set to 0, early stop is enabled.
        max_trials=100, # optional. max tuning times. combined with `timeout` field to decide when to exit tuning.
        strategy="bayesian", # optional. name of tuning strategy. 
    ),
)
```

### MSE

#### Design

`MSE` and `Basic` strategies share similar ideas. The primary difference
between the two strategies is the way sorted op lists are generated in stage II. The `MSE` strategy needs to get the tensors for each operator of raw FP32
models and the quantized model based on the best model-wise tuning
configuration. It then calculates the MSE (Mean Squared Error) for each
operator, sorts those operators according to the MSE value, and performs
the op-wise fallback in this order.

#### Usage

`MSE` is similar to `Basic` but specific the `strategy` field with `mse` in the `tuning_criterion` setting.

```python
from neural_compressor import config

conf = config.PostTrainingQuantConfig(
    tuning_criterion=config.TuningCriterion(
        strategy="mse",
    ),
)
```

### MSE_v2

#### Design

`MSE_v2` is a two-stage fallback strategy, which is composed of three key components. First, a multi-batch order combination based on per-layer fallback MSE values helps evaluate layer sensitivity with few-shot. Second, a sensitivity gradient is proposed to better evaluate the sensitivity, together with the beam search to solve 
the local optimum problem. Third, a quantize-again procedure is introduced 
to remove redundancy in fallback layers to protect performance. MSE_v2 performs
better especially in models with a long full-dataset evaluation time and a 
large number of tuning counts.

#### Usage
To use the `MSE_v2` tuning strategy, your need to specific the `strategy` field with `mse_v2` in the `tuning_criterion` setting. Also, the option
`confidence_batches` can be set optionally inside the `strategy_kwargs` to specify the number of batches to calculate the op sensitivity. Increasing the `confidence_batches` will generally improve the accuracy of the scoring of the impact of OPs at the cost of more time spent in the process of sorting the OPs.

```python
from neural_compressor import config

conf = config.PostTrainingQuantConfig(
    tuning_criterion=config.TuningCriterion(
        strategy="mse_v2",
        strategy_kwargs={"confidence_batches":2},
    ),
)
```

### HAWQ_V2

#### Design
`HAWQ_V2` implements the [Hessian Aware trace-Weighted Quantization of Neural Networks](https://arxiv.org/abs/1911.03852). We made a small change to it by using the hessian trace to scoring the op impact and then fallback the OPs according the scoring result. 

#### Usage
To use the `HAQW_V2` tuning strategy, your need to specific the `strategy` field with `HAQW_V2` in the `tuning_criterion` setting and provide the loss function for calculate the hessian trace. The `hawq_v2_loss` should be set in the filed of `hawq_v2_loss` in the `strategy_kwargs`.

```python
from neural_compressor import config

def model_loss(output, target, criterion):
    return criterion(output, target)

conf = config.PostTrainingQuantConfig(
    tuning_criterion=config.TuningCriterion(
        strategy="hawq_v2",
        strategy_kwargs={"hawq_v2_loss": model_loss},
    ),
)
```

### TPE

#### Design

`TPE` uses sequential model-based optimization methods (SMBOs). **Sequential** refers to running trials one after another and selecting a better
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

- Define a domain of hyperparameter search space.
- Create an objective function which takes in hyperparameters and outputs a
score (e.g., loss, RMSE, cross-entropy) that we want to minimize.
- Collect a few observations (score) using a randomly selected set of
hyperparameters.
- Sort the collected observations by score and divide them into two groups
based on some quantile. The first group (x1) contains observations that
gives the best scores and the second one (x2) contains all other
observations.
- Model the two densities l(x1) and g(x2) using Parzen Estimators (also known as kernel density estimators) which are a simple average of kernels centered on existing data points.
- Draw sample hyperparameters from l(x1). Evaluate them in terms of l(x1)/g(x2), and return the set that yields the minimum value under l(x1)/g(x1) that
corresponds to the greatest expected improvement. Evaluate these
hyperparameters on the objective function.
- Update the observation list in step 3.
8. Repeat steps 4-7 with a fixed number of trials.

>Note: TPE requires many iterations in order to reach an optimal solution;
we recommend running at least 200 iterations. Because every iteration
requires evaluation of a generated model--which means accuracy measurements
on a dataset and latency measurements using a benchmark--this process can
take from 24 hours to few days to complete, depending on the model.

#### Usage

`TPE` usage is similar to `basic` but specific the `strategy` field with `tpe` in the `tuning_criterion` setting.

```python
from neural_compressor import config

conf = config.PostTrainingQuantConfig(
    tuning_criterion=config.TuningCriterion(
        strategy="tpe",
    ),
)
```

### Exhaustive

#### Design

`Exhaustive` strategy is used to sequentially traverse all possible tuning
configurations in a tuning space. From the perspective of the impact on
performance, we currently only traverse all possible quantize tuning
configs. Same reason as `Bayesian`, fallback datatypes are not included for now.

#### Usage

`Exhaustive` usage is similar to `basic` but specific the `strategy` field with `exhaustive` in the `tuning_criterion` setting.


```python
from neural_compressor import config

conf = config.PostTrainingQuantConfig(
    tuning_criterion=config.TuningCriterion(
        strategy="exhaustive",
    ),
)
```

### Random

#### Design

`Random` strategy is used to randomly choose tuning configurations from the
tuning space. As with `Exhaustive` strategy, it also only considers quantize
tuning configs to generate a better-performance quantized model.

#### Usage

`Random` usage is similar to `basic` but specific the `strategy` field with `random` in the `tuning_criterion` setting.

```python
from neural_compressor import config

conf = config.PostTrainingQuantConfig(
    tuning_criterion=config.TuningCriterion(
        strategy="random",
    ),
)
```

### SigOpt

#### Design

`SigOpt` strategy is to use [SigOpt Optimization Loop](https://app.sigopt.com/docs/overview/optimization) method to accelerate and visualize the traversal of the tuning configurations from the tuning space. The metrics add accuracy as constraint and optimize for latency to improve the performance. [SigOpt Projects](https://app.sigopt.com/) can show the result of each tuning experiment.

#### Usage

Compare to `Basic`, `sigopt_api_token` and `sigopt_project_id` is necessary for `SigOpt`.`sigopt_experiment_name` is optional, the default name is `nc-tune`.

For details, [how to use sigopt strategy in neural_compressor](./sigopt_strategy.md) is available.

Note that required options `sigopt_api_token` and `sigopt_project_id`,
and the optional option `sigopt_experiment_name` should be set inside the 
`strategy_kwargs`.

```python
from neural_compressor import config

conf = config.PostTrainingQuantConfig(
    tuning_criterion=config.TuningCriterion(
        strategy="sigopt",
        strategy_kwargs={
          "sigopt_api_token": "YOUR-ACCOUNT-API-TOKEN",
          "sigopt_project_id": "PROJECT-ID",
          "sigopt_experiment_name": "nc-tune",
        },
    ),
)
```

## Customize a New Tuning Strategy

Intel® Neural Compressor supports new strategy extension by implementing a subclass of `TuneStrategy` class in neural_compressor.strategy package
 and registering this strategy by `strategy_registry` decorator.

For example, user can implement a `Abc` strategy like below:

```python
@strategy_registry
class AbcTuneStrategy(TuneStrategy):
    def __init__(self, model, conf, q_dataloader, q_func=None,
                 eval_dataloader=None, eval_func=None, dicts=None):
        ...

    def next_tune_cfg(self):
        # generate the next tuning config
        ...
    
    def traverse(self):
        for tune_cfg in self.next_tune_cfg():
            # do quantization
            ...

```

The `next_tune_cfg` function is used to yield the next tune configuration according to some algorithm or strategy. `TuneStrategy` base class will traverse
 all the tuning space till a quantization configuration meets pre-defined accuracy criterion.

If the traverse behavior of `TuneStrategy` base class does not meet new strategy requirement, it could re-implement `traverse` function with self own logic.
An example like this is under [TPE Strategy](../../neural_compressor/contrib/strategy/tpe.py).