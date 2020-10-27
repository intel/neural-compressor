# Introduction
Intel® Low Precision Optimization Tool aims to help users to fast deploy low-precision inference solution on popular DL frameworks including TensorFlow, Pytorch, MxNet etc. With built-in strategies, it will automatically optimized low-precision recipes for deep learning models to achieve optimal product objectives like inference performance and memory usage with expected accuracy criteria. For now, it support `Basic`, `Bayesian`, `Exhaustive`, `MSE`, `Random` and `TPE` strategies. And the `Basic` strategy is default one.

# Strategy Design
Strategies need to generate the next quantization configuration according to itself logic and last time quantization result. So the function of strategies can be shown in below graph.

<div align="left">
  <img src="imgs/strategy.png" width="700px" />
</div>

Strategies need two sides information. One side comes from adator layer, user pass the framework specific model to initial the quantizator, then strategies will call the `self.adaptor.query_fw_capability(model)` to get the framework and model specific quantization capabilities. On the other hand, strategy will merge the model specific configurations in `yaml` configuration file to filter some capability in first step to generate the tuning space. And then, each strategy will generate the quantiztion config according to itself location and logic with tuning strategy configurations in `yaml` configuration file. All of strategies will finish the tuning processing if the `timeout` or `max_trails` has been reached. The default value of `timeout` is 0, if so, the tuning phase will early stop when the `accuracy` has met the criteria.

# Configurations
The detail configuration templates can be found in [`here`](ilit/template).
### Model specific configurations
For model specific configurations, users can set the quantization approach, and for post training static quantization, we also can set calibration and quantization related parameters for model-wise and op-wise.
```yaml
quantization:                                        # optional. tuning constraints on model-wise for advance user to reduce tuning space.
  approach: post_training_static_quant               # optional. default value is post_training_static_quant.
  calibration:
    sampling_size: 1000, 2000                        # optional. default value is the size of whole dataset. used to set how many portions of calibration dataset is used. exclusive with iterations field.
    dataloader:                                      # optional. if not specified, user need construct a q_dataloader in code for ilit.Quantization.
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

### Strategy tuning part related configurations
In strategy tuning tuning part related configurations, user can choose the the specific tuning strategy and set the accuracy criterion and optimization objective for the tuning. And also can set the stop condition for the tuning by change the `exit_policy`.
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
# How to customize a new strategy
USers can based on the basic `TuneStrategy` class to enable a new strategy with a new `self.next_tune_cfg()` function implement. If the new strategy need more information, user can try to override the `self.traverse()` in the new strategy, such as `TPE` strategy. 

Basic
=============================================
## Design
Basic strategy is design for most of models to do the quantization. It can be divided to three steps. Firstly, `Basic` strategy will try all of model wise tuning configs and get the best quantized model. If all of the model wise tuning configs can't meet the accuracy loss criteria, then it will go to second step. In this step, do `OP` high precision (`FP32`, `BF16` ...) fallback one-by-one based on the best model-wise tuning config, and record the impact of each `OP` on accuracy and sort accordingly. In the finnal step, strategy will try to incrementally  fallback multipul `OP` to high-precision according to the sorted `OP` list generated in step two till achieving the accuracy goal. 

## Usage
`Basic` strategy is the default strategy, so we can use it by default if we don't add the `strategy` filed in our `yaml` configuration file. For example, the classical setting in congiguration file as below.
```yaml
tuning:
  accuracy_criterion:
    relative:  0.01                                  
  exit_policy:
    timeout: 0                                       
  random_seed: 9527  
```

Bayesian
=============================================
## Design
Bayesian optimization is a sequential design strategy for global optimization of black-box functions. The strategy refers to the [Bayesian optimization](https://github.com/fmfn/BayesianOptimization) package bayesian-optimization and changes it to a discrete version that complies with the strategy standard of Intel® Low Precision Optimization Tool. It uses Gaussian Processes to define the prior/posterior distribution over the black-box function with the tuning history, and then finds the tuning configuration that maximizes the expected improvement. For now bayesian strategy just tune op-wise quantize configs, it don't included fallback datatype config. If it add fallback datatype config into the param space of bayesian, it will generate a full FP32 model finally, because in generally, fallback datatype has better acc.  At the end `[DEBUG] Tuning config was evaluated, skip!` will be printed endlessly. Because the param space very small for this tuning and can't get good acc result. If we change the timeout from 0 to a integrate, it can be ended after reach the timeout.

## Usage
If we want to use `Bayesian` strategy, we need to set the `timeout` or `max_trials` as non-zore value as below example. Because sometimes the param space for `baysian` are very small and can't reach the accuracy goal. In this case, the tuning part will hang on. If we set the log level to `debug` by `LOGLEVEL=DEBUG` in the enviroment, `[DEBUG] Tuning config was evaluated, skip!` will be printed endlessly. 

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

MSE
=============================================
## Design
`MSE` and `Basic` has similar ideas. The mainly defierence of those two strategies is in the step 2, which to generate a sorted op lists. In `MSE` strategy step 2, it needs to get the tensors for each Operator of raw FP32 models and the quantized model based on best model-wise tuning configuration. And then calculate the MSE (Mean Squared Error) for each operator, sort those operators according to the MSE value, finally do the op-wise fallback in this order.

## Usage
Similar with `Basic` but need specific the strategy name to `mse`.
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

TPE
=============================================
## Design
Sequential model-based optimization methods (SMBO) are a formalization of Bayesian optimization. The sequential refers to running trials one after another, each time trying better hyperparameters by applying Bayesian reasoning and updating a surrogate model. There are five main aspects of SMBO:

1. Domain: A domain of hyperparameters or search space
2. Objective function: An objective function which takes hyperparameters as input and outputs a score that needs to minimize (or maximize)
4. Surrogate function: The representative surrogate model of the objective function
5. Selection function: A selection criterion to evaluate which hyperparameters to choose next trial from the surrogate model
6. History: A history consisting of (score, hyperparameter) pairs used by the algorithm to update the surrogate model

There are several variants of SMBO, which differ in how to build a surrogate and
the selection criteria (steps 3–4). The TPE builds a surrogate model by
applying Bayes's rule.

>NOTE: TPE requires many iterations to converge to an optimal solution, and
it is recommended to run it for at least 200 iterations. Because every iteration
requires evaluation of a generated model, which means accuracy measurements on a
dataset and latency measurements using a benchmark, this process may take from
24 hours up to few days to complete, depending on a model.

TPE algorithm consists of multiple (the following) steps:

1. Define a domain of hyperparameter search space,
2. Create an objective function which takes in hyperparameters and outputs a score (e.g., loss, RMSE, cross-entropy) that we want to minimize,
3. Get a couple of observations (score) using randomly selected set of hyperparameters,
4. Sort the collected observations by score and divide them into two groups based on some quantile. The first group (x1) contains observations that gave the best scores and the second one (x2)  - all other observations,
5. Two densities l(x1) and g(x2) are modeled using Parzen Estimators (also known as kernel density estimators) which are a simple average of kernels centered on existing data points,
6. Draw sample hyperparameters from l(x1), evaluating them in terms of l(x1)/g(x2), and returning the set that yields the minimum value under l(x1)/g(x1) corresponding to the greatest expected improvement. These hyperparameters are then evaluated on the objective function.
7. Update the observation list in step 3
8. Repeat step 4-7 with a fixed number of trials

## Usage
`TPE`'s usage is similar with `Bayesian`.
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

Exhaustive
=============================================
## Design
This strategy is used to sequentially traverse all the possible tuning configurations in tuning space. For now, from the perspective of the impact on performance, we only traverse all possible quantize tuning configs, don't included fall back data type. 

## Usage
The usage is also similar with `Basic`.
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

Random
=============================================
## Design
This strategy is used to randomly choose tuning configuration from the tuning space. Same with `Exhaustive` strategy, it also just consider quantize tuning configs to generate a better performance quantized model.

## Usage
The usage is also similar with `Basic`.
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
