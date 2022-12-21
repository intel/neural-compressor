SigOpt Strategy
============

1. [Introduction](#introduction)

    1.1 [Preparation](#preparation)
    
    1.2 [SigOpt Platform](#sigopt-platform)
    
    1.3  [Neural Compressor Configuration](#neural-compressor-configuration)

2. [Performance](#performance)
    
    2.1 [Benefit of SigOpt Strategy](#benefit-of-sigopt-strategy)

    2.2 [Performance Comparison of Different Strategies](#performance-comparison-of-different-strategies)

## Introduction

[SigOpt](https://app.sigopt.com/) is an online model development platform that makes it easy to track runs, visualize training, and scale hyperparameter optimization for any type of model. [Optimization Loop](https://app.sigopt.com/docs/overview/optimization) is the backbone of using SigOpt. We can set metrics and realize the interaction between the online platform and tuning configures based on this mechanism.

### Preparation

Before using the `SigOpt` strategy, a SigOpt account is necessary.
- Each account has its own API token. Find your API token and then fill in the field of `sigopt_api_token`. 
- Create a new project and write the corresponding name into the field of `sigopt_project_id`.
- Set the name for this experiment in field of `sigopt_experiment_id`, the default is nc-tune.

### SigOpt Platform 

If you are using the SigOpt products for the first time, please [sign-up](https://app.sigopt.com/signup), if not, please [login](https://app.sigopt.com/login). It is free to apply for an account. Although there are certain restrictions on the model parameters and the number of experiments created, it is sufficient for ordinary customers. If you want higher capacity, please contact support@sigopt.com.

After logging in, you can use `the token api` to connect the local code and the online platform, corresponding to `sigopt_api_token`. It can be obtained [here](https://app.sigopt.com/tokens/info).

SigOpt has two concepts: [project](https://app.sigopt.com/projects) and [experiment](https://app.sigopt.com/experiments). Create a project before experimenting, corresponding to `sigopt_project_id` and `sigopt_experiment_name`. Multiple experiments can be created on each project. After creating the experiment, run through these three simple steps, in a loop:

- Receive a Suggestion from SigOpt
- Evaluate your metrics
- Report an Observation to SigOpt

In our build-in sigopt strategy, the metrics add accuracy as a constraint and optimize for latency.

### Neural Compressor Configuration

Compare to `Basic` strategy, `sigopt_api_token` is necessary for `SigOpt` strategy. Create the corresponding project name `sigopt_project_id` in the account before using the strategy.

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

## Performance

### Benefit of SigOpt Strategy

- Metric based SigOpt is better than self-defining and easy to use. You can read the details [here](https://app.sigopt.com/docs/overview/metric_constraints). 
- Through the token api, the results of each experiment are recorded in your account. You can use the SigOpt data analysis function to analyze the results, such as drawing a chart, calculating the F1 score, etc.

### Performance Comparison of Different Strategies

- MobileNet_v1(tensorflow)

    |strategy|FP32 baseline|int8 accuracy|int8 duration(s)|
    |--------|-------------|-------------|----------------|
    |  basic |  0.8266     | 0.8372      |  88.2132       |
    | sigopt |  0.8266     | 0.8372      |  83.7495       |

- ResNet50_v1(tensorflow)

    |strategy|FP32 baseline|int8 accuracy|int8 duration(s)|
    |--------|-------------|-------------|----------------|
    |  basic |  0.8299     | 0.8294      |  85.0837       |
    | sigopt |  0.8299     | 0.8291      |  83.4469       |

