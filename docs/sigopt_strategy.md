# SigOpt Strategy

[SigOpt](https://app.sigopt.com/) is available via online platform and can be used for model development and performance. [Optimization Loop](https://app.sigopt.com/docs/overview/optimization) is the backbone of using SigOpt, we can set metrics and realize the interaction between online platform and tuning configures based on this mechanism.

## Preparation

Before using `SigOpt` strategy, a SigOpt account is necessary.
- Each account has its own api token. Find your api token and then fill in the configure item `sigopt_api_token`. 
- Create a new project and write the corresponding name into the configure item `sigopt_project_id`.
- Set the name for this experiment in configure item `sigopt_experiment_id`, the default is nc-tune.

### SigOpt introduction

If you are using SigOpt products for the first time, please [sign-up](https://app.sigopt.com/signup), if not, please [login](https://app.sigopt.com/login). It is free to apply for an account. Although there are certain restrictions on the model parameters and the number of experiments created, it is sufficient for ordinary customers. If you want higher capacity, please contact support@sigopt.com.

After logging in, you can use `the token api` to connect the local code and the online platform, corresponding to the configure item `sigopt_api_token`, it can be obtained [here](https://app.sigopt.com/tokens/info).

SigOpt has two concepts: [project](https://app.sigopt.com/projects) and [experiment](https://app.sigopt.com/experiments). Create a project before experimenting, corresponding to `sigopt_project_id` and `sigopt_experiment_name`, Multiple experiments can be created in each project. After creating experiment, run through these three simple steps, in a loop:

- Receive a Suggestion from SigOpt
- Evaluate your metric
- Report an Observation to SigOpt

In Neural Compressor sigopt strategy, the metrics add accuracy as constraint and optimize for latency.

### Neural Compressor configuration

Compare to `Basic` strategy, `sigopt_api_token` is necessary for `SigOpt` strategy. Create the corresponding project name `sigopt_project_id` in the account before using the strategy.

```yaml
tuning:
  strategy:
    name: sigopt
    sigopt_api_token: YOUR-ACCOUNT-API-TOKEN
    sigopt_project_id: PROJECT-ID
    sigopt_experiment_name: nc-tune
  accuracy_criterion:
    relative:  0.01
  exit_policy:
    timeout: 0
  random_seed: 9527

```


## Performance

### Benefit for Sigopt strategy

- Metric based the SigOpt is better than self-define and easy to use. you can read details from [here](https://app.sigopt.com/docs/overview/metric_constraints). 
- Through the token api, the results of each experiment are recorded in your account. You can use the SigOpt data analysis function to analyze the results, such as drawing a chart, calculating F1 score, etc.

### Performance comparison of different strategies

MobileNet_v1 tensorflow

|strategy|FP32 baseline|int8 accuracy|int8 duration(s)|
|--------|-------------|-------------|----------------|
|  basic |  0.8266     | 0.8372      |  88.2132       |
| sigopt |  0.8266     | 0.8372      |  83.7495       |

ResNet50_v1 tensorflow

|strategy|FP32 baseline|int8 accuracy|int8 duration(s)|
|--------|-------------|-------------|----------------|
|  basic |  0.8299     | 0.8294      |  85.0837       |
| sigopt |  0.8299     | 0.8291      |  83.4469       |

