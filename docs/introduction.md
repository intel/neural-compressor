Introduction
=========================================

Intel® Low Precision Optimization Tool is an open source python library to help users to fast deploy low-precision inference solution on popular DL frameworks including TensorFlow, PyTorch, MxNet etc. It automatically optimizes low-precision recipes for deep learning models to achieve optimal product objectives like inference performance and memory usage with expected accuracy criteria.

# Infrastructure

<div align="left">
  <img src="imgs/infrastructure.jpg" width="700px" />
</div>

### Three level APIs

1. User API

   User API is intented to provide best out-of-box experiences and unify the low precision quantization workflow cross multiple DL frameworks.

   ```
   class Quantization(object):
       def __init__(self, conf_fname):
           ...
       def __call__(self, model, q_dataloader, q_func=None, eval_dataloader=None, eval_func=None):
           ...
   ```

   The tuning config and model-specific information are controlled by user config yaml file. As for the format of yaml file, please refer to [ptq.yaml](../ilit/template/ptq.yaml) or [qat.yaml](../ilit/template/qat.yaml) or [pruning.yaml](../ilit/template/pruning.yaml).

   Intel® Low Precision Optimization Tool supports three usages:

   a) Fully yaml configuration: 
      This usage is designed for minimal code changes when integrating with Intel® Low Precision Optimization Tool. All calibration and evaluation process is constructed by ilit upon yaml configuration. including dataloaders used in calibration and evaluation
      phases and quantization tuning settings. For this usage, only model parameter is mandotory.

   
   b) User specifies dataloaders: 
      The second usage is designed for concise yaml configuration and constructing calibration and evaluation dataloaders by code. ilit provides built-in dataloaders and evaluators, user just need provide a dataset implemented __iter__ or __getitem__ methods to tuner.dataloader() function. User specifies dataloaders used in calibration and evaluation phase by code.
      The tool provides built-in dataloaders and evaluators, user just need provide a dataset implemented __iter__ or
      __getitem__ methods and invoke dataloader() with dataset as input parameter before calling Quantization().
   
      After that, User specifies fp32 "model", calibration dataset "q_dataloader" and evaluation dataset "eval_dataloader".
      The calibrated and quantized model is evaluated with "eval_dataloader" with evaluation metrics specified
      in the configuration file. The evaluation tells the tuner whether the quantized model meets
      the accuracy criteria. If not, the tuner starts a new calibration and tuning flow.
   
      For this usage, model, q_dataloader and eval_dataloader parameters are mandotory.
   
   c) User specifed eval_func: 
      The third usage is designed for ease of tuning enablement for models with custom metric evaluation or metrics not supported by Intel® Low Precision Optimization Tool yet. Currently this usage model works for object detection and NLP networks.
      User specifies dataloaders used in calibration phase by code. This usage is quite similar with b), just user specifies a custom "eval_func" which encapsulates the evaluation dataset by itself.
      The calibrated and quantized model is evaluated with "eval_func". The "eval_func" tells the
      tuner whether the quantized model meets the accuracy criteria. If not, the Tuner starts a new
      calibration and tuning flow.
   
      For this usage, model, q_dataloader and eval_func parameters are mandotory

2. Framework Adaptation API

   Framework adaptation layer abstracts out the API differences of various DL frameworks needed for supporting low-precision quantization workflow and provides a unified API for auto-tuning engine to use. The abstracted functionalities include quantization configurations, quantization capabilities, calibration, quantization-aware training, graph transformation for quantization, data loader and metric evaluation, and tensor inspection.

3. Extension API

   Intel® Low Precision Optimization Tool is designed to be highly extensible. New tuning strategies can be added by inheriting "Strategy" class. New frameworks can be added by inheriting "Adaptor" class. New metrics can be added by inheriting "Metric" class. New tuning objectives can be added by inheriting "Objective" class.

# Workflow

<div align="left">
  <img src="imgs/workflow.jpg" width="700px" />
</div>

# Strategies

### Basic Strategy

This strategy is Intel® Low Precision Optimization Tool default tuning strategy, which does model-wise tuning by adjusting gloabl tuning parameters, such as calibration related parameters, kl or minmax algo, quantization related parameters, symmetric or asymmetric, per_channel or per_tensor. If the model-wise tuning result does not meet accuracy goal, this strategy will attempt to do op-wise fallback from bottom to top to prioritize which fallback op has biggest impact on final accuracy, and then do incremental fallback till achieving the accuracy goal.

### Bayesian Strategy

Bayesian optimization is a sequential design strategy for global optimization of black-box functions. The strategy refers to the Bayesian optimization package [bayesian-optimization](https://github.com/fmfn/BayesianOptimization) and changes it to a discrete version that complies with the strategy standard of Intel® Low Precision Optimization Tool. It uses Gaussian Processes to define the prior/posterior distribution over the black-box function, and then finds the tuning configuration that maximizes the expected improvement.

### MSE Strategy

This strategy is very similar to the basic strategy. It needs to get the tensors for each Operator of raw FP32 models and the quantized model based on best model-wise tuning configuration. And then calculate the MSE (Mean Squared Error) for each operator, sort those operators according to the MSE value, finally do the op-wise fallback in this order.

### Random Strategy

This strategy is used to randomly choose tuning configuration from the tuning space.

### Exhaustive Strategy

This strategy is used to sequentially traverse all the possible tuning configurations in tuning space.

# Objectives

Intel® Low Precision Optimization Tool supports below 3 build-in objectives. All objectives are optimized and driven by accuracy metrics.

### 1. Performance

This objective targets best performance of quantized model. It is default objective.

### 2. Memory Footprint

This objective targets minimal memory usage of quantized model.

### 3. Model Size

This objective targets the weight size of quantized model.

# Metrics

Intel® Low Precision Optimization Tool supports 3 built-in metrics, Topk, F1 and CocoMAP. The metric is easily extensible via inheriting Metric class.
