User YAML Configuration Files
=====
1. [Introduction](#introduction)
2. [Supported Feature Matrix](#supported-feature-matrix)
3. [Get Started with User YAML Files](#get-started-with-user-yaml-files)


## Introduction

Intel® Neural Compressor uses YAML files for quick 
and user-friendly configurations. There are two types of YAML files - 
user YAML files and framework YAML files, which are used in 
running user cases and setting up framework capabilities, respectively.

First, let's take a look at a user YAML file, It defines the model, tuning
strategies, tuning calibrations and evaluations, and performance benchmarking
of the passing model vs. original model.

## Supported Feature Matrix

| Optimization Techniques | YAML Configuration Files |
|-------------------------|:------------------------:|
| Quantization            |         &#10004;         |
| Pruning                 |         &#10004;         |
| Distillation            |         &#10004;         |


## Get started with User YAML Files


A complete user YAML file is organized logically into several sections: 

* ***model***: The model specifications define a user model's name, inputs, outputs and framework.
    

```yaml
model:                                               # mandatory. used to specify model specific information.
  name: mobilenet_v1 
  framework: tensorflow                              # mandatory. supported values are tensorflow, pytorch, pytorch_ipex, onnxrt_integer, onnxrt_qlinear or mxnet; allow new framework backend extension.
  inputs: image_tensor                               # optional. inputs field is only required in tensorflow.
  outputs: num_detections,detection_boxes,detection_scores,detection_classes # optional. outputs field is only required in tensorflow.
```
* ***quantization***: The quantization specifications define quantization tuning space and related calibrations. To calibrate, users can 
specify *sampling_size* (optional) and use the subsection *dataloader* to specify
the dataset location using *root* and transformation using *transform*. To 
implement tuning space constraints, users can use the subsection *model_wise* and *op_wise* for specific configurations.
 
```yaml
quantization:                                        # optional. tuning constraints on model-wise for advance user to reduce tuning space.
  calibration:
    sampling_size: 20                                # optional. default value is 100. used to set how many samples should be used in calibration.
    dataloader:
      dataset:
        ImageRecord:
          root: /path/to/imagenet/                   # NOTE: modify to calibration dataset location if needed
      transform:
        BilinearImagenet: 
          height: 224
          width: 224
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
           'activation':  {'dtype': ['uint8', 'fp32'], 
                           'algorithm': ['minmax', 'kl'], 
                           'scheme':['sym']},
           'weight': {'dtype': ['int8', 'fp32'], 
                      'algorithm': ['minmax']}
         }
       }
```

* ***pruning***: The pruning specifications define pruning tuning space. To define the training behavior, uses can 
use the subsection *train* to specify the training hyper-parameters and the training dataloader. 
To define the pruning approach, users can use the subsection *approach* to specify 
pruning target, choose the type of pruning algorithm, and the way to apply it 
during training process. 

```yaml
pruning:
  train:
    dataloader:
      ... 
    epoch: 40
    optimizer:
      Adam:
        learning_rate: 1e-06
        beta_1: 0.9
        beta_2: 0.999
        epsilon: 1e-07
    criterion:
      SparseCategoricalCrossentropy:
        reduction: sum_over_batch_size
        from_logits: False
  approach:
    weight_compression:
      initial_sparsity: 0.0
      target_sparsity: 0.54
      start_epoch: 0
      end_epoch: 19
      pruners:
        - !Pruner
            start_epoch: 0
            end_epoch: 19
            prune_type: basic_magnitude
```
* ***distillation***: The distillation specifications define distillation's tuning
space. Similar to pruning, to define the training behavior, users can use the 
subsection *train* to specify the training hyper-parameters and the training 
dataloader and it is optional if users implement *train_func* and set the attribute
of distillation instance to *train_func*. For criterion, Intel® Neural Compressor provides a built-in 
knowledge distillation loss class to calculate distillation loss.
```yaml
distillation:
  train:
    start_epoch: 0
    end_epoch: 90
    iteration: 1000
    frequency: 1
    dataloader:
      ...
    optimizer:
      SGD:
        learning_rate: 0.001  
        momentum: 0.1
        nesterov: True
        weight_decay: 0.001
    criterion:
      KnowledgeDistillationLoss:
        temperature: 1.0
        loss_types: ['CE', 'CE']
        loss_weights: [0.5, 0.5]
```
* ***evaluation***: The evaluation specifications define the dataloader and metric for accuracy evaluation as well as dataloader 
and configurations for performance benchmarking. 
```yaml
evaluation:                                          # optional. required if user doesn't provide eval_func in neural_compressor.Quantization.
  accuracy:                                          
    metric:
      ...
    dataloader:
      ...
```
* ***tuning***: The tuning specifications define overall tuning targets. Users can
use *accuracy_criterion* to specify the target of accuracy loss percentage and use
*exit_policy* to specify the tuning timeout in seconds. The random
seed can be specified using *random_seed*. 

```yaml
tuning:
  accuracy_criterion:
    relative: 0.01                                  # the tuning target of accuracy loss percentage: 1%
    higher_is_better: True
  exit_policy:
    timeout: 0                                      # tuning timeout (seconds), 0 means early stop
  random_seed: 9527                                 # random seed
```

