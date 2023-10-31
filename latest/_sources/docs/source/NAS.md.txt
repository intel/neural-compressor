# Neural Architecture Search

1. [Introduction](#introduction)

    1.1. [Basic NAS](#basic-nas)

    1.2. [Dynamic NAS](#dynamic-nas)

2. [NAS Support Matrix](#nas-support-matrix)
3. [Get Started with NAS API](#get-started-with-nas-api)

    3.1. [Basic Usage](#basic-usage)

    3.2. [Advanced Usage (Custom NAS)](#advanced-usage-custom-nas)

4. [Examples](#examples)

## Introduction
Neural Architecture Search (NAS) is the process of automating the design of artificial neural networks (ANN) architecture. NAS has been used to design networks that are on par with or outperform hand-designed architectures. Intel® Neural Compressor has supported two different NAS methods: Basic NAS and Dynamic NAS.

### Basic NAS
Our Basic NAS method leverages a specific search algorithm from built-in search algorithms (grid search, random search, and Bayesian optimization are supported in Intel® Neural Compressor now) or user-defined search algorithms to propose the model architecture based on the given search space, then performs the train evaluation process to evaluate the potential of the proposed model architecture, after several iterations of such procedure, best-performing model architectures which lie in Pareto front will be returned.

### Dynamic NAS
Dynamic Neural Architecture Search (DyNAS) is a super-network-based NAS approach that uses the metric predictors for predicting the metrics of the model architecture, it is >4x more sample efficient than typical one-shot predictor-based NAS approaches.  
<br>
The flow of the DyNAS approach is shown in the following figure. In the first phase of the search, a small population of sub-networks is randomly sampled from the super-network and evaluated (validation measurement) to provide the initial training set for the inner predictor loop. After the predictors are trained, a multi-objective evolutionary search is performed in the predictor objective space. After this extensive search is performed, the best-performing sub-network configurations are selected to be the next iteration's validation population. The cycle continues until the search concludes when the user-defined evaluation count is met.  
<br>
![DyNAS Workflow](./imgs/dynas.png)

## NAS Support Matrix

|NAS Algorithm     |PyTorch   |TensorFlow |
|------------------|:--------:|:---------:|
|Basic NAS         |&#10004;  |Not supported yet|
|Dynamic NAS       |&#10004;  |Not supported yet|

## Get Started with NAS API

### Basic Usage

#### 1. Python code + YAML

Simplest launcher code if NAS configuration is defined in user-defined yaml.

```python
from neural_compressor.experimental import NAS

agent = NAS("/path/to/user/yaml")
results = agent.search()
```

#### 2. Python code only

NAS class also support `NASConfig` class as it's argument.

```python
from neural_compressor.conf.config import NASConfig
from neural_compressor.experimental import NAS

config = NASConfig(approach="dynas", search_algorithm="nsga2")
config.dynas.supernet = "ofa_mbv3_d234_e346_k357_w1.2"
config.dynas.metrics = ["acc", "macs"]
config.dynas.population = 50
config.dynas.num_evals = 250
config.dynas.results_csv_path = "search_results.csv"
config.dynas.batch_size = 64
config.dynas.dataset_path = "/datasets/imagenet-ilsvrc2012"  # example
agent = NAS(config)
results = agent.search()
```

### Advanced Usage (Custom NAS)

Intel® Neural Compressor NAS API is defined under `neural_compressor.experimental.nas`, which takes a user defined yaml file or a [NASConfig](../../neural_compressor/conf/config.py#NASConfig) object as input. The user defined yaml or the [NASConfig](../../neural_compressor/conf/config.py#NASConfig) object defines necessary configuration of the NAS process. The [NAS](../../neural_compressor/experimental/nas/nas.py#NAS) class aims to create an object according to the defined NAS approach in the configuration, please note this NAS approach should be registered in the Intel® Neural Compressor.

Currently, Intel® Neural Compressor supported two built-in NAS methods: [Basic NAS](../../neural_compressor/experimental/nas/basic_nas.py#BasicNAS) and [Dynamic NAS](../../neural_compressor/experimental/nas/dynas.py#DyNAS). Both methods are inherited from a base class called [NASBase](../../neural_compressor/experimental/nas/nas.py#NASBase). User can also customize their own NAS approach in Intel® Neural Compressor just by decorating their NAS approach class with function [nas_registry](../../neural_compressor/experimental/nas/nas_utils.py#nas_registry) as well as following the API in [NASBase](../../neural_compressor/experimental/nas/nas.py#NASBase), like the way used in the two built-in NAS methods.

## Examples

Following examples are supported in Intel® Neural Compressor:

- DyNAS MobileNetV3 supernet Example:
  - [DyNAS MobileNetV3 supernet Example](../../examples/notebook/dynas/MobileNetV3_Supernet_NAS.ipynb): DyNAS with MobileNetV3 supernet on ImageNet dataset.
- DyNAS Transformer LT supernet Example:
  - [DyNAS Transformer LT supernet Example](../../examples/notebook/dynas/Transformer_LT_Supernet_NAS.ipynb): DyNAS with Transformer LT supernet on WMT En-De dataset.
