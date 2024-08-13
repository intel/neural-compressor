Pythonic Style Access for Configurations
====

1. [Introduction](#introduction)
2. [Supported Feature Matrix](#supported-feature-matrix)
3. [Get Started with Pythonic API for Configurations](#get-started-with-pythonic-api-for-configurations)

## Introduction
To meet the variety of needs arising from various circumstances, INC now provides a
pythonic style access - Pythonic API - for same purpose of either user or framework configurations.

The Pythonic API for Configuration allows users to specify configurations
directly in their python codes without referring to 
a separate YAML file. While we support both simultaneously, 
the Pythonic API for Configurations has several advantages over YAML files, 
which one can tell from usages in the context below. Hence, we recommend 
users to use the Pythonic API for Configurations moving forward. 

## Supported Feature Matrix

### Pythonic API for User Configurations
| Optimization Techniques | Pythonic API |
|-------------------------|:------------:|
| Quantization            |   &#10004;   |
| Pruning                 |   &#10004;   |
| Distillation            |   &#10004;   |
| NAS                     |   &#10004;   |
### Pythonic API for Framework Configurations

| Framework  | Pythonic API |
|------------|:------------:|
| TensorFlow |   &#10004;   |
| PyTorch    |   &#10004;   |
| ONNX       |   &#10004;   |
| MXNet      |   &#10004;   |

## Get Started with Pythonic API for Configurations

### Pythonic API for User Configurations
Now, let's go through the Pythonic API for Configurations in the order of
sections similar as in user YAML files. 

#### Quantization

To specify quantization configurations, users can use the following 
Pythonic API step by step. 

* First, load the ***config*** module
```python
from neural_compressor import config
```
* Next, assign values to the attributes of *config.quantization* to use specific configurations, and pass the config to *Quantization* API.
```python
config.quantization.inputs = ["image"]  # list of str
config.quantization.outputs = ["out"]  # list of str
config.quantization.backend = "onnxrt_integerops"  # support tensorflow, tensorflow_itex, pytorch, pytorch_ipex, pytorch_fx, onnxrt_qlinearops, onnxrt_integerops, onnxrt_qdq, onnxrt_qoperator, mxnet
config.quantization.approach = "post_training_dynamic_quant"  # support post_training_static_quant, post_training_dynamic_quant, quant_aware_training
config.quantization.device = "cpu"  # support cpu, gpu
config.quantization.op_type_dict = {"Conv": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}}}  # dict
config.quantization.strategy = "mse"  # support basic, mse, bayesian, random, exhaustive
config.quantization.objective = "accuracy"  # support performance, accuracy, modelsize, footprint
config.quantization.timeout = 100  # int, default is 0
config.quantization.accuracy_criterion.relative = 0.5  # float, default is 0.01
config.quantization.reduce_range = (
    False  # bool. default value depends on hardware, True if cpu supports VNNI instruction, otherwise is False
)
config.quantization.use_bf16 = False  # bool
from neural_compressor.experimental import Quantization

quantizer = Quantization(config)
```

#### Distillation
To specify distillation configurations, users can assign values to 
the corresponding attributes.
```python
from neural_compressor import config

config.distillation.optimizer = {"SGD": {"learning_rate": 0.0001}}

from neural_compressor.experimental import Distillation

distiller = Distillation(config)
```
#### Pruning
To specify pruning configurations, users can assign values to the corresponding attributes. 
```python
from neural_compressor import config

config.pruning.weight_compression.initial_sparsity = 0.0
config.pruning.weight_compression.target_sparsity = 0.9
config.pruning.weight_compression.max_sparsity_ratio_per_layer = 0.98
config.pruning.weight_compression.prune_type = "basic_magnitude"
config.pruning.weight_compression.start_epoch = 0
config.pruning.weight_compression.end_epoch = 3
config.pruning.weight_compression.start_step = 0
config.pruning.weight_compression.end_step = 0
config.pruning.weight_compression.update_frequency = 1.0
config.pruning.weight_compression.update_frequency_on_step = 1
config.pruning.weight_compression.prune_domain = "global"
config.pruning.weight_compression.pattern = "tile_pattern_1x1"

from neural_compressor.experimental import Pruning

prune = Pruning(config)
```
#### NAS
To specify nas configurations, users can assign values to the
corresponding attributes.

```python
from neural_compressor import config

config.nas.approach = "dynas"
from neural_compressor.experimental import NAS

nas = NAS(config)
```


#### Benchmark
To specify benchmark configurations, users can assign values to the
corresponding attributes.
```python
from neural_compressor import config

config.benchmark.warmup = 10
config.benchmark.iteration = 10
config.benchmark.cores_per_instance = 10
config.benchmark.num_of_instance = 10
config.benchmark.inter_num_of_threads = 10
config.benchmark.intra_num_of_threads = 10

from neural_compressor.experimental import Benchmark

benchmark = Benchmark(config)
```
### Pythonic API for Framework Configurations
Now, let's go through the Pythonic API for Configurations in setting up similar framework
capabilities as in YAML files. Users can specify a framework's (eg. ONNX Runtime) capability by
assigning values to corresponding attributes. 

```python
config.onnxruntime.precisions = ["int8", "uint8"]
config.onnxruntime.graph_optimization_level = "DISABLE_ALL"  # only onnxruntime has graph_optimization_level attribute
```
