# Neural Architecture Search

## Introduction
Neural Architecture Search (NAS) is the process of automating the design of artificial neural networks (ANN) architecture. NAS has been used to design networks that are on par or outperform hand-designed architectures. Intel® Neural Compressor has supported two different NAS methods: Basic NAS and Dynamic NAS.

## NAS API

### Basic Usage

#### 1. Python code + YAML

Simplest launcher code if NAS configuration is defined in user-defined yaml.

```python
from neural_compressor.experimental import NAS
agent = NAS('/path/to/user/yaml')
results = agent.search()
```

The user-defined yaml follows below syntax, note `dynas` section is optional (only required for 'dynas' approach).

```yaml
nas:
  approach: 'dynas'
  search:
    search_algorithm: 'nsga2'
    seed: 42
  dynas:
    supernet: 'ofa_mbv3_d234_e346_k357_w1.2'
    metrics: ['acc', 'macs']
    population: 50
    num_evals: 250
    results_csv_path: 'search_results.csv'
    batch_size: 64
    dataset_path: '/datasets/imagenet-ilsvrc2012' #example
```

#### 2. Python code only

NAS class also support `NASConfig` class as it's argument.

```python
from neural_compressor.conf.config import NASConfig
from neural_compressor.experimental import NAS
config = NASConfig(approach='dynas', search_algorithm='nsga2')
config.dynas.supernet = 'ofa_mbv3_d234_e346_k357_w1.2'
config.dynas.metrics = ['acc', 'macs']
config.dynas.population = 50
config.dynas.num_evals = 250
config.dynas.results_csv_path = 'search_results.csv'
config.dynas.batch_size = 64
config.dynas.dataset_path = '/datasets/imagenet-ilsvrc2012' #example
agent = NAS(config)
results = agent.search()
```

### Advanced Usage (Custom NAS)

Intel® Neural Compressor NAS API is defined under `neural_compressor.experimental.NAS`, which takes a user defined yaml file or a `NASConfig` object as input. The user defined yaml or the `NASConfig` object defines necessary configuration NAS process. The NAS class below aims to create an object according to the defined NAS approach in the configuration, please note this NAS approach should be registered in the Intel® Neural Compressor.

```python
# nas.py in neural_compressor/experimental/nas
class NAS():
    def __new__(self, conf_fname_or_obj, *args, **kwargs):
        # Create an object according to the defined NAS approach in the configuration.
        ...
```

Currently, Intel® Neural Compressor supported two built-in NAS methods: Basic NAS and Dynamic NAS. Both methods inherit from a base class called `NASBase`, its interface is shown below. User can also customize their own NAS approach in Intel® Neural Compressor just by decorating their NAS approach class with function `nas_registry` as well as following the API in `NASBase`, like the way used in two built-in NAS methods.

```python
# nas.py in neural_compressor/experimental/nas
class NASBase(object):
    def __init__(self, search_space=None, model_builder=None):
        # NAS configuration initialization.
        ...

    def select_model_arch(self):
        # Propose architecture of the model based on search algorithm for next search iteration.
        ...

    def search(self, res_save_path=None):
        # NAS search process.
        ...    

    def estimate(self, model): # pragma: no cover
        # Estimate performance of the model. Depends on specific NAS algorithm.
        ...

    def load_search_results(self, path):
        # Load search results from existing file.
        ...

    def dump_search_results(self, path):
        # Save the search results.
        ...

    def find_best_model_archs(self):
        # Find best performing model architectures in the pareto front.
        ...

    @search_space.setter
    def search_space(self, search_space):
        # Search space should be defined by user.
        ...

    @search_algorithm.setter
    def search_algorithm(self, search_algorithm):
        # Search algorithm used in the NAS process.
        ...

    @model_builder.setter
    def model_builder(self, model_builder):
        # A callable object that returns a model instance based on the model architecture input.
        ...
```

#### Basic NAS
Our Basic NAS method leverages a specific search algorithm from built-in search algorithms (grid search, random search and bayesian optimization are supported in Intel® Neural Compressor now) or user defined search algorithms to propose the model architecture based on the given search space, then perform the train evaluation process to evaluate the potential of the proposed model architecture, after several iterations of such procedure, best performing model architectures which lie in pareto front will be returned. This class is registered to the Intel® Neural Compressor as a built-in NAS method through a decorator `nas_registry`, its interface is shown below.

```python
# basic_nas.py in neural_compressor/experimental/nas
@nas_registry("Basic")
class BasicNAS(NASBase, Component):
    def __init__(self, conf_fname_or_obj, search_space=None, model_builder=None):
        # NAS configuration initialization.
        ...

    def estimate(self, model):
        # Estimate performance of the model with train and evaluation process.
        ...
```

#### Dynamic NAS
Dynamic Neural Architecture Search (DyNAS) is a super-network-based NAS approach which use the metric predictors for predicting the metrics of the model architecture, it is >4x more sample efficient than typical one-shot predictor-based NAS approaches.
<br>
The flow of the DyNAS approach is shown in the following figure. In the first phase of the search, a small population of sub-networks are randomly sampled from the super-network and evaluated (validation measurement) to provide the initial training set for the inner predictor loop. After the predictors are trained, a multi-objective evolutionary search is performed in the predictor objective space. After this extensive search is performed, the best performing sub-network configurations are selected to be the next iteration's validation population. The cycle continues until the search concludes when the user defined evaluation count is met.
<br>
![DyNAS Workflow](./imgs/dynas.png)
<br>
This class is also registered to the Intel® Neural Compressor as a built-in NAS method through a decorator `nas_registry`, its interface is shown below.

```python
# dynas.py in neural_compressor/experimental/nas
@nas_registry("DyNAS")
class DyNAS(NASBase):
    def __init__(self, conf_fname_or_obj):
        # NAS configuration initialization.
        ...

    def estimate(self, individual):
        # Estimate performance of the model.
        ...

    def search(self):
        # DyNAS search process.
        ...

    def create_acc_predictor(self):
        # Create accuracy predictor.
        ...

    def create_macs_predictor(self):
        # Create macs predictor.
        ...

    def create_latency_predictor(self):
        # Create latency predictor.
        ...
```

## Examples

Following examples are supported in Intel® Neural Compressor:

- DyNAS MobileNetV3 supernet Example:
  - [DyNAS MobileNetV3 supernet Example](../examples/notebook/dynas/MobileNetV3_Supernet_NAS.ipynb): DyNAS with MobileNetV3 supernet on ImageNet dataset.
