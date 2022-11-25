Objective
=======

1. [Introduction](#introduction)

    1.1. [Single Objective](#single-objective)

    1.2. [Multiple Objectives](#multiple-objectives)

2. [Objective Support Matrix](#objective-support-matrix)

3. [Get Start with Objective API](#get-start-with-objective-api)

    3.1. [Config Single Objective](#config-single-objective)

    3.2. [Config Multiple Objectives](#config-multiple-objectives)

    3.3. [Config Custom Objective](#config-custom-objective)

4. [Example](#example)

## Introduction

In terms of evaluating the status of a specific model during tuning, we should have general objectives. Intel® Neural Compressor Objective supports code-free configuration through a yaml file. With built-in objectives, users can compress models with different objectives easily. In special cases, users can also register their own objective classes.

### Single Objective

The objective supported by Intel® Neural Compressor is driven by accuracy. If users want to evaluate a specific model with other objectives, they can realize it with `objective` in a yaml file. Default value for `objective` is `performance`, and the other values are `modelsize` and `footprint`.

### Multiple Objectives

In some cases, users want to use more than one objective to evaluate the status of a specific model and they can realize it with `multi_objectives` in a yaml file. Currently `multi_objectives` supports built-in objectives.

If users use `multi_objectives` to evaluate the status of the model during tuning, Neural Compressor will return a model with the best score of `multi_objectives` and meeting `accuracy_criterion` after tuning ending.

Intel® Neural Compressor will normalize the results of each objective to calculate the final weighted multi_objective score.


## Objective Support Matrix

Built-in objectives support list:

| Objective    | Usage                                                    |
| :------      | :------                                                  |
| accuracy     | Evaluate the accuracy                                    |
| performance  | Evaluate the inference time                              |
| footprint    | Evaluate the peak size of memory blocks during inference |
| modelsize    | Evaluate the model size                                  |

## Get Start with Objective API

### Config Single Objective

Users can specify a built-in objective in a yaml file as shown below:

```yaml
tuning:
  objective: performance
```

### Config Multiple Objectives

Users can specify built-in multiple objectives in a yaml file as shown below:

```yaml
tuning:
  multi_objectives:
    objective: [accuracy, performance]
    higher_is_better: [True, False]
    weight: [0.8, 0.2] # default is to calculate the average value of objectives
```

### Config Custom Objective

Users can also register their own objective. Look at `performance` as an example:

```python
class Performance(Objective):
    representation = 'duration (seconds)'

    def start(self):
        self.start_time = time.time()
    def end(self):
        self.duration = time.time() - self.start_time
        assert self.duration >= 0, 'please use start() before end()'
        self._result_list.append(self.duration)
```

After defining their own object, users should pass it to quantizer as below:

```python
from neural_compressor.objective import Objective
from neural_compressor.experimental import Quantization

class CustomObj(Objective):
    representation = 'CustomObj'
    def __init__(self):
        super().__init__()
        # init code here

    def start(self):
        # do needed operators before inference

    def end(self):
        # do needed operators after the end of inference
        # add status value to self._result_list
        self._result_list.append(val)

quantizer = Quantization(yaml_file)
quantizer.objective = CustomObj()
quantizer.model = model
q_model = quantizer.fit()
```
## Example
Refer to [example](../neural_compressor/template/ptq.yaml) as an example.
