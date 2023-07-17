Objective
=======

1. [Introduction](#introduction)

    1.1. [Single Objective](#single-objective)

    1.2. [Multiple Objectives](#multiple-objectives)

2. [Objective Support Matrix](#objective-support-matrix)

3. [Get Started with Objective API](#get-start-with-objective-api)

    3.1. [Config Single Objective](#config-single-objective)

    3.2. [Config Multiple Objectives](#config-multiple-objectives)

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

## Get Started with Objective API

### Config Single Objective

Users can specify a built-in objective in `neural_compressor.config.TuningCriterion` as shown below:

```python
from neural_compressor.config import TuningCriterion
tuning_criterion = TuningCriterion(objective='accuracy')

```

### Config Multiple Objectives

Users can specify built-in multiple objectives in `neural_compressor.config.TuningCriterion` as shown below:

```python
from neural_compressor.config import TuningCriterion
tuning_criterion = TuningCriterion(objective=['performance', 'accuracy'])
```

## Example
Refer to [example](../neural_compressor/template/ptq.yaml) as an example.
