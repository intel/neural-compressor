Objectives
==========

In terms of evaluating the status of a specific model during tuning, we should have general objectives to measure the status of different models. Neural Compressor Objectives supports code-free configuration through a yaml file, with built-in objectives, so users can compress model with different objectives easily. In special cases, users can also register their own objective classes through below method.


## How to use Objectives

### Config built-in objective in a yaml file

Users can specify an Neural Compressor built-in objective as shown below:

```yaml
tuning:
  objective: performance
```

### Config custom objective in code

Users can also register their own objective and pass it to quantizer as below:

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

## Multi-objectives support

In some cases, users want to use more than one objective to evaluate the status of a specific model and they can realize it with multi_objectives of Neural Compressor. Currently multi_objectives supports built-in objectives.

### Config multi_objectives in a yaml file

```yaml
tuning:
  multi_objectives:
    objective: [accuracy, performance]
    higher_is_better: [True, False]
    weight: [0.8, 0.2] # default is to calculate the average value of objectives
```

If users use multi_objectives to evaluate the status of a model during tuning, Neural Compressor will return a model with the best score of multi_objectives and meeting accuracy_criterion after tuning ending. 


When calculating the weighted score of objectives, Neural Compressor will normalize the results of objectives to [0, 1] one by one first.

## Built-in objective support list

| Objective    | Usage                                                    |
| :------      | :------                                                  |
| Accuracy     | Evaluate the accuracy                                    |
| Performance  | Evaluate the inference time                              |
| Footprint    | Evaluate the peak size of memory blocks during inference |
| ModelSize    | Evaluate the model size                                  |
