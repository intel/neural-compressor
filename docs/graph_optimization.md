## Graph Optimization Introduction

The graph optimization is mainly focus on below two kind of scenarios.

1. **FP32 optimization**. This is similar to TensorFlow optimization tool [optimize_for_inference](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference.py) while LPOT enables more optimizations (e.g., common subexpression elimination).

2. **Auto-mixed precision optimization**. LPOT generates the optimal model with auto-mixed precision ([bfloat16](https://cloud.google.com/tpu/docs/bfloat16) & FP32) and allows the further auto-tuning per accuracy requirement.


## How to use it
Generally, we list below three examples to demonstrate the usage of graph optimization API.

#### 1. **FP32 Optimization**

LPOT would run graph optimization under FP32 optimization by default. In other words, it equals the user set the precisions to **'fp32'** explicitly, 

```python
    from lpot.experimental import Graph_Optimization
    graph_optimizer = Graph_Optimization()
    graph_optimizer.precisions = 'fp32' #Optional, default is 'fp32'
    graph_optimizer.input = 'input'  # Optional
    graph_optimizer.output = 'op_to_store'  # Optional
    graph_optimizer.model = '/path/to/model'
    optimized_model = graph_optimizer()
```

#### 2. **Auto-mixed Precision Optimization**

  #### 2.1. *Default auto-mixed precision*

  The only difference between this and default mode is the `bf16` needs to be added into the precisions filed.

  ```python
      from lpot.experimental import Graph_Optimization
      graph_optimizer = Graph_Optimization()
      graph_optimizer.precisions = 'bf16, fp32'
      graph_optimizer.input = 'input'  # Optional
      graph_optimizer.output = 'op_to_store'  # Optional
      graph_optimizer.model = '/path/to/model'
      optimized_model = graph_optimizer()
  ```

  #### 2.2. *Auto-mixed precision with auto-tuning.*

  LPOT also supports the tuning the model under graph optimization mode.
The end user need to replace the quantization field with graph_optimization parts like below. The precisions filed only supports 'bf16' and 'fp32'.
  ```yaml
  graph_optimization:
    precisions: ['bf16', 'fp32']
  ```
  Note, if we remove the evaluation field from the yaml, the graph optimization will only convert the model depends on the precisions setting.

  When the graph_optimization field set and the evaluation field exists in the yaml, LPOT will execute the similar process like quantization. It means the LPOT would convert op into bf16 as much as possible and check the metric later, if the metric meet the criterion, the LPOT would exit or it would fallback one op to fp32 and re-run the above process till it meet the exit policy setting.

  Below is the example of using the yaml to trigger the graph optimization.

  ```python
      from lpot.experimental import Graph_Optimization
      graph_optimizer = Graph_Optimization('/path/to/config.yaml')
      graph_optimizer.model = '/path/to/model'
      optimized_model = graph_optimizer()
  ```