Graph Optimization
==================

## Introduction

Graph optimization is primarily focused on two scenarios, shown below:

1. **FP32 optimization**. This is similar to the TensorFlow optimization tool [optimize_for_inference](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference.py) while LPOT enables more optimizations (such as common subexpression elimination).

2. **Auto-mixed precision optimization**. LPOT generates the optimal model with auto-mixed precision ([bfloat16](https://cloud.google.com/tpu/docs/bfloat16) and FP32) and allows for additional auto-tuning per accuracy requirements.


## How to use it

See the following three examples which demonstrate graph optimization API usage.

### FP32 Optimization

LPOT runs the graph optimization under FP32 Optimization by default. In other words, the **precisions** field is explicitly set to **fp32**: 

```python
    from lpot.experimental import Graph_Optimization
    graph_optimizer = Graph_Optimization()
    graph_optimizer.precisions = 'fp32' #Optional, default is 'fp32'
    graph_optimizer.input = 'input'  # Optional
    graph_optimizer.output = 'op_to_store'  # Optional
    graph_optimizer.model = '/path/to/model'
    optimized_model = graph_optimizer()
```

### Auto-mixed Precision Optimization

#### Default auto-mixed precision

The only difference between this and the default mode (FP32 optimization) is that **bf16** must be added to the **precisions** field.

  ```python
      from lpot.experimental import Graph_Optimization
      graph_optimizer = Graph_Optimization()
      graph_optimizer.precisions = 'bf16, fp32'
      graph_optimizer.input = 'input'  # Optional
      graph_optimizer.output = 'op_to_store'  # Optional
      graph_optimizer.model = '/path/to/model'
      optimized_model = graph_optimizer()
  ```

#### Auto-mixed precision with auto-tuning

LPOT also supports tuning the model in graph optimization mode. The end user must replace the quantization field with graph_optimization parts such as shown below. The **precisions** field only supports **bf16** and **fp32**.

  ```yaml
  graph_optimization:
    precisions: ['bf16', 'fp32']
  ```
Note that if we remove the evaluation field from the yaml file, the graph optimization will only convert the model depending on the precisions setting.

When the graph_optimization field is set and the evaluation field exists in the yaml file, LPOT executes the similar process like quantization. It means the LPOT converts op into bf16 as much as possible and checks the metric later. If the metric meets the criterion, LPOT exits or it fallbacks one op to fp32 and re-runs the above process until it meets the exit policy setting.

Below is an example of using yaml to trigger graph optimization.

  ```python
      from lpot.experimental import Graph_Optimization
      graph_optimizer = Graph_Optimization('/path/to/config.yaml')
      graph_optimizer.model = '/path/to/model'
      optimized_model = graph_optimizer()
  ```