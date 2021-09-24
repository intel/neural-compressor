Graph Optimization
==================

## Introduction

Graph optimization is primarily focused on two scenarios, shown below:

1. **FP32 optimization**. This is similar to the TensorFlow optimization tool [optimize_for_inference](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference.py) while Neural Compressor enables more optimizations (such as common subexpression elimination).

2. **Auto-mixed precision optimization**. Neural Compressor generates the optimal model with auto-mixed precision ([bfloat16](https://cloud.google.com/tpu/docs/bfloat16) and FP32) and allows for additional auto-tuning per accuracy requirements.


## How to use it

See the following three examples which demonstrate graph optimization API usage.

### FP32 Optimization

Neural Compressor runs the graph optimization under FP32 Optimization by default. In other words, the **precisions** field is explicitly set to **fp32**:

```python
    from neural_compressor.experimental import Graph_Optimization
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
      from neural_compressor.experimental import Graph_Optimization
      graph_optimizer = Graph_Optimization()
      graph_optimizer.precisions = 'bf16, fp32'
      graph_optimizer.input = 'input'  # Optional
      graph_optimizer.output = 'op_to_store'  # Optional
      graph_optimizer.model = '/path/to/model'
      optimized_model = graph_optimizer()
  ```
Note the **fp32** is optional when the **bf16** is set to precisions field. The below example has the identical action under the hardware platform supports bf16, e.g, the CPX platform.
  ```python
      from neural_compressor.experimental import Graph_Optimization
      graph_optimizer = Graph_Optimization()
      graph_optimizer.precisions = 'bf16'
      graph_optimizer.model = '/path/to/model'
      optimized_model = graph_optimizer()
  ```
For those platforms without bf16 enabling, like CLX. Neural Compressor also could leverage the graph optimization feature to generate the model under bf16 precision.The usage is just adding the `FORCE_BF16=1` before the cmd.
e.g, `FORCE_BF16=1 /path/to/executable_nc_wrapper`. If we do not add such prefix `FORCE_BF16=1`, the program would exit consequently.


#### Auto-mixed precision with auto-tuning

Neural Compressor also supports tuning the model in graph optimization mode. The end user must replace the quantization field with graph_optimization parts such as shown below. The **precisions** field only supports **bf16** and **fp32**.

  ```yaml
  graph_optimization:
    precisions: ['bf16', 'fp32']
  ```
Note that if we remove the evaluation field from the yaml file, the graph optimization will only convert the model depending on the precisions setting.

When the graph_optimization field is set and the evaluation field exists in the yaml file, Neural Compressor executes the similar process like quantization. It converts op into bf16 as much as possible and checks the metric later. If the metric meets the criterion, Neural Compressor exits or it fallbacks one op to fp32 and re-runs the above process until it meets the exit policy setting.

Below is an example of using yaml to trigger graph optimization.

  ```python
      from neural_compressor.experimental import Graph_Optimization
      graph_optimizer = Graph_Optimization('/path/to/config.yaml')
      graph_optimizer.model = '/path/to/model'
      optimized_model = graph_optimizer()
  ```

Graph_Optimization class also support Graph_Optimization_Conf class as it's argument.

  ```python
      from lpot.experimental import Graph_Optimization
      from lpot.conf.config import Graph_Optimization_Conf
      conf = Graph_Optimization_Conf('/path/to/config.yaml')
      graph_optimizer = Graph_Optimization(conf)
      graph_optimizer.model = '/path/to/model'
      optimized_model = graph_optimizer()
  ```

  ## Examples

  ### FP32 optimization
  The below example demonstrate how to speed up the Resnet50 FP32 throughput performance via Graph Optimization.
  1. Download the pre-trained ResNet-50 model with below command.
  ```shell
    wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet50_fp32_pretrained_model.pb

  ```
  2. Measure the performance on original FP32 model.

  First of all, we create the **resnet50_measurement.yaml** with below settings for leveraging Neural Compressor Benchmark API.

  ```yaml
    model:
      name: resnet50_v1
      framework: tensorflow

    evaluation:
      performance:
        configs:
          cores_per_instance: 28
          num_of_instance: 1
        dataloader:
          batch_size: 100
          dataset:
            dummy:
              shape: [1000, 224, 224, 3]
  ```

  Then, we can leverage the Benchmark API to measure the performance.
  ```python
  from neural_compressor.experimental import Benchmark
  evaluator = Benchmark('/path/to/resnet50_measurement.yaml')
  evaluator.model = '/path/to/resnet50_fp32_pretrained_model.pb'
  evaluator('performance')
  ```

  We got below performance result under Intel Xeon Scalable processor Cascade Lake 8280.
  ```shell
  performance mode benchmark result:
2021-05-28 15:16:11 [INFO] Batch size = 100
2021-05-28 15:16:11 [INFO] Latency: 7.165 ms
2021-05-28 15:16:11 [INFO] Throughput: 139.567 images/sec

```
3. Re-Measure the performance on optimized FP32 model.
  ```python
  from neural_compressor.experimental import Graph_Optimization

  graph_optimizer = Graph_Optimization()
  graph_optimizer.model = '/path/to/resnet50_fp32_pretrained_model.pb'
  output_graph = graph_optimizer()
  output_graph.save('/path/to/fp32_optimized_model')
  ```
Then, We measure the optimized performance via Neural Compressor Benchmark API again.
  ```python
  from neural_compressor.experimental import Benchmark
  evaluator = Benchmark('/path/to/resnet50_measurement.yaml')
  evaluator.model = '/path/to/fp32_optimized_model'
  evaluator('performance')
  ```

Now, the throughput has been improved ~2.3x (325.99 vs 139.56) compared with the initial data.
```shell
performance mode benchmark result:
2021-05-28 15:16:41 [INFO] Batch size = 100
2021-05-28 15:16:41 [INFO] Latency: 3.068 ms
2021-05-28 15:16:41 [INFO] Throughput: 325.992 images/sec
```
