Benchmarking
============

The benchmarking feature of Neural Compressor is used to measure the model performance with the objective settings; the user can get the performance of the models between the float32 model and the quantized low precision model in the same scenarios that they configured in Yaml. Benchmarking is always used after a quantization process.

The following examples show how to use benchmarking.

## Config evaluation filed in a yaml file

```yaml
evaluation:                                          # optional. required if user doesn't provide eval_func in neural_compressor.Quantization.
  accuracy:                                          # optional. required if user doesn't provide eval_func in neural_compressor.Quantization.
    metric:
      topk: 1                                        # built-in metrics are topk, map, f1, allow user to register new metric.
    dataloader:
      batch_size: 30
      dataset:
        ImageFolder:
          root: /path/to/evaluation/dataset          # NOTE: modify to evaluation dataset location if needed
      transform:
        Resize:
          size: 256
        CenterCrop:
          size: 224
        ToTensor: {}
        Normalize:
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
  performance:                                       # optional. used to benchmark performance of passing model.
    configs:
      cores_per_instance: 4
      num_of_instance: 7
    dataloader:
      batch_size: 1
      dataset:
        ImageFolder:
          root: /path/to/evaluation/dataset          # NOTE: modify to evaluation dataset location if needed
      transform:
        Resize:
          size: 256
        CenterCrop:
          size: 224
        ToTensor: {}
        Normalize:
          mean: [0.485, 0.456, 0.406]
```

The above example config two sub-fields named 'accuracy' and 'performance' which indicates that the benchmark module will get the accuracy and performance of the model. The user can also remove the performance field to only get model accuracy or performance. It's flexible enough to configure the benchmark you want.

## Use a user-specific dataloader to run benchmark

In this case, configure your dataloader and Neural Compressor will construct an evaluation function to run the benchmarking. The user can also register the postprocess transform and metric to get the accuracy.

```python
dataset = Dataset() #  dataset class that implement __getitem__ method or __iter__ method
from neural_compressor.experimental import Benchmark, common
evaluator = Benchmark(config.yaml)
evaluator.dataloader = common.DataLoader(dataset, batch_size=batch_size)
# user can also register postprocess and metric, this is optional
evaluator.postprocess = common.Postprocess(postprocess_cls)
evaluator.metric = common.Metric(metric_cls)
results = evaluator()

```
Benchmark class also support BenchmarkConf class as it's argument:
```python
dataset = Dataset() #  dataset class that implement __getitem__ method or __iter__ method
from lpot.experimental import Benchmark, common
from lpot.conf.config import BenchmarkConf
conf = BenchmarkConf(config.yaml)
evaluator = Benchmark(conf)
evaluator.dataloader = common.DataLoader(dataset, batch_size=batch_size)
# user can also register postprocess and metric, this is optional
evaluator.postprocess = common.Postprocess(postprocess_cls)
evaluator.metric = common.Metric(metric_cls)
results = evaluator()

```

### Examples

Refer to the [Benchmark example](../examples/tensorflow/image_recognition/tensorflow_models/quantization/ptq/run_benchmark.sh).

