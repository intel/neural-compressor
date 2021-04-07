tf_example6 example
=====================
This example is used to demonstrate how to use default user-facing APIs to quantize a model.

1. Download the FP32 model
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/mobilenet_v1_1.0_224_frozen.pb

2. Update the root of dataset in conf.ymal
The configuration will will create a TopK metric function for evaluation and configure the batch size, instance number and core number for performacne measurement.    
```yaml
evaluation:                                          # optional. required if user doesn't provide eval_func in Quantization.
 accuracy:                                           # optional. required if user doesn't provide eval_func in Quantization.
    metric:
      topk: 1                                        # built-in metrics are topk, map, f1, allow user to register new metric.
    dataloader:
      batch_size: 32 
      dataset:
        ImageRecord:
          root: /path/to/imagenet/                   # NOTE: modify to evaluation dataset location if needed
      transform:
        ParseDecodeImagenet: {}
        BilinearImagenet: 
          height: 224
          width: 224

 performance:                                        # optional. used to benchmark performance of passing model.
    configs:
      cores_per_instance: 4
      num_of_instance: 7
    dataloader:
      batch_size: 1 
      last_batch: discard 
      dataset:
        ImageRecord:
          root: /path/to/imagenet/                   # NOTE: modify to evaluation dataset location if needed
      transform:
        ParseDecodeImagenet: {}
        ResizeCropImagenet: 
          height: 224
          width: 224
          mean_value: [123.68, 116.78, 103.94]

```

3. Run quantizaiton
We only need to add the following lines for quantization to create an int8 model.
```python
    from lpot import Quantization
    quantizer = Quantization('./conf.yaml')
    quantized_model = quantizer('./mobilenet_v1_1.0_224_frozen.pb')
```
* Run quantization and evaluation:
```shell
    python test.py
``` 

4. Run benchmark accoridng to config
```python
     # Optional, run benchmark 
    from lpot import Benchmark
    evaluator = Benchmark('./conf.yaml')
    results = evaluator(quantized_model)
 
```
