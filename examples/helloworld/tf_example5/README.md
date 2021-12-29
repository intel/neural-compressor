tf_example5 example
=====================
This example is used to demonstrate how to config benchmark in yaml for performance measurement.

### 1. Installation
```shell
pip install -r requirements.txt
```

### 2. Prepare Dataset  
TensorFlow [models](https://github.com/tensorflow/models) repo provides [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) to download, process and convert the ImageNet dataset to the TF records format.
We also prepared related scripts in [TF image_recognition example](../../tensorflow/image_recognition/tensorflow_models/quantization/ptq/README.md#2-prepare-dataset). 

### 3. Download the FP32 model
```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/mobilenet_v1_1.0_224_frozen.pb
```

### 4. Update the root of dataset in conf.yaml
The configuration will create a TopK metric function for evaluation and configure the batch size, instance number and core number for performance measurement.    
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
        ResizeCropImagenet: 
          height: 224
          width: 224
          mean_value: [123.68, 116.78, 103.94]

```

### 5. Run Command
* Run quantization
```shell
python test.py --tune
``` 
* Run benchmark, please make sure benchmark the model should after tuning.
```shell
python test.py --benchmark
``` 

### 6. Introduction
* We only need to add the following lines for quantization to create an int8 model.
```python
    from neural_compressor.experimental import Quantization, common
    quantizer = Quantization('./conf.yaml')
    quantizer.model = common.Model('./mobilenet_v1_1.0_224_frozen.pb')
    quantized_model = quantizer.fit()
    quantized_model.save('./int8.pb')
```
* Run benchmark according to config.
```python
    from neural_compressor.experimental import Quantization,  Benchmark, common
    evaluator = Benchmark('./conf.yaml')
    evaluator.model = common.Model('./int8.pb')
    results = evaluator()
 
```

