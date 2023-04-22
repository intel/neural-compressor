tf_example1 example
=====================

Step-by-Step
============
This example is used to demonstrate how to utilize Neural Compressor builtin dataloader and metric to enabling quantization without coding effort.

## Prerequisite
### 1. Installation
```shell
pip install -r requirements.txt
```
> Note: Validated TensorFlow [Version](/docs/source/installation_guide.md#validated-software-environment).

### 2. Prepare Dataset  
TensorFlow [models](https://github.com/tensorflow/models) repo provides [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) to download, process and convert the ImageNet dataset to the TF records format.
We also prepared related scripts in [TF image_recognition example](/examples/tensorflow/image_recognition/tensorflow_models/mobilenet_v1/quantization/ptq#3-prepare-dataset).

### 3. Download the FP32 model
```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/mobilenet_v1_1.0_224_frozen.pb
```

## Run
### 1. Run Command
```shell
python test.py --dataset_location=/path/to/imagenet/
``` 

### 2. Introduction 
We only need to add the following lines for quantization to create an int8 model.
```python
    from neural_compressor import Metric
    top1 = Metric(name="topk", k=1)
    config = PostTrainingQuantConfig(calibration_sampling_size=[20])
    q_model = fit(
        model="./mobilenet_v1_1.0_224_frozen.pb",
        conf=config,
        calib_dataloader=calib_dataloader,
        eval_dataloader=eval_dataloader,
        eval_metric=top1)
```


