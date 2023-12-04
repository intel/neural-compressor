tf_example3 example
=====================

Step-by-Step
============

This example is used to demonstrate how to convert a TensorFlow model with mix precision.

## Prerequisite
### 1. Installation
```shell
pip install -r requirements.txt
```
> Note: Validated TensorFlow [Version](/docs/source/installation_guide.md#validated-software-environment).

### 2. Download the FP32 model
```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/mobilenet_v1_1.0_224_frozen.pb
```

## Run
### 1. Run Command
```shell
python test.py --dataset_location=/path/to/imagenet/
``` 

### 2. Introduction
We can get a BF16 model using the Mixed Precision API.
```python
    from neural_compressor.config import MixedPrecisionConfig
    from neural_compressor import mix_precision
    from neural_compressor import Metric
    top1 = Metric(k=1)
    config = MixedPrecisionConfig()
    mix_precision_model = mix_precision.fit(
        model="./mobilenet_v1_1.0_224_frozen.pb",
        config=config,
        eval_dataloader=eval_dataloader,
        eval_metric=top1)
```
