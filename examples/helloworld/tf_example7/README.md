tf_example7 example
=====================
This example is used to demonstrate how to quantize a TensorFlow model with dummy dataset.

Step-by-Step
============

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
python test.py
``` 

### 2. Introduction
We can quantize a model only needing to set the dataloader with dummy dataset to generate an int8 model.
```python
    # Built-in dummy dataset
    dataset = Datasets('tensorflow')['dummy'](shape=(1, 224, 224, 3))
    # Built-in calibration dataloader and evaluation dataloader for Quantization.
    dataloader = DataLoader(framework='tensorflow', dataset=dataset)
    # Post Training Quantization Config
    config = PostTrainingQuantConfig()
    # Just call fit to do quantization.
    q_model = fit(model="./mobilenet_v1_1.0_224_frozen.pb",
                  conf=config,
                  calib_dataloader=dataloader)
```