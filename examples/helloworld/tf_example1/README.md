tf_example1 example
=====================
This example is used to demonstrate how to utilize Neural Compressor builtin dataloader and metric to enabling quantization without coding effort.

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
The configuration will create a dataloader of Imagenet and it will do Bilinear resampling to resize the image to 224x224. And it will create a TopK metric function for evaluation.   
```yaml
quantization:                                        # optional. tuning constraints on model-wise for advance user to reduce tuning space.
  calibration:
    sampling_size: 20                                # optional. default value is 100. used to set how many samples should be used in calibration.
    dataloader:
      dataset:
        ImageRecord:
          root: <DATASET>/TF_imagenet/val/           # NOTE: modify to calibration dataset location if needed
      transform:
        BilinearImagenet: 
          height: 224
          width: 224
  model_wise:                                        # optional. tuning constraints on model-wise for advance user to reduce tuning space.
    activation:
      algorithm: minmax
    weight:
      granularity: per_channel

evaluation:                                          # optional. required if user doesn't provide eval_func in neural_compressor.Quantization.
  accuracy:                                          # optional. required if user doesn't provide eval_func in neural_compressor.Quantization.
    metric:
      topk: 1                                        # built-in metrics are topk, map, f1, allow user to register new metric.
    dataloader:
      batch_size: 32
      dataset:
        ImageRecord:
          root: <DATASET>/TF_imagenet/val/           # NOTE: modify to evaluation dataset location if needed
      transform:
        BilinearImagenet: 
          height: 224
          width: 224

```

### 5. Run Command
```shell
python test.py
``` 

### 6. Introduction 
We only need to add the following lines for quantization to create an int8 model.
```python
    from neural_compressor.experimental import Quantization, common
    quantizer = Quantization('./conf.yaml')
    quantizer.model = common.Model("./mobilenet_v1_1.0_224_frozen.pb")
    quantized_model = quantizer.fit()
```


