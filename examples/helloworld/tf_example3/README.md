tf_example3 example
=====================
This example is used to demonstrate how to utilize Neural Compressor builtin dataloader and metric to enabling quantization for models defined in slim.

### 1. Installation
```shell
pip install -r requirements.txt
```

### 2. Prepare Dataset  
TensorFlow [models](https://github.com/tensorflow/models) repo provides [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) to download, process and convert the ImageNet dataset to the TF records format.
We also prepared related scripts in [TF image_recognition example](../../tensorflow/image_recognition/tensorflow_models/quantization/ptq/README.md#2-prepare-dataset). 

### 3. Prepare the FP32 model
```shell
wget http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz
tar -xvf inception_v1_2016_08_28.tar.gz
```

### 4. Config dataloader in conf.yaml
The configuration will help user to create a dataloader of Imagenet and it will do Bilinear resampling to resize the image to 224x224. It also creates a TopK metric function for evaluation.  

```yaml
quantization:                                        # optional. tuning constraints on model-wise for advance user to reduce tuning space.
  calibration:
    sampling_size: 20, 50                            # optional. default value is 100. used to set how many samples should be used in calibration.
    dataloader:
      batch_size: 10
      dataset:
        ImageRecord:
          root: /path/to/imagenet/                   # NOTE: modify to calibration dataset location if needed
      transform:
        BilinearImagenet: 
          height: 224
          width: 224
......
evaluation:                                          # optional. required if user doesn't provide eval_func in Quantization.
  accuracy:                                          # optional. required if user doesn't provide eval_func in Quantization.
    metric:
      topk: 1                                        # built-in metrics are topk, map, f1, allow user to register new metric.
    dataloader:
      batch_size: 1 
      last_batch: discard 
      dataset:
        ImageRecord:
          root: /path/to/imagenet/                   # NOTE: modify to evaluation dataset location if needed
      transform:
        BilinearImagenet: 
          height: 224
          width: 224

```

### 5. Run Command
The cmd of quantization and predict with the quantized model 
```shell
python test.py 
```

### 6. Introduction
In order to do quantization for slim models, we need to get graph from slim .ckpt first. 
```python
    from neural_compressor.experimental import Quantization, common
    quantizer = Quantization('./conf.yaml')

    # Do quantization
    quantizer.model = common.Model('./inception_v1.ckpt')
    quantized_model = quantizer.fit()
 
```

