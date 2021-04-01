tf_example3 example
=====================
This example is used to demonstrate how to utilize LPOT builtin dataloader and metric to enabling quantization for models defined in slim.
 

1. Prepare 
* Prepare the FP32 model
wget http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz
tar -xvf inception_v1_2016_08_28.tar.gz

* Instanll dependencies
pip install -r requirements.txt


2. Config dataloader in conf.ymal
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
        ParseDecodeImagenet:
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
        ParseDecodeImagenet:
        BilinearImagenet: 
          height: 224
          width: 224

```

3. Run quantizaiton
* In order to do quanzation for slim models, we need to get graph from slim .ckpt first. 
```python
    from lpot.experimental import Quantization, common
    quantizer = Quantization('./conf.yaml')

    # Do quantization
    quantizer.model = common.Model('./inception_v1.ckpt')
    quantized_model = quantizer()
 
```

