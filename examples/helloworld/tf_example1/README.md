tf_example1 example
=====================
This example is used to demonstrate how to utilize LPOT builtin dataloader and metric to enabling quantization without coding effort.

1. Download the FP32 model
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/mobilenet_v1_1.0_224_frozen.pb

2. Update the root of dataset in conf.ymal
The configuration will create a dataloader of Imagenet and it will do Bilinear resampling to resize the image to 224x224. And it will create a TopK metric function for evaluation.   
```yaml
quantization:                                        # optional. tuning constraints on model-wise for advance user to reduce tuning space.
  calibration:
    sampling_size: 20                                # optional. default value is 100. used to set how many samples should be used in calibration.
    dataloader:
      batch_size: 1
      dataset:
        ImageRecord:
          root: <DATASET>/TF_imagenet/val/           # NOTE: modify to calibration dataset location if needed
      transform:
        ParseDecodeImagenet:
        BilinearImagenet: 
          height: 224
          width: 224
  model_wise:                                        # optional. tuning constraints on model-wise for advance user to reduce tuning space.
    activation:
      algorithm: minmax
    weight:
      granularity: per_channel

evaluation:                                          # optional. required if user doesn't provide eval_func in lpot.Quantization.
  accuracy:                                          # optional. required if user doesn't provide eval_func in lpot.Quantization.
    metric:
      topk: 1                                        # built-in metrics are topk, map, f1, allow user to register new metric.
    dataloader:
      batch_size: 32
      dataset:
        ImageRecord:
          root: <DATASET>/TF_imagenet/val/           # NOTE: modify to evaluation dataset location if needed
      transform:
        ParseDecodeImagenet:
        BilinearImagenet: 
          height: 224
          width: 224

```

3. Run quantizaiton
We only need to add the following lines for quantization to create an int8 model.
```python
    from lpot.experimental import Quantization, common
    quantizer = Quantization('./conf.yaml')
    quantizer.model = common.Model("./mobilenet_v1_1.0_224_frozen.pb")
    quantized_model = quantizer()
```
* Run quantization and evaluation:
```shell
    python test.py
``` 

