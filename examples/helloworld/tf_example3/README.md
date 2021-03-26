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
    sampling_size: 20, 50                            # optional. default value is the size of whole dataset. used to set how many portions of calibration dataset is used. exclusive with iterations field.
    dataloader:
      batch_size: 10
      dataset:
        ImageRecord:
          root: /path/to/imagenet/         # NOTE: modify to calibration dataset location if needed
      transform:
        ParseDecodeImagenet:
        BilinearImagenet: 
          height: 224
          width: 224
......
evaluation:                                          # optional. required if user doesn't provide eval_func in lpot.Quantization.
  accuracy:                                          # optional. required if user doesn't provide eval_func in lpot.Quantization.
    metric:
      topk: 1                                        # built-in metrics are topk, map, f1, allow user to register new metric.
    dataloader:
      batch_size: 1 
      last_batch: discard 
      dataset:
        ImageRecord:
          root: /path/to/imagenet/          # NOTE: modify to evaluation dataset location if needed
      transform:
        ParseDecodeImagenet:
        BilinearImagenet: 
          height: 224
          width: 224

```

3. Run quantizaiton
* In order to do quanzation for slim models, we need to get graph from slim .ckpt first. 
```python
    import lpot
    from lpot import common
    quantizer = lpot.Quantization('./conf.yaml')

    # Get graph from slim checkpoint
    from tf_slim.nets import inception
    model_func = inception.inception_v1
    arg_scope = inception.inception_v1_arg_scope()
    kwargs = {'num_classes': 1001}
    inputs_shape = [None, 224, 224, 3]
    images = tf.compat.v1.placeholder(name='input', \
    dtype=tf.float32, shape=inputs_shape)

    from lpot.adaptor.tf_utils.util import get_slim_graph
    graph = get_slim_graph('./inception_v1.ckpt', model_func, \
            arg_scope, images, **kwargs)
    
    # Do quantization
    quantizer.model = common.Model('./inception_v1.ckpt')
    quantized_model = quantizer()
 
```

