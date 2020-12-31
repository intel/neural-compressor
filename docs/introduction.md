Introduction
=========================================

Intel® Low Precision Optimization Tool is an open-source Python library designed to help users quickly deploy low-precision inference solutions on popular deep learning (DL) frameworks such as TensorFlow\*, PyTorch\*, MXNet and ONNX Runtime. It automatically optimizes low-precision recipes for deep learning models in order to achieve optimal product objectives, such as inference performance and memory usage, with expected accuracy criteria.


# User-facing API

The API is intended to unify low-precision quantization interfaces cross multiple DL frameworks for the best out-of-the-box experiences.

The API consists of three components:

### quantization-related APIs
```
class Quantization(object):
    def __init__(self, conf_fname):
        ...

    def __call__(self, model, q_dataloader=None, q_func=None, eval_dataloader=None, eval_func=None):
        ...

```
The `conf_fname` parameter used in the class initialization is the path to user yaml configuration file. This is a yaml file that is used to control the entire tuning behavior.

> **LPOT User YAML Syntax**
>
> Intel® Low Precision Optimization Tool provides template yaml files for the [Post-Training Quantization](../lpot/template/ptq.yaml), [Quantization-Aware Traing](../lpot/template/qat.yaml), and [Pruning](../lpot/template/pruning.yaml) scenarios. Refer to these template files to understand the meaning of each field.

> Note that most fields in the yaml templates are optional. View the [HelloWorld Yaml](../examples/helloworld/tf_example2/conf.yaml) example for reference.

For TensorFlow backend, LPOT supports passing the path of keras model, frozen pb, checkpoint, saved model as the input of `model` parameter of `Quantization()`.

For PyTorch backend, LPOT supports the instance of `torch.nn.model` as the input of `model` parameter of `Quantization()`.

For MXNet backend, LPOT supports the instance of `mxnet.symbol.Symbol` and `gluon.HybirdBlock` as the input of `model` parameter of `Quantization()`.


### pruning-related APIs (POC)
```
class Pruning(object):
    def __init__(self, conf_fname):
        ...

    def on_epoch_begin(self, epoch):
        ...

    def on_batch_begin(self, batch_id):
        ...

    def on_batch_end(self):
        ...

    def on_epoch_end(self):
        ...

    def __call__(self, model, q_dataloader=None, q_func=None, eval_dataloader=None, eval_func=None):
        ...
```

### benchmarking-related APIs
```
class Benchmark(object):
    def __init__(self, conf_fname):
        ...

    def __call__(self, model, b_dataloader=None, b_func=None):
        ...
```

# Quantization API Usage

Essentially Intel® Low Precision Optimization Tool constructs the quantization process by asking user to provide three components, `dataloader`, `model` and `metric`, through code or yaml configuration.

## dataloader

User can implement `dataloader` component by filling code into below:

```
class dataset(object):
  def __init__(self, *args):
      # initialize dataset related info here
      ...

  def __getitem__(self, index):
      # return a tuple containing 1 image and 1 label
      # exclusive with __iter__() magic method
      ...

  def __iter__(self):
      # return a tuple containing 1 image and 1 label
      # exclusive with __getitem__() magic method
      ...

  def __len__(self):
      # return total length of dataset
      ...
```

or by user yaml configuration through setting up `dataloader` field in `calibration` and `quantization` section of yaml file with LPOT build-in dataset/dataloader/transform.

## metric
User can implement `metric` component by filling code into below:
```
from lpot.metric.metric import Metric
class metric(Metric):
  def __init__(self, *args):
      # initialize metric related info here
      ...

  def update(self, predict, label):
      # metric evaluation per mini-batch
      ...

  def reset(self):
      # reset variable if needed
      ...

  def result(self):
      # calculate the whole batch final evaluation result
      # return a float value which is higher-is-better.
      ...
```
or by user yaml configuration through setting up `accuracy` field in `evaluation` section of yaml file with LPOT build-in metrics.

## run quantization on model

If `dataloader` and `metric` component get configured by code, the quantization process would start with below lines:

```
quantizer = Quantization('/path/to/user.yaml')
dataloader = tuner.dataloader(dataset, batch_size=100)
quantizer.metric('metric', metric)
q_model = quantizer('/path/to/model', q_dataloader = dataloader, eval_dataloader = dataloader)
```

If `dataloader` and `metric` components get fully configured by yaml, the quantization process would start with below lines:

```
quantizer = Quantization('/path/to/user.yaml')
q_model = quantizer('/path/to/model')
```
Examples of this usage are at [TensorFlow Classification Models](../examples/tensorflow/image_recognition/README.md).
