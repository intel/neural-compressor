API Documentation
=================

## Introduction

Intel® Neural Compressor is an open-source Python library designed to help users quickly deploy low-precision inference solutions on popular deep learning (DL) frameworks such as TensorFlow*, PyTorch*, MXNet, and ONNX Runtime. It automatically optimizes low-precision recipes for deep learning models in order to achieve optimal product objectives, such as inference performance and memory usage, with expected accuracy criteria.


## User-facing APIs

These APIs are intended to unify low-precision quantization interfaces cross multiple DL frameworks for the best out-of-the-box experiences.

> **Note**
>
> Neural Compressor is continuously improving user-facing APIs to create a better user experience. 

> Two sets of user-facing APIs exist. One is the default one supported from Neural Compressor v1.0 for backwards compatibility. The other set consists of new APIs in 
the `neural_compressor.experimental` package.

> We recommend that you use the APIs located in neural_compressor.experimental. All examples have been updated to use the experimental APIs.

The major differences between the default user-facing APIs and the experimental APIs are:

1. The experimental APIs abstract the `neural_compressor.experimental.common.Model` concept to cover those cases whose weight and graph files are stored separately.
2. The experimental APIs unify the calling style of the `Quantization`, `Pruning`, and `Benchmark` classes by setting model, calibration dataloader, evaluation dataloader, and metric through class attributes rather than passing them as function inputs.
3. The experimental APIs refine Neural Compressor built-in transforms/datasets/metrics by unifying the APIs cross different framework backends.

## Experimental user-facing APIs

Experimental user-facing APIs consist of the following components:

### Quantization-related APIs

```python
# neural_compressor.experimental.Quantization
class Quantization(object):
    def __init__(self, conf_fname_or_obj):
        ...

    def __call__(self):
        ...

    @property
    def calib_dataloader(self):
        ...

    @property
    def eval_dataloader(self):
        ...

    @property
    def model(self):
        ...

    @property
    def metric(self):
        ...

    @property
    def postprocess(self, user_postprocess):
        ...

    @property
    def q_func(self):
        ...

    @property
    def eval_func(self):
        ...

```
The `conf_fname_or_obj` parameter used in the class initialization is the path to the user yaml configuration file or Quantization_Conf class. This yaml file is used to control the entire tuning behavior on the model.

**Neural Compressor User YAML Syntax**

> Intel® Neural Compressor provides template yaml files for [Post-Training Quantization](../neural_compressor/template/ptq.yaml), [Quantization-Aware Training](../neural_compressor/template/qat.yaml), and [Pruning](../neural_compressor/template/pruning.yaml) scenarios. Refer to these template files to understand the meaning of each field.

> Note that most fields in the yaml templates are optional. View the [HelloWorld Yaml](../examples/helloworld/tf_example2/conf.yaml) example for reference.

```python
# Typical Launcher code
from neural_compressor.experimental import Quantization, common

# optional if Neural Compressor built-in dataset could be used as model input in yaml
class dataset(object):
  def __init__(self, *args):
      ...

  def __getitem__(self, idx):
      # return single sample and label tuple without collate. label should be 0 for label-free case
      ...

  def len(self):
      ...

# optional if Neural Compressor built-in metric could be used to do accuracy evaluation on model output in yaml
class custom_metric(object):
    def __init__(self):
        ...

    def update(self, predict, label):
        # metric update per mini-batch
        ...

    def result(self):
        # final metric calculation invoked only once after all mini-batch are evaluated
        # return a scalar to neural_compressor for accuracy-driven tuning.
        # by default the scalar is higher-is-better. if not, set tuning.accuracy_criterion.higher_is_better to false in yaml.
        ...

quantizer = Quantization(conf.yaml)
quantizer.model = '/path/to/model'
# below two lines are optional if Neural Compressor built-in dataset is used as model calibration input in yaml
cal_dl = dataset('/path/to/calibration/dataset')
quantizer.calib_dataloader = common.DataLoader(cal_dl, batch_size=32)
# below two lines are optional if Neural Compressor built-in dataset is used as model evaluation input in yaml
dl = dataset('/path/to/evaluation/dataset')
quantizer.eval_dataloader = common.DataLoader(dl, batch_size=32)
# optional if Neural Compressor built-in metric could be used to do accuracy evaluation in yaml
quantizer.metric = common.Metric(custom_metric) 
q_model = quantizer.fit()
q_model.save('/path/to/output/dir') 
```

`model` attribute in `Quantization` class is an abstraction of model formats across different frameworks. Neural Compressor supports passing the path of `keras model`, `frozen pb`, `checkpoint`, `saved model`, `torch.nn.model`, `mxnet.symbol.Symbol`, `gluon.HybirdBlock`, and `onnx model` to instantiate a `neural_compressor.experimental.` class and set to `quantizer.model`.

`calib_dataloader` and `eval_dataloader` attribute in `Quantization` class is used to set up a calibration dataloader by code. It is optional to set if the user sets corresponding fields in yaml.

`metric` attribute in `Quantization` class is used to set up a custom metric by code. It is optional to set if user finds Neural Compressor built-in metric could be used with their model and sets corresponding fields in yaml.

`postprocess` attribute in `Quantization` class is not necessary in most of the use cases. It is only needed when the user wants to use the built-in metric but the model output can not directly be handled by Neural Compressor built-in metrics. In this case, the user can register a transform to convert the model output to the expected one required by the built-in metric.

`q_func` attribute in `Quantization` class is only for `Quantization Aware Training` case, in which the user needs to register a function that takes `model` as the input parameter and executes the entire training process with self-contained training hyper-parameters. 

`eval_func` attribute in `Quantization` class is reserved for special cases. If the user had an evaluation function when train a model, the user must implement a `calib_dataloader` and leave `eval_dataloader` as None. Then, modify this evaluation function to take `model` as the input parameter and return a higher-is-better scaler. In some scenarios, it may reduce development effort.


### Pruning-related APIs (POC)

```python
class Pruning(object):
    def __init__(self, conf_fname_or_obj):
        ...

    def on_epoch_begin(self, epoch):
        ...

    def on_step_begin(self, batch_id):
        ...

    def on_step_end(self):
        ...

    def on_epoch_end(self):
        ...

    def __call__(self):
        ...

    @property
    def model(self):
        ...

    @property
    def q_func(self):
        ...

```

This API is used to do sparsity pruning. Currently, it is a Proof of Concept; Neural Compressor only supports `magnitude pruning` on PyTorch.

To learn how to use this API, refer to the [pruning document](../docs/pruning.md).

### Benchmarking-related APIs
```python
class Benchmark(object):
    def __init__(self, conf_fname_or_obj):
        ...

    def __call__(self):
        ...

    @property
    def model(self):
        ...

    @property
    def metric(self):
        ...

    @property
    def b_dataloader(self):
        ...

    @property
    def postprocess(self, user_postprocess):
        ...
```

This API is used to measure model performance and accuracy.

To learn how to use this API, refer to the [benchmarking document](../docs/benchmark.md).

## Default user-facing APIs

The default user-facing APIs exist for backwards compatibility from the v1.0 release. Refer to [v1.1 API](https://github.com/intel/neural-compressor/blob/v1.1/docs/introduction.md) to understand how the default user-facing APIs work.

View the [HelloWorld example](/examples/helloworld/tf_example6) that uses default user-facing APIs for user reference. 

Full examples using default user-facing APIs can be found [here](https://github.com/intel/neural-compressor/tree/v1.1/examples).
