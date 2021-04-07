Introduction
=========================================

Intel® Low Precision Optimization Tool is an open-source Python library designed to help users quickly deploy low-precision inference solutions on popular deep learning (DL) frameworks such as TensorFlow, PyTorch, MXNet and ONNX Runtime. It automatically optimizes low-precision recipes for deep learning models in order to achieve optimal product objectives, such as inference performance and memory usage, with expected accuracy criteria.


# User-facing API

The API is intended to unify low-precision quantization interfaces cross multiple DL frameworks for the best out-of-the-box experiences.

> **NOTE**
>
> LPOT is keeping improving the user-facing APIs for better user experience. 
>
> Now there are two sets of user-facing APIs. One is the default one supported from LPOT v1.0 for backward compatibility. Another one is the new APIs in lpot.experimental package.
> We recommend user to use the one in lpot.experimental. All of examples have been updated to use this experimental APIs.
>
> The major differences between the default use-facing APIs and the experiemntal APIs are:
>   1. The experimental APIs abstract `lpot.experimental.common.Model` concept to cover those cases whose weight and graph files are stored seperately.
>   2. The experimental APIs unifiy the calling style of `Quantization`, `Pruning`, and `Benchmark` class by setting model, calibration dataloader, evaluation dataloader, metric through class attributes rather than passing as function inputs.
>   3. The experimental APIs refine LPOT built-in transforms/datasets/metrics by unifying the APIs cross different framework backends.

## Experimental user-facing APIs

The experimental user-facing APIs consist of below components:

### quantization-related APIs
```python
# lpot.experimental.Quantization
class Quantization(object):
    def __init__(self, conf_fname):
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
The `conf_fname` parameter used in the class initialization is the path to user yaml configuration file. This is a yaml file that is used to control the entire tuning behavior on the model.

> **LPOT User YAML Syntax**
>
> Intel® Low Precision Optimization Tool provides template yaml files for the [Post-Training Quantization](../lpot/template/ptq.yaml), [Quantization-Aware Traing](../lpot/template/qat.yaml), and [Pruning](../lpot/template/pruning.yaml) scenarios. Refer to these template files to understand the meaning of each field.

> Note that most fields in the yaml templates are optional. View the [HelloWorld Yaml](../examples/helloworld/tf_example2/conf.yaml) example for reference.

```python
# Typical Launcher code
from lpot.experimental import Quantization, common

# optional if LPOT built-in dataset could be used as model input in yaml
class dataset(object):
  def __init__(self, *args):
      ...

  def __getitem__(self, idx):
      # return single sample and label tuple without collate. label should be 0 for label-free case
      ...

  def len(self):
      ...

# optional if LPOT built-in metric could be used to do accuracy evaluation on model output in yaml
class custom_metric(object):
    def __init__(self):
        ...

    def update(self, predict, label):
        # metric update per mini-batch
        ...

    def result(self):
        # final metric calculation invoked only once after all mini-batch are evaluated
        # return a scalar to lpot for accuracy-driven tuning.
        # by default the scalar is higher-is-better. if not, set tuning.accuracy_criterion.higher_is_better to false in yaml.
        ...

quantizer = Quantization(conf.yaml)
quantizer.model = common.Model('/path/to/model')
# below two lines are optional if LPOT built-in dataset is used as model calibration input in yaml
cal_dl = dataset('/path/to/calibration/dataset')
quantizer.calib_dataloader = common.DataLoader(cal_dl, batch_size=32)
# below two lines are optional if LPOT built-in dataset is used as model evaluation input in yaml
dl = dataset('/path/to/evaluation/dataset')
quantizer.eval_dataloader = common.DataLoader(dl, batch_size=32)
# optional if LPOT built-in metric could be used to do accuracy evaluation in yaml
quantizer.metric = common.Metric(custom_metric) 
q_model = quantizer()
q_model.save('/path/to/output/dir') 
```

`model` attribute in `Quantization` class is an abstraction of model formats cross different frameworks. LPOT supports passing the path of `keras model`, `frozen pb`, `checkpoint`, `saved model`, `torch.nn.model`, `mxnet.symbol.Symbol`, `gluon.HybirdBlock`, and `onnx model` to instantiate a `lpot.experimental.common.Model()` class and set to `quantizer.model`.

`calib_dataloader` and `eval_dataloader` attribute in `Quantization` class is used to setup a calibration dataloader by code. It is optional to set if user sets corresponding fields in yaml.

`metric` attribute in `Quantization` class is used to setup a custom metric by code. It is optional to set if user finds LPOT built-in metric could be used with their model and sets corresponding fields in yaml.

`postprocess` attribute in `Quantization` class is not necessary in most of usage cases. It will only be needed when user wants to use LPOT built-in metric but model output could not directly be handled by LPOT built-in metrics. In this case, user could register a transform to convert model output to expected one required by LPOT built-in metric.

`q_func` attribute in `Quantization` class is only for `Quantization Aware Training` case, in which user need to register a function that takes `model` as input parameter and executes entire training process with self contained training hyper-parameters. 

`eval_func` attribute in `Quantization` class is reserved for special case. If user have had a evaluation function when train a model, user just needs to implement a `calib_dataloader` and leave `eval_dataloader` as None, modify this evaluation function to take `model` as input parameter and return a higher-is-better scaler. In some scenarios, it may reduce developement effort.


### pruning-related APIs (POC)
```python
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

    def __call__(self):
        ...

    @property
    def model(self):
        ...

    @property
    def q_func(self):
        ...

```

This API is used to do sparsity pruning. Currently it is Proof-of-Concept, LPOT only supports `magnitude pruning` on PyTorch.

For how to use this API, please refer to [Pruning Document](./pruning.md)

### benchmarking-related APIs
```python
class Benchmark(object):
    def __init__(self, conf_fname):
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

This API is used to measure the model performance and accuarcy. 

For how to use this API, please refer to [Benchmark Document](./benchmark.md)

## Default user-facing APIs

The default user-facing APIs would exist for backward compatiblity from v1.0 release. User could refer to [v1.1 API](https://github.com/intel/lpot/blob/v1.1/docs/introduction.md) to understand how default user-facing APIs work.

A [HelloWorld example](../examples/helloworld/tf_example6) using default user-facing APIs is provided for user reference.

Full examples using default user-facing APIs could be found at [here](https://github.com/intel/lpot/tree/v1.1/examples).
