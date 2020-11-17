Introduction
=========================================

Intel® Low Precision Optimization Tool is an open-source Python library designed to help users quickly deploy low-precision inference solutions on popular deep learning (DL) frameworks such as TensorFlow\*, PyTorch\*, and MXNet. It automatically optimizes low-precision recipes for deep learning models in order to achieve optimal product objectives, such as inference performance and memory usage, with expected accuracy criteria.


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

### pruning-related APIs
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

The `conf_fname` parameter used in the above class initialization is the path to the Intel® Low Precision Optimization Tool configuration file. This is a yaml file that is used to control the entire tuning behavior.

# YAML Syntax

Intel® Low Precision Optimization Tool provides template yaml files for the [PTQ](../ilit/template/ptq.yaml), [QAT](../ilit/template/qat.yaml), and [Pruning](../ilit/template/pruning.yaml) scenarios. Refer to the complete template to understand the meaning of each field.

Note that most fields in the yaml templates are optional. Additionally, a typical yaml file must be very concise. View the [HelloWorld Yaml](../examples/helloworld/tf2.x/conf.yaml) example for reference.

# Quantization API Usage

Intel® Low Precision Optimization Tool supports three different usages, depending on how the user code is organized:

### Template-based yaml setting + 3 lines of code changes

This first usage is designed for minimal code changes when integrating with Intel® Low Precision Optimization Tool. All calibration and evaluation processes are constructed by yaml, including dataloaders used in the calibration and evaluation phases and in the quantization tuning settings. For this usage, only the `model` parameter is mandatory.

View examples of this usage at [TensorFlow Classification Models](../examples/tensorflow/image_recognition/README.md).

### Concise template-based yaml + few lines of code changes

The second usage is designed for a concise yaml configuration by moving the calibration and evaluation dataloader construction from yaml to code. If the user model is using **ilit**-supported evaluation metrics (such as TOPK and MAP), this usage is a good choice.

The user must provide a **dataloader**-implemented `iter` or `getitem` method and `batch_size` attribute, which usually already exists or can be easily developed in the user code. ilit also provides built-in dataloaders to support dynamic batching; the user can set up a **dataset**-implemented `iter` or `getitem` method to yield one single batch. The `quantization().dataloader()` takes this dataset as an input parameter to construct the ilit dataloader.

After that, the user specifies the fp32 `model`, the `q_dataloader` calibration dataset, and the `eval_dataloader` evaluation dataset. The `eval_dataloader` parameter evaluates the calibrated and quantized model; the evaluation metrics are specified in the yaml configuration file. The evaluation tells the tuner if the quantized model meets accuracy criteria. If it does not, the tuner starts a new calibration and tuning flow. For this usage, the `model`, `q_dataloader`, and `eval_dataloader` parameters are mandatory.

### Most concise template-based yaml + few lines of code changes

The third usage is designed for ease of tuning enablement for models with custom metric evaluations or for metrics not yet supported by Intel® Low Precision Optimization Tool. Currently, this usage model works for object detection and NLP networks.

The user constructs the calibration dataloader by code and passes it to the `q_dataloader` parameter. This usage is similar to the second usage; the difference is that the user specifies a custom `eval_func` parameter that encapsulates the evaluation dataset and evaluation process by `self`. The FP32 and quantized INT8 model is evaluated by `eval_func`. The `eval_func` yields a higher-is-better accuracy value to the tuner; the tuner checks to see if the quantized model meets the accuracy criteria. If it does not, the tuner starts a new calibration and tuning flow. For this usage, the `model`, `q_dataloader` and `eval_func` parameters are mandatory.

