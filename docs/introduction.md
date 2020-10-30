Introduction
=========================================

Intel® Low Precision Optimization Tool is an open source python library to help users to fast deploy low-precision inference solution on popular DL frameworks including TensorFlow, PyTorch, MxNet etc. It automatically optimizes low-precision recipes for deep learning models to achieve optimal product objectives like inference performance and memory usage with expected accuracy criteria.


# User facing API

The API is intented to unify the low precision quantization interfaces cross multiple DL frameworks for best out-of-box experiences.

It consists of three components:

### quantization related APIs
```
class Quantization(object):
    def __init__(self, conf_fname):
        ...
    
    def __call__(self, model, q_dataloader=None, q_func=None, eval_dataloader=None, eval_func=None):
        ...

```

### pruning related APIs
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

### benchmarking related APIs
```
class Benchmark(object):
    def __init__(self, conf_fname):
        ...

    def __call__(self, model, b_dataloader=None, b_func=None):
        ...
```

The conf_fname parameter used in above class initialization is a path to Intel® Low Precision Optimization Tool configuration file, which is a yaml file and used to control the whole tuning behavior.

# YAML Syntax

Intel® Low Precision Optimization Tool provides three template yaml files for [PTQ](../ilit/template/ptq.yaml), [QAT](../ilit/template/qat.yaml), [Pruning](../ilit/template/pruning.yaml) scenarios. User could refer to this complete template to understand the meaning of each fields.

> Most of fields in yaml template is optional and a typical yaml file needed is very concise. for example, [HelloWorld Yaml](../examples/helloworld/tf2.x/conf.yaml)

# How to use the quantization API

Intel® Low Precision Optimization Tool supports three different usages replying on how user code orgnized:

### *Template-based yaml setting + 3 lines code changes*

This first usage is designed for minimal code changes when integrating with Intel® Low Precision Optimization Tool. All calibration and evaluation process is constructed by yaml, including dataloaders used in calibration and evaluation phases and quantization tuning settings. For this usage, only model parameter is mandotory.

Examples of this usage are at [TensorFlow Classification Models](../examples/tensorflow/image_recognition/README.md).

### *Concise template-based yaml + few lines code changes*

The second usage is designed for concise yaml configuration by moving calibration and evaluation dataloader construction from yaml to code. If user model is using ilit supported evaluation metrics, this usages will be a good choice.
 
user need provide a *dataloader* implemented __iter__ or __getitem__ methods and batch_size attribute, which usually have existed or easily develop in user code. ilit also provides built-in dataloaders to support dynamic batching, user can implement a *dataset* implemented __iter__ or __getitem__ methods to yield one single batch. Quanitzation().dataloader() will take this dataset as input parameter to construct ilit dataloader.    

After that, User specifies fp32 "model", calibration dataset "q_dataloader" and evaluation dataset "eval_dataloader". The calibrated and quantized model is evaluated with "eval_dataloader" with evaluation metrics specified in the yaml configuration file. The evaluation tells the tuner whether the quantized model meets the accuracy criteria. If not, the tuner starts a new calibration and tuning flow. For this usage, model, q_dataloader and eval_dataloader parameters are mandotory.

### *Most concise template-based yaml + few lines code changes*
   
The third usage is designed for ease of tuning enablement for models with custom metric evaluation or metrics not supported by Intel® Low Precision Optimization Tool yet. Currently this usage model works for object detection and NLP networks.

User constructs calibration dataloader by code and pass to "q_dataloader" parameter. This usage is quite similar with the second usage, just user specifies a custom "eval_func" which encapsulates the evaluation dataset and evaluation process by self. The FP32 and quantized INT8 model is evaluated with "eval_func". The "eval_func" yields a higher-is-better accuracy value to the tuner, the tuner will check whether the quantized model meets the accuracy criteria. If not, the Tuner starts a new calibration and tuning flow. For this usage, model, q_dataloader and eval_func parameters are mandotory

