DataLoader
==========

1. [Introduction](#introduction)

2. [Supported Framework Dataloader Matrix](#supported-framework-dataloader-matrix)

3. [Get Started with Dataloader](#get-started-with-dataloader)

    3.1 [Use Intel® Neural Compressor DataLoader API](#use-intel-neural-compressor-dataloader-api)

    3.2 [Build Custom Dataloader with Python API](#build-custom-dataloader-with-python-api)

4. [Examples](#examples)

## Introduction

Deep Learning often encounters large datasets that are memory-consuming. Previously, working with large datasets required loading them into memory all at once. The constant lack of memory resulted in the need for an efficient data generation scheme. This is not only about handling the lack of memory in large datasets, but also about making the process of loading data faster using a multi-processing thread. We call the data generation object a DataLoader.

With the importance of a dataloader, different frameworks can have their own DataLoader module. As for Intel® Neural Compressor, it implements an internal dataloader and provides a unified `DataLoader` API for the following two reasons:

- The framework-specific dataloader has different features and APIs that will make it hard to use them same way in Neural Compressor.

- Neural Compressor treats batch size as a tuning parameter which means it can dynamically change the batch size to reach the accuracy goal.

The unified  `DataLoader` API takes a [dataset](./dataset.md) as the input parameter and loads data from the dataset when needed. In special cases, users can also define their own dataloader classes, which must have `batch_size` attribute and `__iter__` function.

Of cause, users can also use frameworks own dataloader in Neural Compressor.

## Supported Framework Dataloader Matrix

| Framework     | Status     |
|---------------|:----------:|
| TensorFlow    |  &#10004;  |
| Keras         |  &#10004;  |
| PyTorch       |  &#10004;  |
| ONNX Runtime   |  &#10004;  |

## Get Started with DataLoader

### Use Intel® Neural Compressor DataLoader API

Acceptable parameters for `DataLoader` API including:

| Parameter     | Description     |
|---------------|:----------:|
|framework (str)| different frameworks, such as `tensorflow`, `tensorflow_itex`, `keras`, `pytorch` and `onnxruntime`.|
|dataset (object)| A dataset object from which to get data. Dataset must implement `__iter__` or `__getitem__` method.|
|batch_size (int, optional)| How many samples per batch to load. Defaults to 1.|
|collate_fn (Callable, optional)| Callable function that processes the batch you want to return from your dataloader. Defaults to None.|
|last_batch (str, optional)| How to handle the last batch if the batch size does not evenly divide by the number of examples in the dataset. 'discard': throw it away. 'rollover': insert the examples to the beginning of the next batch.Defaults to 'rollover'.|
|sampler (Iterable, optional)| Defines the strategy to draw samples from the dataset.Defaults to None.|
|batch_sampler (Iterable, optional)| Returns a batch of indices at a time. Defaults to None.|
|num_workers (int, optional)| how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Defaults to 0.|
|pin_memory (bool, optional)| If True, the data loader will copy Tensors into device pinned memory before returning them. Defaults to False.|
|shuffle (bool, optional)| Set to ``True`` to have the data reshuffled at every epoch. Defaults to False.|
|distributed (bool, optional)| Set to ``True`` to support distributed computing. Defaults to False.|

Users can use the unified `DataLoader` API in the following manners.

```python
from neural_compressor.data import DataLoader
from neural_compressor import quantization, PostTrainingQuantConfig

dataloader = DataLoader(framework="tensorflow", dataset=dataset)
config = PostTrainingQuantConfig()
q_model = quantization.fit(model, config, calib_dataloader=dataloader, eval_func=eval)
```
> Note: `DataLoader(framework='onnxruntime', dataset=dataset)` failed in neural-compressor v2.2. We have fixed it in this [PR](https://github.com/intel/neural-compressor/pull/1048).

### Build Custom Dataloader with Python API

Users can define their own dataloaders as shown as below:

```python
class NewDataloader:
    def __init__(self, batch_size, **kwargs):
        self.batch_size = batch_size
        self.dataset = []
        # operations to add (input_data, label) pairs into self.dataset

    def __iter__(self):
        for input_data, label in self.dataset:
            yield input_data, label


from neural_compressor import quantization, PostTrainingQuantConfig

config = PostTrainingQuantConfig()
dataloader = NewDataloader(batch_size, **kwargs)
q_model = quantization.fit(model, config, calib_dataloader=dataloader, eval_func=eval)
```

## Examples

- Refer to this [example](https://github.com/intel/neural-compressor/blob/master/examples/onnxrt/body_analysis/onnx_model_zoo/ultraface/quantization/ptq_static) for how to define a customised dataloader.

- Refer to this [example](https://github.com/intel/neural-compressor/blob/master/examples/onnxrt/nlp/bert/quantization/ptq_static) for how to use internal dataloader.
