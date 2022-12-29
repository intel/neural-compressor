DataLoader
==========

1. [Introduction](#introduction)

2. [Supported Framework Dataloader Matrix](#supported-framework-dataloader-matrix)

3. [Get Start with Dataloader API](#get-start-with-dataloader-api)

4. [Examples](#examples)

## Introduction

Deep Learning often encounters large datasets that are memory-consuming. Previously, working with large datasets required loading them into memory all at once. The constant lack of memory resulted in the need for an efficient data generation scheme. This is not only about handling the lack of memory in large datasets, but also about making the process of loading data faster using a multi-processing thread. We call the data generation object a DataLoader.

With the importance of a dataloader, different frameworks can have their own DataLoader module. As for Intel® Neural Compressor, it has implemented an internal dataloader and provides a unified DataLoader API for the following three reasons:

- The framework-specific dataloader has different features and APIs that will make it hard to use them same way in Neural Compressor.

- Neural Compressor treats batch size as a tuning parameter which means it can dynamically change the batch size to reach the accuracy goal.

- Internal dataloader makes it easy to config dataloaders in a yaml file without any code modification.

The internal dataloader takes a [dataset](./dataset.md) as the input parameter and loads data from the dataset when needed. In special cases, users can also define their own dataloader classes, which must have `batch_size` attribute and `__iter__` function.

## Supported Framework Dataloader Matrix

| Framework     | Status     |
|---------------|:----------:|
| TensorFlow    |  &#10004;  |
| PyTorch       |  &#10004;  |
| ONNX Runtime   |  &#10004;  |
| MXNet         |  &#10004;  |

## Get Start with Dataloader API

### Config Dataloader in a Yaml File

Users can use internal dataloader in the following manners. In this case, the dataloader is created after the Quantization object is initialized. As calibration and evaluation may have different transforms and datasets, users can config different dataloaders in a yaml file.

```yaml
quantization:
  approach: post_training_static_quant
  calibration:
    dataloader:
      dataset:
        COCORaw:
          root: /path/to/calibration/dataset
      filter:
        LabelBalance:
          size: 1
      transform:
        Resize:
          size: 300

evaluation:
  accuracy:
    metric: 
      ...
    dataloader:
      batch_size: 16
      dataset:
        COCORaw:
          root: /path/to/evaluation/dataset
      transform:
        Resize:
          size: 300
  performance:
    dataloader:
      batch_size: 16
      dataset:
        dummy_v2:
          input_shape: [224, 224, 3] 
```

### Create a User-specific Dataloader

Users can define their own dataloaders as shown as below:

```python

class Dataloader:
    def __init__(self, batch_size, **kwargs):
        self.batch_size = batch_size
        self.dataset = []
        # operations to add (input_data, label) pairs into self.dataset

    def __iter__(self):
        for input_data, label in self.dataset:
            yield input_data, label

from neural_compressor import quantization, PostTrainingQuantConfig
config = PostTrainingQuantConfig()
dataloader = Dataloader(batch_size, **kwargs)
q_model = quantization.fit(model, config, calib_dataloader=dataloader,
    eval_func=eval)
q_model.save(args.output_model)
```

## Examples

- Refer to this [example](https://github.com/intel/neural-compressor/tree/master/examples/onnxrt/body_analysis/onnx_model_zoo/ultraface/quantization/ptq) for how to define a customised dataloader.

- Refer to this [example](https://github.com/intel/neural-compressor/tree/v1.14.2/examples/onnxrt/image_recognition/resnet50/quantization/ptq) for how to use internal dataloader.
