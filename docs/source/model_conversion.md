Model Conversion 
==================

## Introduction

Model conversion is used to convert different TensorFlow model format to another. 

Now it supports QAT(quantization aware training) model to default(quantized) model. In the future, we will add tflite to default model support.

## How to use it

See the following example which demonstrate model conversion API usage.

```python
    from neural_compressor.experimental import ModelConversion, common
    conversion = ModelConversion()
    conversion.source = 'QAT'
    conversion.destination = 'default'
    conversion.model = '/path/to/trained/saved_model'
    q_model = conversion()
    q_model.save('/path/to/quantized/saved_model')
```

After this conversion is done, user could measure the accuracy or performance on quantized model.
  ```python
    from neural_compressor.experimental import Benchmark, common
    evaluator = Benchmark('/path/to/yaml')
    evaluator.model = '/path/to/quantized/saved_model'
    evaluator.b_dataloader = ...       # create benchmark dataloader like examples/tensorflow/qat/benchmark.py
    evaluator('accuracy')
  ```
