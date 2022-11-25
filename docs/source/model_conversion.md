Model Conversion 
==================

## Introduction

Model conversion is used to convert different TensorFlow model format to another. 

Now it supports converting a Keras model to QAT(quantization aware training) model. In the future, we will add tflite to default model support.

## How to use it
Currently, model conversion can be used in q_func during quantzation process of QAT.
See the following example which demonstrate model conversion API usage.

```python
    from neural_compressor.experimental import ModelConversion, common
    conversion = ModelConversion()
    q_model = conversion.fit('/path/to/trained/saved_model')
    q_model.save('/path/to/quantized/saved_model')
```
