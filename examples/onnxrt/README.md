ONNX model quantization
======

Currently Neural Compressor supports dynamic and static quantization for onnx models.

## Dynamic quantization

Dynamic quantization calculates the quantization parameter (scale and zero point) for activations dynamically.


### How to use

Users can use dynamic quantization method by these ways:

1. Write configuration in yaml file

```yaml
model:
  name: model_name
  framework: onnxrt_integerops

quantization:
  approach: post_training_dynamic_quant

# other items are omitted
```

2. Write configuration with python code

```python
from neural_compressor import conf
from neural_compressor.experimental import Quantization
conf.model.framework = 'onnxrt_integerops'
conf.quantization.approach = 'post_training_dynamic_quant'

quantizer = Quantization(conf)
```

## Static quantization

Static quantization leverages the calibration data to calculates the quantization parameter of activations. There are two ways to represent quantized ONNX models: operator oriented with QLinearOps and tensor oriented (QDQ format).

### How to use

#### Operator oriented with QLinearOps

Users can quantize ONNX models with QLinearOps by these ways:

1. Write configuration in yaml file

```yaml
model:
  name: model_name
  framework: onnxrt_qlinearops

quantization:
  approach: post_training_static_quant

# other items are omitted
```

2. Write configuration with python code

```python
from neural_compressor import conf
from neural_compressor.experimental import Quantization
conf.model.framework = 'onnxrt_qlinearops'
conf.quantization.approach = 'post_training_static_quant'

quantizer = Quantization(conf)
```

#### Tensor oriented (QDQ format)

Users can quantize ONNX models with QDQ format by these ways:

1. Write configuration in yaml file

```yaml
model:
  name: model_name
  framework: onnxrt_qdqops

quantization:
  approach: post_training_static_quant

# other items are omitted
```

2. Write configuration with python code

```python
from neural_compressor import conf
from neural_compressor.experimental import Quantization
conf.model.framework = 'onnxrt_qdqops'
conf.quantization.approach = 'post_training_static_quant'

quantizer = Quantization(conf)
```

