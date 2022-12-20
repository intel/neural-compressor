Export
=====

1. [Introduction](#introduction)

2. [Supported Framework Model Matrix](#supported-framework-model-matrix)

3. [Examples](#examples)

4. [Appendix](#appendix)

# Introduction
Open Neural Network eXchange (ONNX) is an open standard format for representing machine learning models. Exporting FP32 PyTorch/Tensorflow models has become popular and easy to use. However, for Intel Neural Compressor, we hope to export the int8 model to ONNX to achieve higher applicability on multiple frameworks.

Here we briefly introduce our export API for PyTorch FP32/INT8 models. First, the int8 ONNX model is not directly exported from the int8 PyTorch model, but quantized after obtaining the FP32 ONNX model using the mature torch.onnx.export API. To ensure that the quantization process of ONNX is as consistent as possible with PyTorch, we reuse three key pieces of information from the Neural Compressor model to perform ONNX quantization.

 - Quantized operations: Only operations quantized in PyTorch will be quantized in ONNX.
 - Scale info: Scale information is collected from the PyTorch quantization process.
 - Weights of quantization aware training(QAT): for quantization-aware training, the updated weights are passed to the ONNX model.

<a target="_blank" href="./_static/imgs/export.png" text-align:center>
    <center> 
        <img src="./_static/imgs/export.png" alt="Architecture" width=650 height=200> 
    </center>
</a>

# Supported Framework Model Matrix

| Export | Post-training Dynamic Quantization | Post-training Static Quantization | Quantization Aware Training |
| :---: | :---: | :---: | :---: |
| FP32 PyTorch Model -> FP32 ONNX Model | &#10004; | &#10004; | &#10004; |
| INT8 PyTorch Model -> INT8 QDQ ONNX Model | &#10004; | &#10004; | &#10004; |
| INT8 PyTorch Model -> INT8 QLinear ONNX Model | &#10004; | &#10004; | &#10004; |

# Examples

## FP32 Model Export
```python
from neural_compressor.experimental.common import Model
from neural_compressor.config import Torch2ONNXConfig
inc_model = Model(model)
fp32_onnx_config = Torch2ONNXConfig(
    dtype="fp32",
    example_inputs=torch.randn(1, 3, 224, 224),
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={"input": {0: "batch_size"},
                    "output": {0: "batch_size"}},
)
inc_model.export('fp32-model.onnx', fp32_onnx_config)
```

## INT8 Model Export

```python
# q_model is a Neural Compressor model after performing quantization.
from neural_compressor.config import Torch2ONNXConfig
int8_onnx_config = Torch2ONNXConfig(
    dtype="int8",
    opset_version=14,
    quant_format="QDQ", # or QLinear
    example_inputs=torch.randn(1, 3, 224, 224),
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={"input": {0: "batch_size"},
                    "output": {0: "batch_size"}},
)
q_model.export('int8-model.onnx', int8_onnx_config)
```

# Appendix

> TODO: Add descriptions for three recipes