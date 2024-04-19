<div align="center">

Intel® Neural Compressor
===========================
<h3> An open-source Python library supporting popular model compression techniques on ONNX Runtime</h3>

[![python](https://img.shields.io/badge/python-3.8%2B-blue)](https://github.com/intel/neural-compressor)
[![version](https://img.shields.io/badge/release-2.5-green)](https://github.com/intel/neural-compressor/releases)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/intel/neural-compressor/blob/master/LICENSE)
[![coverage](https://img.shields.io/badge/coverage-85%25-green)](https://github.com/intel/neural-compressor)
[![Downloads](https://static.pepy.tech/personalized-badge/neural-compressor?period=total&units=international_system&left_color=grey&right_color=green&left_text=downloads)](https://pepy.tech/project/neural-compressor)



---
<div align="left">

Intel® Neural Compressor aims to provide popular model compression techniques such as quantization on [ONNX Runtime](https://onnxruntime.ai/).
In particular, the tool provides the key features, typical examples, and open collaborations as below:

* Support a wide range of Intel hardware such as [Intel Xeon Scalable Processors](https://www.intel.com/content/www/us/en/products/details/processors/xeon/scalable.html), [Intel Xeon CPU Max Series](https://www.intel.com/content/www/us/en/products/details/processors/xeon/max-series.html); support AMD CPU, ARM CPU, and NVidia GPU with limited testing

* Validate popular LLMs such as [LLama2](./examples/onnxrt/nlp/huggingface_model/text_generation/llama/) and broad models such as [BERT-base](./examples/onnxrt/nlp/onnx_model_zoo/bert-squad/), and [ResNet50](./examples/onnxrt/image_recognition/onnx_model_zoo/resnet50/) from popular model hubs such as [Hugging Face](https://huggingface.co/), [ONNX Model Zoo](https://github.com/onnx/models#models), by leveraging automatic [accuracy-driven](./docs/source/design.md#workflow) quantization strategies

* Collaborate with software platforms such as [Microsoft Olive](https://github.com/microsoft/Olive), and open AI ecosystem such as [Hugging Face](https://huggingface.co/blog/intel), [ONNX](https://github.com/onnx/models#models) and [ONNX Runtime](https://github.com/microsoft/onnxruntime)

## Installation

### Install from pypi
```Shell
pip install neural-compressor
```
> **Note**: 
> Further installation methods can be found under [Installation Guide](./docs/source/installation_guide.md).

## Getting Started

Setting up the environment:  
```bash
pip install "neural-compressor>=2.3" onnxruntime onnx
```
After successfully installing these packages, try your first quantization program.

### Weight-Only Quantization (LLMs)
Following example code demonstrates Weight-Only Quantization on LLMs, it supports Intel CPU, Nvidia GPU, best device will be selected automatically. 

Run the example:
```python
from neural_compressor_ort.quantization import matmul_nbits_quantizer
algo_config = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig()
quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
    model,
    n_bits=4,
    block_size=32,
    is_symmetric=True,
    algo_config=algo_config,
)
quant.process()
best_model = quant.model
```   

### Static Quantization

```python
from neural_compressor_ort.quantization import quantize, StaticQuantConfig
from neural_compressor_ort.quantization.calibrate import CalibrationDataReader

class DataReader(CalibrationDataReader):
    def __init__(self):
        self.encoded_list = []
        # append data into self.encoded_list

        self.iter_next = iter(self.encoded_list)

    def get_next(self):
        return next(self.iter_next, None)

    def rewind(self):
        self.iter_next = iter(self.encoded_list)

data_reader = DataReader()
config = StaticQuantConfig(
    calibration_data_reader=data_reader
)
quantize(model, output_model_path, config)
```

## Documentation

<table class="docutils">
  <thead>
  <tr>
    <th colspan="8">Overview</th>
  </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="3" align="center"><a href="./docs/source/design.md#architecture">Architecture</a></td>
      <td colspan="3" align="center"><a href="./docs/source/design.md#workflow">Workflow</a></td>
      <td colspan="3" align="center"><a href="examples/README.md">Examples</a></td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th colspan="8">Feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td colspan="4" align="center"><a href="./docs/source/quantization.md">Quantization</a></td>
          <td colspan="4" align="center"><a href="./docs/source/smooth_quant.md">SmoothQuant</td>
      <tr>
          <td colspan="4" align="center"><a href="./docs/source/quantization_weight_only.md">Weight-Only Quantization (INT8/INT4) </td>
           </td>
          <td colspan="4" align="center"><a href="./docs/source/quantization_layer_wise.md">Layer-Wise Quantization </td>
      </tr>
  </tbody>
</table>



## Additional Content

* [Contribution Guidelines](./docs/source/CONTRIBUTING.md)
* [Legal Information](./docs/source/legal_information.md)
* [Security Policy](SECURITY.md)

## Communication 
- [GitHub Issues](https://github.com/intel/neural-compressor/issues): mainly for bug reports, new feature requests, question asking, etc.
- [Email](mailto:inc.maintainers@intel.com): welcome to raise any interesting research ideas on model compression techniques by email for collaborations.  