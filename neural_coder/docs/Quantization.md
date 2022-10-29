Neural Coder for Quantization
===========================
This feature helps automatically enable quantization on Deep Learning models and automatically evaluates for the best performance on the model. It is a code-free solution that can help users enable quantization algorithms on a model with no manual coding needed. Supported features include Post-Training Static Quantization, Post-Training Dynamic Quantization, and Mixed Precision.


## Features Supported
- Post-Training Static Quantization for [Stock PyTorch](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html) (with FX backend)
- Post-Training Static Quantization for [IPEX](https://github.com/intel/intel-extension-for-pytorch/blob/v1.12.0/docs/tutorials/features/int8.md)
- Post-Training Dynamic Quantization for [Stock PyTorch](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html)
- Mixed Precision for [Stock PyTorch](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)

## Models Supported
- HuggingFace [Transformers](https://github.com/huggingface/transformers) models
- [torchvision](https://pytorch.org/vision/stable/index.html) models
- Broad models (under development)

## Usage
- PyPI distribution with a one-line API call
- [JupyterLab extension](../extensions/neural_compressor_ext_lab/README.md)

## Example
### PyPI distribution:
HuggingFace [Transformers](https://github.com/huggingface/transformers) models: [text-classification/run_glue.py](https://github.com/huggingface/transformers/blob/v4.21-release/examples/pytorch/text-classification/run_glue.py)
```
from neural_coder import auto_quant
auto_quant(
    code="https://github.com/huggingface/transformers/blob/v4.21-release/examples/pytorch/text-classification/run_glue.py",
    args="--model_name_or_path albert-base-v2 --task_name sst2 --do_eval --output_dir result",
)
```

[torchvision](https://pytorch.org/vision/stable/index.html) models: [imagenet/main.py](https://github.com/pytorch/examples/blob/main/imagenet/main.py)
```
from neural_coder import auto_quant
auto_quant(
    code="https://github.com/pytorch/examples/blob/main/imagenet/main.py",
    args="-a alexnet --pretrained -e /path/to/imagenet/",
)
```
