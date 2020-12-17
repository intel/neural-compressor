## Quantization

Quantization refers to processes that enable lower precision inference and training by performing computations at fixed point integers that are lower than floating points. This often leads to smaller model sizes and faster inference time. Quantization is particularly useful in deep learning inference and training, where moving data more quickly and reducing bandwidth bottlenecks is optimal. Intel is actively working on techniques that use lower numerical precision by using training with 16-bit multipliers and inference with 8-bit or 16-bit multipliers. Refer to the Intel article on [lower numerical precision inference and training in deep learning](https://software.intel.com/content/www/us/en/develop/articles/lower-numerical-precision-deep-learning-inference-and-training.html).

Quantization methods include the following three classes:

* Post-Training Quantization (PTQ)
* Quantization-Aware Training (QAT)
* Dynamic Quantization

Intel® Low Precision Optimization Tool currently supports PTQ and QAT. Using MobileNetV2 as an example, this document provides tutorials for both. It also provides helper functions for evaluation.

Dynamic Quantization currently is only supported with onnxruntime backend, please refer to [dynamic quantization](./dynamic_quantization.md) for details.

>Note: These quantization tutorials use [PyTorch examples](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#model-architecture) as allowed by PyTorch's [License](https://github.com/pytorch/pytorch/blob/master/LICENSE). Refer to [PyTorch](https://github.com/pytorch/tutorials/blob/master/advanced_source/static_quantization_tutorial.py) for updates.


* See also [PTQ](PTQ.md)
* See also [QAT](QAT.md)
