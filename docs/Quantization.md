Quantization
============

Quantization refers to processes that enable lower precision inference and training by performing computations at fixed point integers that are lower than floating points. This often leads to smaller model sizes and faster inference time. Quantization is particularly useful in deep learning inference and training, where moving data more quickly and reducing bandwidth bottlenecks is optimal. Intel is actively working on techniques that use lower numerical precision by using training with 16-bit multipliers and inference with 8-bit or 16-bit multipliers. Refer to the Intel article on [lower numerical precision inference and training in deep learning](https://software.intel.com/content/www/us/en/develop/articles/lower-numerical-precision-deep-learning-inference-and-training.html).

Quantization methods include the following three classes:

* [Post-Training Quantization (PTQ)](./PTQ.md)
* [Quantization-Aware Training (QAT)](./QAT.md)
* [Dynamic Quantization](./dynamic_quantization.md)

> **Note** 
>
> Dynamic Quantization currently only supports the onnxruntime backend.

