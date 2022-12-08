# Dynamic Quantization

> **Note**

> Dynamic quantization currently supports the pytorch and onnxruntime backend.

As a [quantization](./Quantization.md) class, dynamic quantization allows users to determine the scale factor for activations dynamically based on the data range that's observed at runtime as opposed to using other methods that entail multiplying a float point value by some scale factor and then rounding the result to a whole number. As noted in [PyTorch documentation][PyTorch-Dynamic-Quantization], this "ensures that the scale factor is 'tuned' so that as much signal as possible about each observed dataset is preserved." Since it does not use a lot of tuning parameters, dynamic quantization is a good match for NLP models.


The pytorch bert_base model provides an example where users can create a specific quantization method like the following yaml:


```yaml
model:                                               # mandatory. used to specify model specific information.
  name: bert 
  framework: pytorch                                 # mandatory. possible values are pytorch, onnxrt_integerops.

quantization:
  approach: post_training_dynamic_quant              # optional. default value is post_training_static_quant

tuning:
  accuracy_criterion:
    relative:  0.01                                  # optional. default value is relative, other value is absolute. this example allows relative accuracy loss: 1%.
  exit_policy:
    timeout: 0                                       # optional. tuning timeout (seconds). default value is 0 which means early stop. combine with max_trials field to decide when to exit.
  random_seed: 9527                                  # optional. random seed for deterministic tuning.
```
[PyTorch-Dynamic-Quantization]: https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html
