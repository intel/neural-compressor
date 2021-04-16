# Dynamic Quantization

### Now only onnxruntime backend support dynamic quantization.

[^1]The key idea with dynamic quantization as described here is that we are going to determine the scale factor for activations dynamically based on the data range observed at runtime. This ensures that the scale factor is “tuned” so that as much signal as possible about each observed dataset is preserved.

Dynamic quantization is relatively free of tuning parameters which makes it well suited to be added into production pipelines as a standard part of NLP models.

Take onnxruntime bert_base model as an example, users can specific quantization method like the following yaml:


```yaml
model:                                               # mandatory. used to specify model specific information.
  name: bert 
  framework: onnxrt_integerops                       # mandatory. possible values are tensorflow, mxnet, pytorch, pytorch_ipex, onnxrt_integerops and onnxrt_qlinearops.

quantization:
  approach: post_training_dynamic_quant              # optional. default value is post_training_static_quant
                                                     # possible value is post_training_static_quant, 
                                                     # post_training_dynamic_quant
                                                     # quant_aware_training                                 
  calibration:
    sampling_size: 8, 16, 32

tuning:
  accuracy_criterion:
    relative:  0.01                                  # optional. default value is relative, other value is absolute. this example allows relative accuracy loss: 1%.
  exit_policy:
    timeout: 0                                       # optional. tuning timeout (seconds). default value is 0 which means early stop. combine with max_trials field to decide when to exit.
  random_seed: 9527                                  # optional. random seed for deterministic tuning.
```
[^1]: https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html
