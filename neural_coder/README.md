Neural Coder
===========================
## What do we offer?

Neural Coder is a novel component under Intel® Neural Compressor to further simplify the deployment of deep learning models via one-click automated code changes for device switch (e.g., CUDA to CPU) and optimization enabling. Subsequently, Neural Coder can also perform automated benchmark on all applicable optimization sets acquired from the automated enabling, and evaluate for the best out-of-box performance.

Neural Coder leverages static program analysis techniques and heuristic optimization rules to simplify the usage of various Deep Learning optimization APIs for increasing computation efficiency of AI models and improving user experience for general AI customers. We demonstrate great improvement of developer productivity and aim to facilitate enhanced Deep Learning acceleration adoption via this toolkit.

Neural Coder helps you code Deep Learning optimizations automatically into your scripts. For example, to apply 
- Automatic Mixed Precision (torch.cpu.amp.autocast)
- JIT Script computation graph transformation (torch.jit.script)
- Channels Last memory format transformation (torch.channels_last)

simultaneously on below PyTorch evaluation code, we generate the optimized code in one-click by detecting the correct position to insert the correct API code lines:
```diff
  import torch
  import torchvision.models as models
  my_model = models.resnet50(pretrained=True)
+ import torch
+ with torch.no_grad():
+     my_model = my_model.to(memory_format=torch.channels_last)
+ import torch
+ with torch.no_grad():
+     my_model.eval()
+     my_model = torch.jit.script(my_model)
+     my_model = torch.jit.freeze(my_model)
  my_model.eval()
  batch_size = 112
  input = torch.rand(batch_size, 3, 224, 224)
  with torch.no_grad():
+     import torch
+     with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
          my_model(input)
```

## Getting Started!

### Auto-Quant Feature
We provide a feature named Auto-Quant that helps automatically enable quantization features on a PyTorch model script and automatically evaluates for the best performance on the model. It is a code-free solution that can help users enable quantization algorithms on a PyTorch model with no manual coding needed. Supported features include Post-Training Static Quantization, Post-Training Dynamic Quantization, and Mixed Precision. For more details please refer to this [guide](docs/AutoQuant.md).

### General Guide
We currently provide 3 main user-facing APIs: enable, bench and superbench.
#### Enable
Users can use ```enable()``` to enable specific features into DL scripts:
```
from neural_coder import enable
enable(
    code="neural_coder/examples/vision/resnet50.py",
    features=[
        "pytorch_jit_script",
        "pytorch_channels_last",
    ],
)
```
To run benchmark directly on the optimization together with the enabling:
```
from neural_coder import enable
enable(
    code="neural_coder/examples/vision/resnet50.py",
    features=[
        "pytorch_jit_script",
        "pytorch_channels_last"
    ],
    run_bench=True,
)
```
#### Bench
To run benchmark on your code with an existing patch:
```
from neural_coder import bench
bench(
    code="neural_coder/examples/vision/resnet50.py",
    patch_path="${your_patch_path}",
)
```
#### SuperBench
To sweep on optimization sets with a fixed benchmark configuration:
```
from neural_coder import superbench
superbench(code="neural_coder/examples/vision/resnet50.py")
```
To sweep on benchmark configurations for a fixed optimization set:
```
from neural_coder import superbench
superbench(
    code="neural_coder/examples/vision/resnet50.py",
    sweep_objective="bench_config",
    bench_feature=[
        "pytorch_jit_script",
        "pytorch_channels_last",
    ],
)
```

## Contact
Please contact us at [inc.maintainers@intel.com](mailto:inc.maintainers@intel.com) for any Neural Coder related question.
