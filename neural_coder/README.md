Neural Coder
===========================
## What do we offer?

Neural Coder is a novel deployment toolkit for one-click acceleration on Deep Learning scripts via performing automated code insertions of CUDA to CPU platform conversions and Deep Learning optimization APIs. Subsequently, Neural Coder can perform automated benchmark on all applicable optimization sets acquired from the automated enabling, and evaluate for the best out-of-box performance.

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
We currently provide 3 user-facing APIs: enable, bench and superbench.
#### Enable
Users can use ```enable()``` to enable specific features into DL scripts:
```
from neural_coder import enable
enable(code="examples/vision/resnet50.py", 
       features=["pytorch_jit_script", "pytorch_channels_last"])
```
To run benchmark directly on the optimization together with the enabling:
```
from neural_coder import enable
enable(code="examples/vision/resnet50.py", 
       features=["pytorch_jit_script", "pytorch_channels_last"], 
       run_bench=True, 
       mode="throughput")
```
#### Bench
To run benchmark on your code with an existing patch:
```
from neural_coder import bench
bench(code="examples/vision/resnet50.py",
      patch_path="${your_patch_path}",
      mode="throughput")
```
#### SuperBench
To sweep on optimization sets with a fixed benchmark configuration:
```
from neural_coder import superbench
superbench(code="examples/vision/resnet50.py", 
           sweep_objective="feature", 
           mode="throughput")
```
To sweep on benchmark configurations for a fixed optimization set:
```
from neural_coder import superbench
superbench(code="examples/vision/resnet50.py", 
           sweep_objective="bench_config", 
           bench_feature=["pytorch_jit_script"，"pytorch_channels_last"])
```

## Contact
Please contact us at [kai.yao@intel.com](mailto:kai.yao@intel.com) for any Neural Coder related question.
