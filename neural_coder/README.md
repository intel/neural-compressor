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

There are currently 3 ways to use Neural Coder for automatic quantization enabling and benchmark.

### Jupyter Lab Extension
We offer Neural Coder as an extension plugin in Jupyter Lab. This enables users to utilize Neural Coder while writing their Deep Learning models in Jupyter Lab coding platform. Users can simply search for ```jupyter-lab-neural-compressor``` in the Extension Manager in JupyterLab and install Neural Coder with one click. For more details, please refer to this [guide](extensions/neural_compressor_ext_lab/README.md).

[AWS Amazon SageMaker](https://aws.amazon.com/sagemaker/) users can also use Neural Coder as an extension following this [guide](docs/AWSSageMakerSupport.md).

### Python Launcher
Neural Coder can be used as a Python Launcher. Users can run the Python Deep Learning model code as it is with automatic enabling of optimizations by simply adding an inline prefix ```-m neural_coder``` to the Python command line. For more details, please refer to this [guide](docs/PythonLauncher.md).

### Python API
There are 3 user-facing APIs for Neural Coder: enable, bench and superbench. For more details, please refer to this [guide](docs/PythonAPI.md). We have provided a [list](docs/SupportMatrix.md) of supported Deep Learning optimization features. Specifically for quantization, we provide an auto-quantization API that helps automatically enable quantization on Deep Learning models and automatically evaluates for the best performance on the model with no manual coding needed. Supported features include Post-Training Static Quantization, Post-Training Dynamic Quantization, and Mixed Precision. For more details, please refer to this [guide](docs/Quantization.md).

## Contact
Please contact us at [inc.maintainers@intel.com](mailto:inc.maintainers@intel.com) for any Neural Coder related question.
