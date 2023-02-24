User Guide
===========================

Intel® Neural Compressor aims to provide popular model compression techniques such as quantization, pruning (sparsity), distillation, and neural architecture search on mainstream frameworks such as [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), [ONNX Runtime](https://onnxruntime.ai/), and [MXNet](https://mxnet.apache.org/),
as well as Intel extensions such as [Intel Extension for TensorFlow](https://github.com/intel/intel-extension-for-tensorflow) and [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch). 
In addition, the tool showcases the key features, typical examples, and broad collaborations as below:

* Support a wide range of Intel hardware such as [Intel Xeon Scalable processor](https://www.intel.com/content/www/us/en/products/details/processors/xeon/scalable.html), [Intel Xeon CPU Max Series](https://www.intel.com/content/www/us/en/products/details/processors/xeon/max-series.html), [Intel Data Center GPU Flex Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/flex-series.html), and [Intel Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series.html) with extensive testing; support AMD CPU, ARM CPU, and NVidia GPU through ONNX Runtime with limited testing

* Validate more than 10,000 models such as [Stable Diffusion](https://github.com/intel/neural-compressor/tree/readme/update/examples/pytorch/nlp/huggingface_models/text-to-image/quantization), [GPT-J](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_static/fx), [BERT-Large](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/fx), and [ResNet50](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/fx) from popular model hubs such as [Hugging Face](https://huggingface.co/), [Torch Vision](https://pytorch.org/vision/stable/index.html), and [ONNX Model Zoo](https://github.com/onnx/models#models), by leveraging zero-code optimization solution [Neural Coder](https://github.com/intel/neural-compressor/tree/readme/update/neural_coder#what-do-we-offer) and automatic [accuracy-driven](https://github.com/intel/neural-compressor/blob/readme/update/docs/source/design.md#workflow) quantization strategies

* Collaborate with cloud marketplace such as [Google Cloud Platform](https://console.cloud.google.com/marketplace/product/bitnami-launchpad/inc-tensorflow-intel?project=verdant-sensor-286207), [Amazon Web Services](https://aws.amazon.com/marketplace/pp/prodview-yjyh2xmggbmga#pdp-support), and [Azure](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/bitnami.inc-tensorflow-intel), software platforms such as [Alibaba Cloud](https://www.intel.com/content/www/us/en/developer/articles/technical/quantize-ai-by-oneapi-analytics-on-alibaba-cloud.html) and [Tencent TACO](https://new.qq.com/rain/a/20221202A00B9S00), and open AI ecosystem such as [Hugging Face](https://huggingface.co/blog/intel), [PyTorch](https://pytorch.org/tutorials/recipes/intel_neural_compressor_for_pytorch.html), [ONNX](https://github.com/onnx/models#models), and [Lightning AI](https://github.com/Lightning-AI/lightning/blob/master/docs/source-pytorch/advanced/post_training_quantization.rst)

## Documentation
The below documents could help you to get familiar with concepts and modules in Intel® Neural Compressor. Learn how to utilize the APIs in Intel® Neural Compressor to conduct quantization, pruning (sparsity), distillation, and neural architecture search on mainstream frameworks.

The user can get a quick understand about design structure and workflow of Intel® Neural Compressor in the `Overview` part. This part has provided a wide range examples to help the user to get a quick start of using Intel® Neural Compressor. `Python-based APIs` contains more details about the functional APIs in Intel® Neural Compressor, which introduces the mechanism of each function and tutorial to help the user write their own code step by step. `Neural Coder` show our special Python launcher to help the user to achieve automatic deep learning optimization. If you're familiar with our Intel® Neural Compressor, you can find more advanced usages to help you optimize your code and improve the model performance in `Advanced Topics`. We also provide a comprehensive migration document to help the user update their code from our previous 1.X version to the new 2.X version. 

<table class="docutils">
  <thead>
  <tr>
    <th colspan="9">Overview</th>
  </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="4" align="center"><a href="./docs/source/design.md#architecture">Architecture</a></td>
      <td colspan="3" align="center"><a href="./docs/source/design.md#workflow">Workflow</a></td>
      <td colspan="1" align="center"><a href="https://intel.github.io/neural-compressor/latest/docs/source/api-doc/apis.html">APIs</a></td>
      <td colspan="1" align="center"><a href="./docs/source/bench.md">GUI</a></td>
    </tr>
    <tr>
      <td colspan="2" align="center"><a href="./examples#notebook-examples">Notebook</a></td>
      <td colspan="1" align="center"><a href="./examples">Examples</a></td>
      <td colspan="1" align="center"><a href="./docs/source/validated_model_list.md">Results</a></td>
      <td colspan="5" align="center"><a href="https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html">Intel oneAPI AI Analytics Toolkit</a></td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th colspan="9">Python-based APIs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td colspan="2" align="center"><a href="./docs/source/quantization.md">Quantization</a></td>
        <td colspan="3" align="center"><a href="./docs/source/mixed_precision.md">Advanced Mixed Precision</a></td>
        <td colspan="2" align="center"><a href="./docs/source/pruning.md">Pruning (Sparsity)</a></td> 
        <td colspan="2" align="center"><a href="./docs/source/distillation.md">Distillation</a></td>
    </tr>
    <tr>
        <td colspan="2" align="center"><a href="./docs/source/orchestration.md">Orchestration</a></td>        
        <td colspan="2" align="center"><a href="./docs/source/benchmark.md">Benchmarking</a></td>
        <td colspan="3" align="center"><a href="./docs/source/distributed.md">Distributed Compression</a></td>
        <td colspan="3" align="center"><a href="./docs/source/export.md">Model Export</a></td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th colspan="9">Neural Coder (Zero-code Optimization)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td colspan="1" align="center"><a href="./neural_coder/docs/PythonLauncher.md">Launcher</a></td>
        <td colspan="2" align="center"><a href="./neural_coder/extensions/neural_compressor_ext_lab/README.md">JupyterLab Extension</a></td>
        <td colspan="3" align="center"><a href="./neural_coder/extensions/neural_compressor_ext_vscode/README.md">Visual Studio Code Extension</a></td>
        <td colspan="3" align="center"><a href="./neural_coder/docs/SupportMatrix.md">Supported Matrix</a></td>
    </tr>    
  </tbody>
  <thead>
      <tr>
        <th colspan="9">Advanced Topics</th>
      </tr>
  </thead>
  <tbody>
      <tr>
          <td colspan="1" align="center"><a href="./docs/source/adaptor.md">Adaptor</a></td>
          <td colspan="2" align="center"><a href="./docs/source/tuning_strategies.md">Strategy</a></td>
          <td colspan="3" align="center"><a href="./docs/source/distillation_quantization.md">Distillation for Quantization</a></td>
          <td colspan="3" align="center">SmoothQuant (Coming Soon)</td>
      </tr>
      <tr>
        <td colspan="3" align="center"><a href="./docs/source/metric.md">Metric</a></td>        
        <td colspan="3" align="center"><a href="./docs/source/objective.md">Objective</a></td>
        <td colspan="3" align="center"><a href="./docs/source/migration.md">Code Migration</a></td>
      </tr>
  </tbody>
</table>

