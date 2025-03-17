Infrastructure of Intel® Neural Compressor
=======
1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Supported Feature Matrix](#supported-feature-matrix)

## Introduction
Intel® Neural Compressor provides deep-learning model compression techniques like quantization, knowledge distillation, pruning/sparsity and neural architecture search (NAS). These features are already validated on intel cpu/gpu. Quantized models by Intel® Neural Compressor are validated on broad hardware platforms: amd cpu/arm cpu/nvidia gpu through OnnxRuntime extension provider. Intel® Neural Compressor supports different deep-learning frameworks via unified interfaces, so users can define their own evaluation function to support various models. For quantization, 420+ examples are validated with a performance speedup geomean of 2.2x and up to 4.2x on Intel VNNI. Over 30 pruning and knowledge distillation samples are also available. 

Neural Coder automatically inserts quantization code snippets on a PyTorch model script with one-line API, so this feature can increase the productivity. Intel® Neural Compressor provides other no-code features like GUI etc., so users can do basic optimization by uploading the models and clicking the button. After the optimization, the users will get optimized models and performance results.


## Architecture
<a target="_blank" href="./imgs/architecture.png">
  <img src="./imgs/architecture.png" alt="Architecture" width=914 height=420>
</a>

Intel® Neural Compressor has unified interfaces which dispatch tasks to different frameworks via adaptor layer. The adaptor layer is the bridge between the tuning strategy and vanilla framework quantization APIs. Users can select tuning strategies and the strategy module contains model configs and tuning configs. Model configs define the quantization approach, if it's post-training static quantization, users need to set more parameters like calibration and so on. There are several tuning strategies for users to choose from while the basic strategy is set as default.

## Supported Feature Matrix
[Quantization](quantization.md):
<table class="center">
    <thead>
        <tr>
            <th>Types</th>
            <th>Framework</th>
            <th>Backend</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="3" align="center">Post-Training Static Quantization (PTQ)</td>
            <td align="center">PyTorch</td>
            <td align="center"><a href="https://pytorch.org/docs/stable/quantization.html#eager-mode-quantization">PyTorch Eager</a>/<a href="https://pytorch.org/docs/stable/quantization.html#prototype-fx-graph-mode-quantization">PyTorch FX</a>/<a href="https://github.com/intel/intel-extension-for-pytorch">IPEX</a></td>
        </tr>
        <tr>
            <td align="center">TensorFlow</td>
            <td align="center"><a href="https://github.com/tensorflow/tensorflow">TensorFlow</a>/<a href="https://github.com/Intel-tensorflow/tensorflow">Intel TensorFlow</a></td>
        </tr>
        <tr>
            <td align="center">ONNX Runtime</td>
            <td align="center"><a href="https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/quantize.py">QLinearops/QDQ</a></td>
        </tr>
        <tr>
            <td rowspan="2" align="center">Post-Training Dynamic Quantization</td>
            <td align="center">PyTorch</td>
            <td align="center"><a href="https://pytorch.org/docs/stable/quantization.html#eager-mode-quantization">PyTorch eager mode</a>/<a href="https://pytorch.org/docs/stable/quantization.html#prototype-fx-graph-mode-quantization">PyTorch fx mode</a>/<a href="https://github.com/intel/intel-extension-for-pytorch">IPEX</a></td>
        </tr>
        <tr>
            <td align="center">ONNX Runtime</td>
            <td align="center"><a href="https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/quantize.py">QIntegerops</a></td>
        </tr>  
        <tr>
            <td rowspan="2" align="center">Quantization-aware Training (QAT)</td>
            <td align="center">PyTorch</td>
            <td align="center"><a href="https://pytorch.org/docs/stable/quantization.html#eager-mode-quantization">PyTorch eager mode</a>/<a href="https://pytorch.org/docs/stable/quantization.html#prototype-fx-graph-mode-quantization">PyTorch fx mode</a>/<a href="https://github.com/intel/intel-extension-for-pytorch">IPEX</a></td>
        </tr>
        <tr>
            <td align="center">TensorFlow</td>
            <td align="center"><a href="https://github.com/tensorflow/tensorflow">TensorFlow</a>/<a href="https://github.com/Intel-tensorflow/tensorflow">Intel TensorFlow</a></td>
        </tr>
    </tbody>
</table>
<br>
<br>

[Pruning](pruning.md):
<table>
<thead>
  <tr>
    <th>Pruning Type</th>
    <th>Pruning Granularity</th>
    <th>Pruning Algorithm</th>
    <th>Framework</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="3">Unstructured Pruning</td>
    <td rowspan="3">Element-wise</td>
    <td>Magnitude</td>
    <td>PyTorch, TensorFlow</td>
  </tr>
  <tr>
    <td>Pattern Lock</td>
    <td>PyTorch</td>
  </tr>
  <tr>
    <td>SNIP with momentum</td>
    <td>PyTorch</td>
  </tr>
  <tr>
    <td rowspan="6">Structured Pruning</td>
    <td rowspan="2">Filter/Channel-wise</td>
    <td>Gradient Sensitivity</td>
    <td>PyTorch</td>
  </tr>
  <tr>
    <td>SNIP with momentum</td>
    <td>PyTorch</td>
  </tr>
  <tr>
    <td rowspan="2">Block-wise</td>
    <td>Group Lasso</td>
    <td>PyTorch</td>
  </tr>
  <tr>
    <td>SNIP with momentum</td>
    <td>PyTorch</td>
  </tr>
  <tr>
    <td rowspan="2">Element-wise</td>
    <td>Pattern Lock</td>
    <td>PyTorch</td>
  </tr>
  <tr>
    <td>SNIP with momentum</td>
    <td>PyTorch</td>
  </tr>
</tbody>
</table>
</br>
</br>

[Distillation](distillation.md):
|Distillation Algorithm                          |PyTorch   |TensorFlow |
|------------------------------------------------|:--------:|:---------:|
|Knowledge Distillation                          |&#10004;  |&#10004;   |
|Intermediate Layer Knowledge Distillation       |&#10004;  |Will be supported|
|Self Distillation                               |&#10004;  |&#10006;   |


</br>
</br>


[Orchestration](orchestration.md):
<table>
    <thead>
        <tr>
            <th>Orchestration</th>
            <th>Combinations</th>
            <th>Supported</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>One-shot</td>
            <td>Pruning + Quantization Aware Training</td>
            <td>&#10004;</td>
        </tr>
        <tr>
            <td>Distillation + Quantization Aware Training</td>
            <td>&#10004;</td>
        </tr>
        <tr>
            <td>Distillation + Pruning</td>
            <td>&#10004;</td>
        </tr>
        <tr>
            <td>Distillation + Pruning + Quantization Aware Training</td>
            <td>&#10004;</td>
        </tr>
        <tr>
            <td rowspan=4>Multi-shot</td>
            <td>Pruning then Quantization</td>
            <td>&#10004;</td>
        </tr>
        <tr>
            <td>Distillation then Quantization</td>
            <td>&#10004;</td>
        </tr>
        <tr>
            <td>Distillation then Pruning</td>
            <td>&#10004;</td>
        </tr>
        <tr>
            <td>Distillation then Pruning then Quantization</td>
            <td>&#10004;</td>
        </tr>
    </tbody>
</table>


[Mixed precision](mixed_precision.md):
|Framework     |         |
|--------------|:-----------:|
|TensorFlow    |&#10004;     |
|PyTorch       |&#10004;     |
|ONNX          |plan to support in the future |
|MXNet         |&#10004;     |
