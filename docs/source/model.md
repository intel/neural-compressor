Model
=====

1. [Introduction](#introduction)

2. [Supported Framework Model Matrix](#supported-framework-model-matrix)

3. [Examples](#examples)


## Introduction

The Neural Compressor Model feature is used to encapsulate the behavior of model building and saving. By simply providing information such as different model formats and framework_specific_info, Neural Compressor performs optimizations and quantization on this model object and returns a Neural Compressor Model object for further model persistence or benchmarking. A Neural Compressor Model helps users to maintain necessary model information which is required during optimization and quantization such as the input/output names, workspace path, and other model format knowledge. This helps unify the features gap brought by different model formats and frameworks.
<a target="_blank" href="./imgs/inc_model.png" text-align:center>
    <center> 
        <img src="./imgs/model.png" alt="Architecture" width=480 height=200> 
    </center>
</a>


## Supported Framework Model Matrix

<table>
    <thead>
        <tr>
            <th>Framework</th>
            <th>Input Model Format</th>
            <th>Output Model Format</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=11>TensorFlow</td>
            <td>frozen pb</td>
            <td>frozen pb</td>
        </tr>
        <tr>
            <td>graph object(tf.compat.v1.Graph)</td>
            <td>frozen pb</td>
        </tr>
        <tr>
            <td>graphDef object(tf.compat.v1.GraphDef)</td>
            <td>frozen pb</td>
        </tr>
        <tr>
            <td>tf1.x checkpoint</td>
            <td>frozen pb</td>
        </tr>
        <tr>
            <td>keras.Model object</td>
            <td>keras saved model</td>
        </tr>
        <tr>
            <td>keras saved model</td>
            <td>keras saved model</td>
        </tr>
        <tr>
            <td>tf2.x saved model</td>
            <td>saved model</td>
        </tr>
        <tr>
            <td>tf2.x h5 format model</td>
            <td>saved model</td>
        </tr>
        <tr>
            <td>slim checkpoint</td>
            <td>frozen pb</td>
        </tr>
        <tr>
            <td>tf1.x saved model</td>
            <td>saved model</td>
        </tr>
        <tr>
            <td>tf2.x checkpoint</td>
            <td>saved model</td>
        </tr>
        <tr>
            <td rowspan=2>PyTorch</td>
            <td>torch.nn.Module</td>
            <td>frozen pt</td>
        </tr>
        <tr>
            <td>torch.nn.Module</td>
            <td>json file (intel extension for pytorch)</td>
        </tr>
        <tr>
            <td rowspan=2>ONNX</td>
            <td>frozen onnx</td>
            <td>frozen onnx</td>
        </tr>
        <tr>
            <td>onnx.onnx_ml_pb2.ModelProto</td>
            <td>frozen onnx</td>
        </tr>
        <tr>
            <td rowspan=2>MXNet</td>
            <td>mxnet.gluon.HybridBlock</td>
            <td>save_path.json</td>
        </tr>
        <tr>
            <td>mxnet.symbol.Symbol</td>
            <td>save_path-symbol.json and save_path-0000.params</td>
        </tr>
    </tbody>
</table>


## Examples

Users can create, use, and save models in the following manners:

```python
from neural_compressor.model import Model
inc_model = Model(input_model)
```

or

```python
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig

conf = PostTrainingQuantConfig()
q_model = quantization.fit(model = inc_model, conf=conf)
q_model.save("saved_result")
```
