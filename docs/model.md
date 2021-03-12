Model
===============

Model feature of LPOT is used to encapsulate the behavior of model building and saving. By just give information of different model format and framework_specific_info, LPOT will do optimizations and quantization on this model object and return a LPOT Model object for further model persisting or benchmark. LPOT Model make it possible to keep neccessary information of a model which is needed during optimization and quantization like the input/output names, workspace path and other model format knowledges. This helps unify the features gap bring by different model format and frameworks.

User can create, use and save a model in this way:

```python
from lpot import Quantization, common
quantizer = Quantization('./conf.yaml')
quantizer.model = common.Model('/path/to/model')
q_model = quantizer()
q_model.save(save_path)

```

# Framework model support list

#### TensorFlow

| Model format | Supported? | Example | Comments | Save format |
| ------ | ------ |------|------|------|
| frozen pb | Yes | [../examples/tensorflow/image_recognition](../examples/tensorflow/image_recognition), [../examples/tensorflow/oob_models](../examples/tensorflow/oob_models) | | frozen pb | 
| Graph object | Yes | [../examples/tensorflow/style_transfer](../examples/tensorflow/style_transfer), [../examples/tensorflow/recommendation/wide_deep_large_ds](../examples/tensorflow/recommendation/wide_deep_large_ds) | | frozen pb |
| GraphDef object | Yes | | | frozen pb |
| tf1.x checkpoint | Yes | [../examples/helloworld/tf_example4](../examples/helloworld/tf_example4), [../examples/tensorflow/object_detection](../examples/tensorflow/object_detection) | | frozen pb |
| keras.Model object | Yes | | | frozen pb |
| keras saved model | Yes | [../examples/helloworld/tf_example2](../examples/helloworld/tf_example2) | | frozen pb | 
| tf2.x saved model | Yes | | | saved model |
| tf2.x h5 format model  | TBD || |
| slim checkpoint | Yes | [../examples/helloworld/tf_example3](../examples/helloworld/tf_example3) | | frozen pb |
| tf1.x saved model | Yes| | | saved model |
| tf2.x checkpoint | No | | As tf2.x checkpoint only has weight and does not contain any description of the computation, please use different tf2.x model for quantization | |

The following methods can be used in tensorflow model

```python
graph_def = model.graph_def
input_tensor_names = model.input_tensor_names
model.input_tensor_names = input_tensor_names
output_tensor_names = model.output_tensor_names
model.output_tensor_names = output_tensor_names
input_node_names = model.input_node_names
output_node_names = model.output_node_names
input_tensor = model.input_tensor
output_tensor = model.output_tensor
```

#### MXNet

| Model format | Supported? | Example | Comments | Save format |
| ------ | ------ |------|------|------|
| mxnet.gluon.HybridBlock | Yes | | | export HybridBlock as save_path.json |
| mxnet.symbol.Symbol | Yes | | | save a save_path-symbol.json file and a save_path-0000.params file |

#### PyTorch

| Model format | Supported? | Example | Comments | Save format |
| ------ | ------ |------|------|------|
| torch.nn.model | Yes | | | Without Intel PyTorch Extension(IPEX): save the configure and weights files as "best_configure.yaml" and "best_model_weights.pt" to the given path <br> With IPEX: save the configure as "best_configure.json" to the given path |

* Loading model:

```python
# Without IPEX
model                 # fp32 model
from lpot.utils.pytorch import load
quantized_model = load(
    os.path.abspath(os.path.expanduser(Path)), model)

# With IPEX
import intel_pytorch_extension as ipex
model                 # fp32 model
model.to(ipex.DEVICE)
try:
    new_model = torch.jit.script(model)
except:
    new_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224).to(ipex.DEVICE))
ipex_config_path = os.path.join(os.path.expanduser(args.tuned_checkpoint),
                                "best_configure.json")
conf = ipex.AmpConf(torch.int8, configure_file=ipex_config_path)
with torch.no_grad():
    with ipex.AutoMixPrecision(conf, running_mode='inference'):
        output = new_model(input.to(ipex.DEVICE))
```

