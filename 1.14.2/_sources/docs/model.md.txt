Model
=====

The Neural Compressor Model feature is used to encapsulate the behavior of model building and saving. By simply providing information such as different model formats and framework_specific_info, Neural Compressor performs optimizations and quantization on this model object and returns an Neural Compressor Model object for further model persisting or benchmarking. An Neural Compressor Model helps users to maintain necessary model information which is needed during optimization and quantization such as the input/output names, workspace path, and other model format knowledge. This helps unify the features gap brought by different model formats and frameworks.

Users can create, use, and save models in the following manner:

```python
from neural_compressor import Quantization, common
quantizer = Quantization('./conf.yaml')
quantizer.model = '/path/to/model'
q_model = quantizer.fit()
q_model.save(save_path)

```

## Framework model support list

### TensorFlow

| Model format | Parameters | Comments | Usage |
| ------ | ------ |------|------|
| frozen pb | **model**(str): path to frozen pb <br> **framework_specific_info**(dict): information about model and framework, such as input_tensor_names, input_tensor_names, workspace_path and name <br> **kwargs**(dict): other required parameters | **Examples**: <br> [../examples/tensorflow/image_recognition](../examples/tensorflow/image_recognition) <br> [../examples/tensorflow/oob_models](../examples/tensorflow/oob_models) <br> **Save format**: <br> frozen pb | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is the path of model, like ./path/to/frozen.pb** |
| Graph object | **model**(tf.compat.v1.Graph): tf.compat.v1.Graph object  <br> **framework_specific_info**(dict): information about model and framework, such as input_tensor_names, input_tensor_names, workspace_path and name <br> **kwargs**(dict): other required parameters | **Examples**: <br> [../examples/tensorflow/style_transfer](../examples/tensorflow/style_transfer) <br> [../examples/tensorflow/recommendation/wide_deep_large_ds](../examples/tensorflow/recommendation/wide_deep_large_ds) <br> **Save format**: <br> frozen pb | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is the object of tf.compat.v1.Graph** |
| Graph object | **model**(tf.compat.v1.GraphDef) tf.compat.v1.GraphDef object <br> **framework_specific_info**(dict): information about model and framework, such as input_tensor_names, input_tensor_names, workspace_path and name <br> **kwargs**(dict): other required parameters | **Save format**: <br> frozen pb | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is the object of tf.compat.v1.GraphDef** |
| tf1.x checkpoint | **model**(str): path to checkpoint <br> **framework_specific_info**(dict): information about model and framework, such as input_tensor_names, input_tensor_names, workspace_path and name <br> **kwargs**(dict): other required parameters | **Examples**: <br> [../examples/helloworld/tf_example4](../examples/helloworld/tf_example4) <br> [../examples/tensorflow/object_detection](../examples/tensorflow/object_detection)  <br> **Save format**: <br> frozen pb | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is the path of model, like ./path/to/ckpt/** |
| keras.Model object | **model**(tf.keras.Model): tf.keras.Model object <br> **framework_specific_info**(dict): information about model and framework, such as input_tensor_names, input_tensor_names, workspace_path and name <br> **kwargs**(dict): other required parameters | **Save format**: <br> keras saved model | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is the object of tf.keras.Model** |
| keras saved model | **model**(str): path to keras saved model <br> **framework_specific_info**(dict): information about model and framework, such as input_tensor_names, input_tensor_names, workspace_path and name <br> **kwargs**(dict): other required parameters | **Examples**: <br> [../examples/helloworld/tf_example2](../examples/helloworld/tf_example2) <br> **Save format**: <br> keras saved model | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is the path of model, like ./path/to/saved_model/** |
| tf2.x saved model | **model**(str): path to saved model <br> **framework_specific_info**(dict): information about model and framework, such as input_tensor_names, input_tensor_names, workspace_path and name <br> **kwargs**(dict): other required parameters | **Save format**: <br> saved model | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is the path of model, like ./path/to/saved_model/** |
| tf2.x h5 format model  | | TBD | |
| slim checkpoint | **model**(str): path to slim checkpoint <br> **framework_specific_info**(dict): information about model and framework, such as input_tensor_names, input_tensor_names, workspace_path and name <br> **kwargs**(dict): other required parameters | **Examples**: <br> [../examples/helloworld/tf_example3](../examples/helloworld/tf_example3) <br> **Save format**: <br> frozen pb | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is thepath of model, like ./path/to/model.ckpt**|
| tf1.x saved model | **model**(str): path to saved model, **framework_specific_info**(dict): information about model and framework, such as input_tensor_names, input_tensor_names, workspace_path and name <br> **kwargs**(dict): other required parameters | **Save format**: <br> saved model | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is the path of model, like ./path/to/saved_model/** |
| tf2.x checkpoint | | Not support yes. As tf2.x checkpoint only has weight and does not contain any description of the computation, please use different tf2.x model for quantization | |

The following methods can be used in the TensorFlow model:

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

### MXNet

| Model format | Parameters | Comments | Usage |
| ------ | ------ |------|------|
| mxnet.gluon.HybridBlock | **model**(mxnet.gluon.HybridBlock): mxnet.gluon.HybridBlock object <br> **framework_specific_info**(dict): information about model and framework <br> **kwargs**(dict): other required parameters | **Save format**: <br> save_path.json | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is mxnet.gluon.HybridBlock object** |
| mxnet.symbol.Symbol | **model**(tuple): tuple of symbol, arg_params, aux_params <br> **framework_specific_info**(dict): information about model and framework <br> **kwargs**(dict): other required parameters | **Save format**: <br> save_path-symbol.json and save_path-0000.params | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is the tuple of symbol, arg_params, aux_params** |

* Get symbol, arg_params, aux_params from symbol and param files.

```python
import mxnet as mx
from mxnet import nd

symbol = mx.sym.load(symbol_file_path)
save_dict = nd.load(param_file_path)
arg_params = {}
aux_params = {}
for k, v in save_dict.items():
    tp, name = k.split(':', 1)
    if tp == 'arg':
        arg_params[name] = v
    if tp == 'aux':
        aux_params[name] = v
```

### PyTorch

| Model format | Parameters | Comments | Usage |
| ------ | ------ |------|------|
| torch.nn.model | **model**(torch.nn.model): torch.nn.model object <br> **framework_specific_info**(dict): information about model and framework <br> **kwargs**(dict): other required parameters | **Save format**: <br> Without Intel PyTorch Extension(IPEX): /save_path/best_configure.yaml and /save_path/best_model_weights.pt <br> With IPEX: /save_path/best_configure.json | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is torch.nn.model object** |

* Loading model:

```python
# Without IPEX
from neural_compressor.utils.pytorch import load
quantized_model = load(
    os.path.abspath(os.path.expanduser(Path)), model) # model is a fp32 model

# With IPEX
import intel_pytorch_extension as ipex
model.to(ipex.DEVICE) # model is a fp32 model
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

