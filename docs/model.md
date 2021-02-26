Model
===============

Model feature of LPOT is used to encapsulate the behavior of model building and saving. It can create TensorflowModel, MXNetModel, PyTorchModel, PyTorchIpexModel, ONNXModel based on the information of model path, framework_specific_info and other kwargs.
User can create, use and save a model in this way:

```python
from lpot.model import MODELS
model = MODELS[framework](model, framework_specific_info, **kwargs)
model.model = new_model # pytorch, pytorch_ipex, mxnet, onnxrt_qlinearops, onnxrt_integerops
model.graph_def = graph_def # tesnsorflow
model.save(save_path)

# The following methods can only be used in tensorflow model
input_tensor_names = model.input_tensor_names
model.input_tensor_names = input_tensor_names
output_tensor_names = model.output_tensor_names
model.output_tensor_names = output_tensor_names
input_node_names = model.input_node_names
output_node_names = model.output_node_names
input_tensor = model.input_tensor
output_tensor = model.output_tensor
```
