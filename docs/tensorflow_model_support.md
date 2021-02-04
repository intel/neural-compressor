Tensorflow Model Support in Intel® Low Precision Optimization Tool
===========================================

Intel® Low Precision Optimization Tool supports diffrent model formats of TensorFlow 1.x and 2.x.


| TensorFlow model format | Supported? | Example | Comments |
| ------ | ------ |------|------|
| frozen pb | Yes | [../examples/tensorflow/image_recognition](../examples/tensorflow/image_recognition), [../examples/tensorflow/oob_models](../examples/tensorflow/oob_models) | |
| Graph object | Yes | [../examples/tensorflow/style_transfer](../examples/tensorflow/style_transfer), [../examples/tensorflow/recommendation/wide_deep_large_ds](../examples/tensorflow/recommendation/wide_deep_large_ds) | |
| GraphDef object | Yes | | |
| tf1.x checkpoint | Yes | [../examples/helloworld/tf_example4](../examples/helloworld/tf_example4), [../examples/tensorflow/object_detection](../examples/tensorflow/object_detection) | |
| keras.Model object | Yes | | |
| keras saved model | Yes | [../examples/helloworld/tf_example2](../examples/helloworld/tf_example2) | |
| tf2.x saved model | TBD | | |
| tf2.x h5 format model  | TBD ||
| slim checkpoint | Yes | [../examples/helloworld/tf_example3](../examples/helloworld/tf_example3) |
| tf1.x saved model | No| | No plan to support it |
| tf2.x checkpoint | No | | As tf2.x checkpoint only has weight and does not contain any description of the computation, please use different tf2.x model for quantization |


# Usage

You can directly pass the directory or object to quantizer, for example:
```python
from lpot import Quantization
quantizer = Quantization('./conf.yaml')
dataset = mnist_dataset(mnist.test.images, mnist.test.labels)
data_loader = quantizer.dataloader(dataset=dataset, batch_size=1)
# model parameter could be one of frozen_pb, Graph, GraphDef, checkpoint_path, keras.Model and keras_savedmodel_path
q_model = quantizer(frozen_pb, q_dataloader=data_loader, eval_func=eval_func)

```


