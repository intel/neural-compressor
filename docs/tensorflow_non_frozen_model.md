Tensorflow Non-Frozen Model Support in Intel® Low Precision Optimization Tool
===========================================

Besides frozen pb (.pb), Intel® Low Precision Optimization Tool also support non-frozen models for TensorFlow 1.x and 2.x, more formats will be supported in future releases.

# MetaGraph
In this case, a model consists of three or four files stored in the same directory:

* model_name.meta
* model_name.index
* model_name.data-00000-of-00001 (digit part may vary)
* checkpoint (optional)

You can directly pass the directory to tuner, for example:
```python
from ilit import Quantization
quantizer = Quantization('./conf.yaml')
dataset = mnist_dataset(mnist.test.images, mnist.test.labels)
data_loader = quantizer.dataloader(dataset=dataset, batch_size=1)
q_model = quantizer('./model', q_dataloader=data_loader, eval_func=eval_func)
```

# SavedModel format of TensorFlow 1.x and 2.x versions
In this case, a model consists of a special directory with a .pb file and several subfolders: variables, assets, and assets.extra. For more information about the SavedModel directory, refer to the README file in the TensorFlow repository.

You can pass the tf.keras.Model to tuner, for example:
```python
    # Load saved model
    model = tf.keras.models.load_model("./models/saved_model")

    # Run ilit to get the quantized graph
    quantizer = Quantization('./conf.yaml')
    dataloader = quantizer.dataloader(dataset=(test_images, test_labels))
    quantized_model = quantizer(model, q_dataloader=dataloader, eval_func=eval_func)
``` 

# GraphDef
If you have GraphDef object you can also pass it to tuner directly. 

```python
from ilit import Quantization
import os

graph_def = tf.compat.v1.GraphDef()

with open(model_file, "rb") as f:
   graph_def.ParseFromString(f.read())

# Run ilit to get the quantized pb
quantizer = Quantization('./conf.yaml')
dataloader = quantizer.dataloader(dataset=(test_images, test_labels))
quantized_model = quantizer(graph_def, q_dataloader=dataloader, eval_func=eval_func)

```

