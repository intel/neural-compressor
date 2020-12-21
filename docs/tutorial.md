Tutorial
=========================================

This tutorial will introduce step by step instructions on how to integrate models with Intel® Low Precision Optimization Tool (LPOT) with examples. 

First user needs to prepare FP32 model, and then configure the model information and tuning parameters in yaml. LPOT supports builtin dataloaders for pupular dataset such as imagenet and coco, and supports builtin evaluators like topk, Loss and MSE, to use the builtin dataloader and evaluators user can configure them in yaml file. Then user can add a simple launcher to call quantization. Of course, user can also define their own dataloader and evaluators. This tutorial will use examples to introduce how to do quantization and run benchmark with both builtin and customized dataloader and evaluators.  

<table>
  <tr>
    <td>Tutorial</td>
  </tr>
  <tr>
    <td><img src="docs/imgs/tutorial.png" width=640 height=320></td>
  </tr>
 </table>
 
Here is an example of ymal file and laucher code, the ymal file defined the tuning accuracy target and evaluation funciton, and only 3 lines of code are added to launch LPOT for quantization.
<table>
  <tr>
    <td>ymal</td>
    <td>launcher</td>
  </tr>
  <tr>
    <td><img src="docs/imgs/ymal.png" width=640 height=320></td>
    <td><img src="docs/imgs/launcher.png" width=640 height=320></td>
  </tr>
 </table>

To define a customized dataloader or evaluator for quantization, user can implement them follow this template:
<table>
  <tr>
    <td>Template for customized dataloader and evaluator</td>
  </tr>
  <tr>
    <td><img src="docs/imgs/template.png" width=640 height=320></td>
  </tr>
 </table>

Next, let's introduce how to do quantization in different scenarios. 

# Coding free quantization
The examples/helloworld/tf_coding_free demonstrates how to utilize LPOT builtin dataloader and evalautors for quantizaiton, and how to use LPOT Benchmark class for performance and accuracy measurement, and user only need to add 3 lines of launcher code for tuning. See [README](examples/helloworld/tf_coding_free/README.md)


# Customized dataloader
With a Keras saved model as example, [examples/helloworld/tf2.x_custom_dataloader](examples/helloworld/tf2.x_custom_dataloader] demonstrates how to define a customized dataloader. 

First define a dataset class on mnist.
```
class Dataset(object):
  def __init__(self):
      # TODO:initialize dataset related info here
      (train_images, train_labels), (test_images,
                 test_labels) = keras.datasets.fashion_mnist.load_data()
      self.test_images = test_images.astype(np.float32) / 255.0
      self.labels = test_labels
      pass

  def __getitem__(self, index):
      # TODO:get item magic method
      # return a tuple containing 1 image and 1 label
      # for example, return img, label
      return self.test_images[index], self.labels[index]

  def __len__(self):
      # TODO:get total length of dataset, such as how many images in the dataset
      # if the total length is not able to know, pls implement __iter__() magic method
      # rather than above two methods.
      return len(self.test_images)

```
Then define a dataloader based on the mnist dataset, run quantization on the customized dataloader. q_model is the quantized model. 
```
import lpot
quantizer = lpot.Quantization('./conf.yaml')
dataset = Dataset()
# Define a customer data loader
dataloader = quantizer.dataloader(dataset, batch_size=1)
q_model = quantizer('../models/simple_model', q_dataloader = dataloader, eval_dataloader = dataloader)
```
# Customized evaluator
Example examples/helloworld/tf2.x_custom_metric shows how to define and use a customized evaluator for quantization, this evaluator calculate accuracy and it will be registered in Qunatization object with metric() funciton. See [README](examples/helloworld/tf2.x_custom_metric/README.md)
```
quantizer.metric('hello_metric', MyMetric)
```

```
class MyMetric(Metric):
  def __init__(self, *args):
      # TODO:initialize metric related info here
      self.pred_list = []
      self.label_list = []
      self.samples = 0
      pass

  def update(self, predict, label):
      # TODO:metric evaluation per evaluation
      self.pred_list.extend(np.argmax(predict, axis=1))
      self.label_list.extend(label)
      self.samples += len(label)
      pass

  def reset(self):
      # TODO:reset variable if needed
      self.pred_list = []
      self.label_list = []
      self.samples = 0
      pass

  def result(self):
      # TODO:calculate the whole batch final evaluation result
      # return a float value which is higher-is-better.
      # for example, return coco_map_value
      correct_num = np.sum(
            np.array(self.pred_list) == np.array(self.label_list))
      return correct_num / self.samples

```

# The interface is similiar for different TensorFlow models
1. see example on tf1.x frozen pb at [tf1.x_pb](examples/helloworld/tf1.x_pb/).
2. see example on tf1.x checkpoint at [tf1.x_ckpt](examples/helloworld/tf1.x_ckpt/).
3. To quantize a slim .ckpt model, we need to get the graph. See a full example on slim at [examples/helloworld/tf1.x_slim](examples/helloworld/tf1.x_slim). 
```
   import lpot
    quantizer = lpot.Quantization('./conf.yaml')

    # Get graph from slim checkpoint
    from tf_slim.nets import inception
    model_func = inception.inception_v1
    arg_scope = inception.inception_v1_arg_scope()
    kwargs = {'num_classes': 1001}
    inputs_shape = [None, 224, 224, 3]
    images = tf.compat.v1.placeholder(name='input', \
    dtype=tf.float32, shape=inputs_shape)

    from lpot.adaptor.tf_utils.util import get_slim_graph
    graph = get_slim_graph('./inception_v1.ckpt', model_func, \
            arg_scope, images, **kwargs)

    # Do quantization
    quantized_model = quantizer(graph)

```



