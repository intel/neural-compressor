tf_example2 example
=====================

Step-by-Step
============

This is Hello World to demonstrate how to quick start with Intel® Low Precision Optimization Tool. It is a Keras model on mnist dataset defined by helloworld/train.py, we will implement a customized metric and a customized dataloader for quantization and evaluation.


## Prerequisite

### 1. Installation
```shell
# Install Intel® Low Precision Optimization Tool
pip install lpot
```
### 2. Install Intel Tensorflow
```shell
pip install intel-tensorflow==2.3.0
```

### 3. Prepare FP32 model
```shell
cd <WORK_DIR>/examples/helloworld
python train.py
```
## Run Command
  # The cmd of quantization and predict with the quantized model 
  ```shell
  python test.py 
  ```
## Introduction 
This exmaple can demonstrate the steps to do quantization on Keras generated saved model with customized dataloader and metric. 
### 1.Add inputs and outputs information into conf.yaml, to get the input and output tensor name please refer to helloworld/train.py.  

### 2.Define a customer dataloader for mnist  

```python
class Dataset(object):
  def __init__(self):
      (train_images, train_labels), (test_images,
                 test_labels) = keras.datasets.fashion_mnist.load_data()
      self.test_images = test_images.astype(np.float32) / 255.0
      self.labels = test_labels
      pass

  def __getitem__(self, index):
      return self.test_images[index], self.labels[index]

  def __len__(self):
      return len(self.test_images)

```

### 3.Define a customized metric  
This customized metric will caculate accuracy.
```python
class MyMetric(object):
  def __init__(self, *args):
      self.pred_list = []
      self.label_list = []
      self.samples = 0
      pass

  def update(self, predict, label):
      self.pred_list.extend(np.argmax(predict, axis=1))
      self.label_list.extend(label)
      self.samples += len(label) 
      pass

  def reset(self):
      self.pred_list = []
      self.label_list = []
      self.samples = 0
      pass

  def result(self):
      correct_num = np.sum(
            np.array(self.pred_list) == np.array(self.label_list))
      return correct_num / self.samples

```
### 4.Use the customized data loader and metric for quantization 
```python
quantizer = Quantization('./conf.yaml')
dataset = Dataset()
quantizer.metric = common.Metric(MyMetric, 'hello_metric')
quantizer.calib_dataloader = common.DataLoader(dataset, batch_size=1)
quantizer.eval_dataloader = common.DataLoader(dataset, batch_size=1)
quantizer.model = common.Model('../models/saved_model')
q_model = quantizer()

```

### 5. Run quantized model
please get the input and output op name from lpot_workspace/tensorflow/hello_world/deploy.yaml
```yaml
model:
  name: hello_world
  framework: tensorflow
  inputs:
  - input
  outputs:
  - output
```
Run inference on the quantized model
```python
import tensorflow as tf
with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session() as sess:
     tf.compat.v1.import_graph_def(q_model.graph_def, name='')
     styled_image = sess.run(['output:0'], feed_dict={'input:0':dataset.test_images})
     print("Inference is done.")
```
