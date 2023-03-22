tf_example2 example
=====================

Step-by-Step
============

This is Hello World to demonstrate how to quickly start with Intel® Neural Compressor. It is a Keras model on mnist dataset defined by helloworld/train.py, we will implement a customized metric and a customized dataloader for quantization and evaluation.


## Prerequisite

### 1. Installation
```shell
pip install -r requirements.txt
```
> Note: Validated TensorFlow [Version](/docs/source/installation_guide.md#validated-software-environment).

### 2. Prepare FP32 model
```shell
cd <WORK_DIR>/examples/helloworld
python train.py
```
## Run Command
The cmd of quantization and predict with the quantized model 
```shell
python test.py 
```
## Introduction 
This example can demonstrate the steps to do quantization on Keras generated saved model with customized dataloader and metric. 

### 1. Define a customer dataloader for mnist  

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

### 2. Define a customized metric  
This customized metric will calculate accuracy.
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
### 3. Use the customized data loader and metric for quantization 
```python
    dataset = Dataset()
    dataloader = DataLoader(framework='tensorflow', dataset=dataset)
    config = PostTrainingQuantConfig(backend='itex')
    q_model = fit(
        model='../models/saved_model',
        conf=config,
        calib_dataloader=dataloader,
        eval_dataloader=dataloader,
        eval_metric=MyMetric())

```

### 4. Run quantized model
Please get the input and output op name from nc_workspace/tensorflow/hello_world/deploy.yaml

Run inference on the quantized model
```python
    keras_model = q_model.model
    predictions = keras_model.predict_on_batch(dataset.test_images)
    print("Inference is done.")
```
