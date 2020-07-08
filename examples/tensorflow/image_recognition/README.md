Step-by-Step
============

This document is used to list steps of reproducing Intel Optimized TensorFlow image recognition models iLiT tuning zoo result.

> **Note**: 
> Most of those models are both supported in Intel optimized TF 1.15.x and Intel optimized TF 2.x. We use 1.15.2 as an example.

# Prerequisite

### 1. Installation
  Recommend python 3.6 or higher version.

  ```Shell
  # Install iLiT
  pip install ilit

  # Install Intel Optimized Tensorflow 1.15.2
  pip install intel-tensorflow==1.15.2
  
  ```

### 2. Prepare Dataset

  TensorFlow [models](https://github.com/tensorflow/models) repo provides [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) to download, process and convert the ImageNet dataset to the TF records format.

### 3. Prepare pre-trained model
  In this version, iLiT just support PB file as input for TensorFlow backend, so we need prepared model pre-trained pb files. For some models pre-trained pb can be found in [IntelAI Models](https://github.com/IntelAI/models/tree/v1.6.0/benchmarks#tensorflow-use-cases), we can found the download link in README file of each model. And for others models in Google [models](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models), we can get the pb files by convert the checkpoint files. We will give a example with Inception_v1 to show how to get the pb file by a checkpoint file.

  1. Download the checkpoint file from [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)
  ```shell
  wget http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz
  tar -xvf inception_v1_2016_08_28.tar.gz
  ```

  2. Exporting the Inference Graph
  ```shell
  git clone https://github.com/tensorflow/models
  cd models/research/slim
  python export_inference_graph.py \
          --alsologtostderr \
          --model_name=inception_v1 \
          --output_file=/tmp/inception_v1_inf_graph.pb
  ```

  3. Use [Netron](https://lutzroeder.github.io/netron/) to get the input/output layer name of inference graph pb, for Inception_v1 the output layer name is `InceptionV1/Logits/Predictions/Reshape_1`

  4. Freezing the exported Graph, please use the tool `freeze_graph.py` in [tensorflow v1.15.2](https://github.com/tensorflow/tensorflow/blob/v1.15.2/tensorflow/python/tools/freeze_graph.py) repo 
  ```shell
  python freeze_graph.py \
          --input_graph=/tmp/inception_v1_inf_graph.pb \
          --input_checkpoint=./inception_v1.ckpt \
          --input_binary=true \
          --output_graph=./frozen_inception_v1.pb \
          --output_node_names=InceptionV1/Logits/Predictions/Reshape_1
  ```

# Run

> *Note*: 
> The model name with `*` means it comes from [models](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models), please follow the step [Prepare pre-trained model](#3-prepare-pre-trained-model) to get the pb files.

### 1. ResNet50 V1

  Download pre-trained PB
  ```shell
  wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet50_fp32_pretrained_model.pb
  ```

  ```Shell
  cd examples/tensorflow/image_recognition
  python main.py -b 10 -a 28 -e 1 -g /PATH/TO/resnet50_fp32_pretrained_model.pb \
          -i input -o predict -r -d /PATH/TO/imagenet/ \
          --resize_method crop --config ./resnet50_v1.yaml
  ```

### 2. ResNet50 V1.5

  Download pre-trained PB
  ```shell
  wget https://zenodo.org/record/2535873/files/resnet50_v1.pb
  ```

  ```Shell
  cd examples/tensorflow/image_recognition
  python main.py -b 10 -a 28 -e 1 -g /PATH/TO/resnet50_v1.pb \
          -i input_tensor -o softmax_tensor -r -d /PATH/TO/imagenet/ \
          --resize_method=crop --r_mean 123.68 --g_mean 116.78 --b_mean 103.94 \
          --config ./resnet50_v1_5.yaml
  ```

### 3. ResNet101

  Download pre-trained PB
  ```shell
  wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet101_fp32_pretrained_model.pb
  ```

  ```Shell
  cd examples/tensorflow/image_recognition
  python main.py -b 10 -a 28 -e 1 -g /PATH/TO/resnet101_fp32_pretrained_model.pb \
          -i input -o resnet_v1_101/predictions/Reshape_1 -r -d /PATH/TO/imagenet/ \
          --image_size 224 --resize_method vgg --label_adjust --config ./resnet101.yaml
  ```

### 4. MobileNet V1

  Download pre-trained PB
  ```shell
  wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/mobilenet_v1_1.0_224_frozen.pb
  ```

  ```Shell
  cd examples/tensorflow/image_recognition
  python main.py -b 10 -a 28 -e 1 -g /PATH/TO/mobilenet_v1_1.0_224_frozen.pb \
          -i input -o MobilenetV1/Predictions/Reshape_1 -r -d /PATH/TO/imagenet/ \
          --resize_method bilinear --config ./mobilenet_v1.yaml
  ```

### 5. MobileNet V2*

  ```Shell
  cd examples/tensorflow/image_recognition
  python main.py -b 10 -a 28 -e 1 -g /PATH/TO/frozen_mobilenet_v2.pb \
          -i input -o MobilenetV2/Predictions/Reshape_1 -r -d /PATH/TO/imagenet/ \
          --resize_method bilinear --config ./mobilenet_v2.yaml
  ```

### 6. Inception V1*

  ```Shell
  cd examples/tensorflow/image_recognition
  python main.py -b 10 -a 28 -e 1 -g /PATH/TO/frozen_inception_v1.pb \
          -i input -o InceptionV1/Logits/Predictions/Reshape_1 -r -d /PATH/TO/imagenet/ \
          --image_size 224 --resize_method bilinear --config ./inceptionv1.yaml
  ```

### 7. Inception V2*

  ```Shell
  cd examples/tensorflow/image_recognition
  python main.py -b 10 -a 28 -e 1 -g /PATH/TO/frozen_inception_v2.pb \
          -i input -o InceptionV2/Predictions/Reshape_1 -r -d /PATH/TO/imagenet/ \
          --image_size 224 --resize_method bilinear --config ./inceptionv2.yaml
  ```

### 8. Inception V3

  Download pre-trained PB
  ```shell
  wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/inceptionv3_fp32_pretrained_model.pb
  ```

  ```Shell
  cd examples/tensorflow/image_recognition
  python main.py -b 10 -a 28 -e 1 -g /PATH/TO/inceptionv3_fp32_pretrained_model.pb \
          -i input -o predict -r -d /PATH/TO/imagenet/  --image_size 299 \
          --resize_method bilinear --config ./inceptionv3.yaml
  ```

### 9. Inception V4

  Download pre-trained PB
  ```shell
  wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/inceptionv4_fp32_pretrained_model.pb
  ```

  ```Shell
  cd examples/tensorflow/image_recognition
  python main.py -b 10 -a 28 -e 1 -g /PATH/TO/inceptionv4_fp32_pretrained_model.pb \
          -i input -o InceptionV4/Logits/Predictions -r -d /PATH/TO/imagenet/  --image_size 299 \
          --resize_method bilinear --config ./inceptionv4.yaml
  ```

### 10. Inception ResNet V2*

  ```Shell
  cd examples/tensorflow/image_recognition
  python main.py -b 10 -a 28 -e 1 -g /PATH/TO/frozen_inception_resnet_v2.pb \
          -i input -o InceptionResnetV2/Logits/Predictions -r -d /PATH/TO/imagenet/  \
          --image_size 299 --resize_method bilinear --config ./irv2.yaml
  ```

Examples of enabling iLiT auto tuning on TensorFlow ResNet50 V1.5
=======================================================

This is a tutorial of how to enable a TensorFlow image recognition model with iLiT.

# User Code Analysis

iLiT supports two usages:

1. User specifies fp32 "model", calibration dataset "q_dataloader", evaluation dataset "eval_dataloader" and metric in tuning.metric field of model-specific yaml config file.

2. User specifies fp32 "model", calibration dataset "q_dataloader" and a custom "eval_func" which encapsulates the evaluation dataset and metric by itself.

As ResNet50 V1.5 is a typical image recognition model, use Top-K as metric which is built-in supported by iLiT. So here we integrate Tensorflow [ResNet50 V1.5](https://github.com/IntelAI/models/tree/v1.6.0/models/image_recognition/tensorflow/resnet50v1_5/inference) in [IntelAI Models](https://github.com/IntelAI/models/tree/v1.6.0) with iLiT by the first use case for simplicity. 

### Write Yaml config file

In examples directory, there is a template.yaml. We could remove most of items and only keep mandotory item for tuning. 


```
# resnet50_v1_5.yaml

framework:
  - name: tensorflow
    inputs: input_tensor
    outputs: softmax_tensor

calibration:                                         
  - iterations: 5, 10
    algorithm:
        activation: minmax

tuning:
    metric:  
      - topk: 1
    accuracy_criterion:
      - relative: 0.01  
    timeout: 0
    random_seed: 9527
```

Here we choose topk built-in metric and set accuracy target as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means early stop as well as a tuning config meet accuracy target.

### prepare

There are three preparation steps in here:
1. Prepare environment
```shell
pip install intel-tensorflow==1.5.2 ilit
```
2. Get the model source code
```shell
git clone -b v1.6.0 https://github.com/IntelAI/models intelai_models
cd intelai_models/models/image_recognition/tensorflow/resnet50v1_5/inference
```
3. Prepare the ImageNet dataset and pretrainined PB file
```shell
wget https://zenodo.org/record/2535873/files/resnet50_v1.pb
```
4. Add load graph and dataloader part in `eval_image_classifier_inference.py`, which needed in iLiT
```python
def load_graph(model_file):
  """This is a function to load TF graph from pb file

  Args:
      model_file (string): TF pb file local path

  Returns:
      graph: TF graph object
  """
  graph = tf.Graph()
  graph_def = tf.compat.v1.GraphDef()

  import os
  file_ext = os.path.splitext(model_file)[1]

  with open(model_file, "rb") as f:
    if file_ext == '.pbtxt':
      text_format.Merge(f.read(), graph_def)
    else:
      graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def, name='')

  return graph

class Dataloader(object):
  """This is a example class that wrapped the model specified parameters,
    such as dataset path, batch size.
    And more importantly, it provides the ability to iterate the dataset.
  Yields:
      tuple: yield data and label 2 numpy array
  """
  def __init__(self, data_location, subset, input_height, input_width,
                batch_size, num_cores, resize_method='crop'):
    self.batch_size = batch_size
    self.subset = subset
    self.dataset = datasets.ImagenetData(data_location)
    self.total_image = self.dataset.num_examples_per_epoch(self.subset)
    self.preprocessor = self.dataset.get_image_preprocessor()(
        input_height,
        input_width,
        batch_size,
        num_cores,
        resize_method)
    self.n = int(self.total_image / self.batch_size)

  def __iter__(self):
    images, labels, filenames = self.preprocessor.minibatch(self.dataset, subset=self.subset, cache_data=False)
    with tf.compat.v1.Session() as sess:
        for i in range(self.n):
            yield sess.run([images, labels])
```

### code update

After completed preparation steps, we just need add a tuning part in `eval_classifier_optimized_graph` class.

```python
  def auto_tune(self):
    """This is iLiT tuning part to generate a quantized pb

    Returns:
        graph: it will return a quantized pb
    """
    import ilit
    fp32_graph = load_graph(self.args.input_graph)
    tuner = ilit.Tuner(self.args.config)
    dataloader = Dataloader(self.args.data_location, 'validation',
                            RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, self.args.batch_size,
                            num_cores=self.args.num_cores,
                            resize_method='crop')
    q_model = tuner.tune(
                        fp32_graph,
                        q_dataloader=dataloader,
                        eval_func=None,
                        eval_dataloader=dataloader)
    return q_model
```

Finally, add one line in `__main__` function of `eval_image_-classifier_inference.py` to use iLiT by yourself as below.
```python
q_graph = evaluate_opt_graph.auto_tune()
```
We can use below cmd to test it.
```shell
python eval_image_classifier_inference.py -b 10 -a 28 -e 1 -m resnet50_v1_5 \
        -g /PATH/TO/resnet50_v1.pb -d /PATH/TO/imagenet/
```
The iLiT tune() function will return a best quantized model during timeout constrain.
