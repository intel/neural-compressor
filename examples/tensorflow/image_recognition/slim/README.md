Step-by-Step
============

This document is used to list steps of reproducing Intel Optimized TensorFlow slim models tuning  result.

> **Note**: 
> Slim models are only supported in Intel optimized TF 1.15.x. We use 1.15.2 as an example.

# Prerequisite

### 1. Installation
  Recommend python 3.6 or higher version.

  ```shell
  pip install -r requirements.txt
  
  ```

### 2. Prepare Dataset

  TensorFlow [models](https://github.com/tensorflow/models) repo provides [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) to download, process and convert the ImageNet dataset to the TF records format.
  We also prepared related scripts in `imagenet_prepare` directory. To download the raw images, the user must create an account with image-net.org. If you have downloaded the raw data and preprocessed the validation data by moving the images into the appropriate sub-directory based on the label (synset) of the image. we can use below command ro convert it to tf records format.

  ```shell
  cd examples/tensorflow/image_recognition
  # convert validation subset
  bash prepare_dataset.sh --output_dir=./data --raw_dir=/PATH/TO/img_raw/val/ --subset=validation
  # convert train subset
  bash prepare_dataset.sh --output_dir=./data --raw_dir=/PATH/TO/img_raw/train/ --subset=train
  ```

### 3. Prepare pre-trained model
  This tool support slim ckpt file as input for TensorFlow backend, so we can directly download the ckpt model. The demonstrated models are in Google [models](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models). We will give a example with Inception_v1:

  Download the checkpoint file from [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)
  ```shell
  wget http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz
  tar -xvf inception_v1_2016_08_28.tar.gz
  ```
> **Note**: 
> slim model need module tf_slim by default and to run the slim nets, user specific model should define model_func, and arg_scope and register use TFSlimNetsFactory's register API, there is an example model inception_v4.py and registered in main.py
  ```python
  factory = TFSlimNetsFactory()
  input_shape = [None, 299, 299, 3]
  factory.register('inception_v4', inception_v4, input_shape, inception_v4_arg_scope)

  ```
> tf_slim default supported nets are: ['alexnet_v2', 'overfeat', 'vgg_a', 'vgg_16', 'vgg_19', 'inception_v1', 'inception_v2', 'inception_v3','resnet_v1_50', 'resnet_v1_101', 'resnet_v1_152', 'resnet_v1_200','resnet_v2_50', 'resnet_v2_101', 'resnet_v2_152', 'resnet_v2_200']
> make sure you input_graph name like the default nets, eg: vgg_16.ckpt will map to nets vgg_16 while vgg16 will throw a not found error.

# Run
## tune
  ./run_tuning.sh --config=model.yaml --input_model=/path/to/input_model.ckpt --output=/path/to/save/lpot_tuned.pb

## benchmark
  ./run_tuning.sh --config=model.yaml --input_model=/path/to/lpot_tuned.pb

### 1. resnet_v1_50

  ```shell
  cd examples/tensorflow/image_recognition/slim
  bash run_tuning.sh --config=resnet_v1_50.yaml \
      --input_model=/PATH/TO/resnet_v1_50.ckpt \
      --output_model=./lpot_resnet_v1_50.pb
  ```

### 2. resnet_v1_101

  ```shell
  cd examples/tensorflow/image_recognition/slim
  bash run_tuning.sh --config=../resnet101.yaml \
          --input_model=/PATH/TO/resnet_v1_101.ckpt \
          --output_model=./lpot_resnet_v1_101.pb
  ```

### 3. resnet_v1_152

  ```shell
  cd examples/tensorflow/image_recognition/slim
  bash run_tuning.sh --config=resnet_v1_152.yaml \
          --input_model=/PATH/TO/resnet_v1_152.ckpt \
          --output_model=./lpot_resnet_v1_152.pb

  ```

### 4. resnet_v2_50

  ```shell
  cd examples/tensorflow/image_recognition/slim
  bash run_tuning.sh --config=../resnet_v2_50.yaml \
          --input_model=/PATH/TO/resnet_v2_50.ckpt \
          --output_model=./lpot_resnet_v2_50.pb

  ```

### 5. resnet_v2_101

  ```shell
  cd examples/tensorflow/image_recognition/slim
  bash run_tuning.sh --config=../resnet_v2_101.yaml \
          --input_model=/PATH/TO/resnet_v2_101.ckpt \
          --output_model=./lpot_resnet_v2_101.pb

  ```

### 6. resnet_v2_152

  ```shell
  cd examples/tensorflow/image_recognition/slim
  bash run_tuning.sh --config=../resnet_v2_152.yaml \
          --input_model=/PATH/TO/resnet_v2_152.ckpt \
          --output_model=./lpot_resnet_v2_152.pb

  ```

### 7. inception_v1

  ```shell
  cd examples/tensorflow/image_recognition/slim
  bash run_tuning.sh --config=../inception_v1.yaml \
          --input_model=/PATH/TO/inception_v1.ckpt \
          --output_model=./lpot_inception_v1.pb

  ```

### 8. inception_v2

  ```shell
  cd examples/tensorflow/image_recognition/slim
  bash run_tuning.sh --config=../inception_v2.yaml \
      --input_model=/PATH/TO/inception_v2.ckpt \
      --output_model=./lpot_inception_v2.pb
  ```

### 9. inception_v3

  ```shell
  cd examples/tensorflow/image_recognition/slim
  bash run_tuning.sh --config=inception_v3.yaml \
      --input_model=/PATH/TO/inception_v3.ckpt \
      --output_model=./lpot_inception_v3.pb
  ```

### 10. inception_v4

  ```shell
  cd examples/tensorflow/image_recognition/slim
  bash run_tuning.sh --config=../inception_v4.yaml \
      --input_model=/PATH/TO/inception_v4.ckpt \
      --output_model=./lpot_inception_v4.pb
  ```

### 11. vgg16

  ```shell
  cd examples/tensorflow/image_recognition/slim
  bash run_tuning.sh --config=../vgg16.yaml \
          --input_model=/PATH/TO/vgg_16.ckpt \
          --output_model=./lpot_vgg_16.pb
  ```

### 12. vgg19

  ```shell
  cd examples/tensorflow/image_recognition/slim
  bash run_tuning.sh --config=../vgg19.yaml \
          --input_model=/PATH/TO/vgg_19.ckpt \
          --output_model=./lpot_vgg_19.pb
  ```

Examples of enabling Intel® Low Precision Optimization Tool auto tuning on TensorFlow Inception V1
=======================================================

This is a tutorial of how to enable a TensorFlow slim model with Intel® Low Precision Optimization Tool.

# User Code Analysis

Intel® Low Precision Optimization Tool supports two usages:

1. User specifies fp32 "model", yaml configured calibration dataloader in calibration field and evaluation dataloader in evaluation field, metric in tuning.metric field of model-specific yaml config file.

> *Note*: 
> you should change the model-specific yaml file dataset path to your own dataset path

2. User specifies fp32 "model", calibration dataset "q_dataloader" and a custom "eval_func" which encapsulates the evaluation dataset and metric by itself.

As Inception V1 is a typical image recognition model, use Top-K as metric which is built-in supported by Intel® Low Precision Optimization Tool. It's easy to directly use 1 method that to configure a yaml file.

### Write Yaml config file

In examples directory, there is a template.yaml. We could remove most of items and only keep mandotory item for tuning. 


```yaml
# inceptionv1.yaml

model:                                               # mandatory. lpot uses this model name and framework name to decide where to save tuning history and deploy yaml.
  name: inceptionv1
  framework: tensorflow                              # mandatory. supported values are tensorflow, pytorch, pytorch_ipex, onnxrt_integer, onnxrt_qlinear or mxnet; allow new framework backend extension.
  inputs: input
  outputs: InceptionV1/Logits/Predictions/Reshape_1

quantization:                                        # optional. tuning constraints on model-wise for advance user to reduce tuning space.
  calibration:
    sampling_size: 5, 10                             # optional. default value is 100. used to set how many samples should be used in calibration.
    dataloader:
      dataset:
        ImageRecord:
          root: /path/to/calibration/dataset         # NOTE: modify to calibration dataset location if needed
      transform:
        ParseDecodeImagenet:
        ResizeCropImagenet: 
          height: 224
          width: 224
          mean_value: [123.68, 116.78, 103.94]
  model_wise:                                        # optional. tuning constraints on model-wise for advance user to reduce tuning space.
    activation:
      algorithm: minmax

evaluation:                                          # optional. required if user doesn't provide eval_func in lpot.Quantization.
  accuracy:                                          # optional. required if user doesn't provide eval_func in lpot.Quantization.
    metric:
      topk: 1                                        # built-in metrics are topk, map, f1, allow user to register new metric.
    dataloader:
      batch_size: 10
      dataset:
        ImageRecord:
          root: /path/to/evaluation/dataset          # NOTE: modify to evaluation dataset location if needed
      transform:
        ParseDecodeImagenet:
        ResizeCropImagenet: 
          height: 224
          width: 224
          mean_value: [123.68, 116.78, 103.94]
  performance:                                       # optional. used to benchmark performance of passing model.
    configs:
      cores_per_instance: 4
      num_of_instance: 7
    dataloader:
      batch_size: 1 
      dataset:
        ImageRecord:
          root: /path/to/evaluation/dataset          # NOTE: modify to evaluation dataset location if needed
      transform:
        ParseDecodeImagenet:
        ResizeCropImagenet: 
          height: 224
          width: 224
          mean_value: [123.68, 116.78, 103.94]

tuning:
  accuracy_criterion:
    relative:  0.01                                  # optional. default value is relative, other value is absolute. this example allows relative accuracy loss: 1%.
  exit_policy:
    timeout: 0                                       # optional. tuning timeout (seconds). default value is 0 which means early stop. combine with max_trials field to decide when to exit.
  random_seed: 9527                                  # optional. random seed for deterministic tuning.

```

Here we choose topk built-in metric and set accuracy target as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means early stop as well as a tuning config meet accuracy target.

### prepare

There are three preparation steps in here:
1. Prepare environment
```shell
pip install intel-tensorflow==1.15.2 lpot
```
2. Prepare the ImageNet dataset and pretrainined ckpt file
```shell
wget http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz
```

### code update

This tool support tune and benchmark the model, when in the tune phase, make sure to use get_slim_graph to get the slim graph and thransfer to the tool

```python

    from lpot.experimental import Quantization
    from lpot.adaptor.tf_utils.util import get_slim_graph
    quantizer = Quantization(self.args.config)
    slim_graph = get_slim_graph(args.input_graph, model_func, arg_scope, images, **kwargs)
    q_model = quantizer(slim_graph)
    save(q_model, args.output_graph)
```

when in benchmark phase:

```python

    from lpot.experimental import Benchmark
    evaluator = Benchmark(args.config)
    results = evaluator(model=args.input_graph)
```
