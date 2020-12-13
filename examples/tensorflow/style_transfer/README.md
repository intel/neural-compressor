Step-by-Step
============

This document is used to list steps of reproducing TensorFlow style transfer Intel® Low Precision Optimization Tool tuning zoo result.


## Prerequisite

### 1. Installation
```Shell
# Install Intel® Low Precision Optimization Tool
pip instal lpot
```
### 2. Install Intel Tensorflow 1.15/2.0/2.1
```shell
pip intel-tensorflow==1.15.2 [2.0,2.1]
```

### 3. Install Additional Dependency packages
```shell
cd examples/tensorflow/style_transfer && pip install -r requirements.txt
```

### 4. Prepare Dataset
There are two folders named style_images and content_images
you can use these two folders to generated stylized images for test
you can also prepare your own style_images or content_images

### 5. Prepare Pretrained model

#### Automated approach
Run the `prepare_model.py` script located in `LowPrecisionInferenceTool/examples/tensorflow/style_transfer`.

```
usage: prepare_model.py [-h] [--model_path MODEL_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH directory to put models, default is ./model
```

#### Manual approach

```shell
wget https://storage.googleapis.com/download.magenta.tensorflow.org/models/arbitrary_style_transfer.tar.gz
tar -xvzf arbitrary_style_transfer.tar.gz ./model
```

## Run Command
  ```Shell
  python style_tune.py --output_dir=./result --style_images_paths=./style_images --content_images_paths=./content_images --model_dir=./model --precision=quantized
  ```

Details of enabling Intel® Low Precision Optimization Tool on style transfer for Tensorflow.
=========================

This is a tutorial of how to enable style_transfer model with Intel® Low Precision Optimization Tool.
## User Code Analysis
1. User specifies fp32 *model*, calibration dataset *q_dataloader*, evaluation dataset *eval_dataloader* and metric in tuning.metric field of model-specific yaml config file.

2. User specifies fp32 *model*, calibration dataset *q_dataloader* and a custom *eval_func* which encapsulates the evaluation dataset and metric by itself.

For style_transfer, we applied the latter one because we don't have metric for style transfer model.The first one is to implement the q_dataloader and implement a fake *eval_func*. As lpot have implement a style_transfer dataset, so only eval_func should be prepared after load the graph

### Evaluation Part Adaption
As style transfer don't have a metric to measure the accuracy, we only implement a fake eval_func
```python
def eval_func(model):
    return 1.
```

### Write Yaml config file
In examples directory, there is a conf.yaml. We could remove most of items and only keep mandatory item for tuning. We also implement a calibration dataloader

```yaml
model:
  name: style_transfer
  framework: tensorflow
  inputs: import/style_input,import/content_input
  outputs: import/transformer/expand/conv3/conv/Sigmoid

quantization:
  calibration:
    dataloader:
      batch_size: 2
      dataset:
        style_transfer:
          content_folder: ./content_images/          # NOTE: modify to content images path if needed
          style_folder: ./style_images/              # NOTE: modify to style images path if needed

evaluation:
  accuracy:
    dataloader:
      batch_size: 2
      dataset:
        style_transfer:
          content_folder: ./content_images/          # NOTE: modify to content images path if needed
          style_folder: ./style_images/              # NOTE: modify to style images path if needed

tuning:
    accuracy_criterion:
      relative: 0.01
    exit_policy:
      timeout: 0
    random_seed: 9527
```
Here we set the input tensor and output tensors name into *inputs* and *outputs* field. In this case we only calibration and quantize the model without tune the accuracy

### Code update

After prepare step is done, we just need add 2 lines to get the quantized model.
```python
from lpot import Quantization

quantizer = Quantization(args.config)
q_model = quantizer(graph, eval_func=eval_func)
```

The Intel® Low Precision Optimization Tool quantizer() function will return a best quantized model during timeout constrain.
