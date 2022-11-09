Step-by-Step
============

This document is used to list steps of reproducing TensorFlow style transfer Intel® Neural Compressor tuning zoo result.
This example can run on Intel CPUs and GPUs.

## Prerequisite

### 1. Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```
### 2. Install Intel Tensorflow
```shell
pip install intel-tensorflow
```
> Note: Supported Tensorflow [Version](../../../../../../README.md#supported-frameworks).

### 3. Install Additional Dependency packages
```shell
cd examples/tensorflow/style_transfer/arbitrary_style_transfer/quantization/ptq 
pip install -r requirements.txt
```

### 4. Install Intel Extension for Tensorflow
#### Quantizing the model on Intel GPU
Intel Extension for Tensorflow is mandatory to be installed for quantizing the model on Intel GPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[gpu]
```
For any more details, please follow the procedure in [install-gpu-drivers](https://github.com/intel-innersource/frameworks.ai.infrastructure.intel-extension-for-tensorflow.intel-extension-for-tensorflow/blob/master/docs/install/install_for_gpu.md#install-gpu-drivers)

#### Quantizing the model on Intel CPU(Experimental)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's not mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

### 5. Prepare Dataset
There are two folders named style_images and content_images
you can use these two folders to generated stylized images for test
you can also prepare your own style_images or content_images

### 6. Prepare Pretrained model

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
  ```shell
  python style_tune.py --output_dir=./result --style_images_paths=./style_images --content_images_paths=./content_images --input_model=./model/model.ckpt
  ```
### Quantize with neural_compressor
#### 1. Tune model with neural_compressor
  ```shell
  bash run_tuning.sh --dataset_location=style_images/,content_images/ --input_model=./model/model.ckpt --output_model=saved_model
  ```
#### 2. check benchmark of tuned model
  ```shell
  bash run_benchmark.sh --dataset_location=style_images/,content_images/ --input_model=saved_model.pb --batch_size=1
  ```

Details of enabling Intel® Neural Compressor on style transfer for Tensorflow.
=========================

This is a tutorial of how to enable style_transfer model with Intel® Neural Compressor.
## User Code Analysis
1. User specifies fp32 *model*, calibration dataset *q_dataloader*, evaluation dataset *eval_dataloader* and metric in tuning.metric field of model-specific yaml config file.

2. User specifies fp32 *model*, calibration dataset *q_dataloader* and a custom *eval_func* which encapsulates the evaluation dataset and metric by itself.

For style_transfer, we applied the latter one because we don't have metric for style transfer model.The first one is to implement the q_dataloader and implement a fake *eval_func*. As neural_compressor have implement a style_transfer dataset, so only eval_func should be prepared after load the graph

### Evaluation Part Adaption
As style transfer don't have a metric to measure the accuracy, we only implement a fake eval_func
```python
def eval_func(model):
    return 1.
```

### Write Yaml config file
In examples directory, there is a conf.yaml for tuning the model on Intel CPUs. The 'framework' in the yaml is set to 'tensorflow'. If running this example on Intel GPUs, the 'framework' should be set to 'tensorflow_itex' and the device in yaml file should be set to 'gpu'. The conf_itex.yaml is prepared for the GPU case. We could remove most of items and only keep mandatory item for tuning. We also implement a calibration dataloader and have evaluation field for creation of evaluation function at internal neural_compressor.

```yaml
device: cpu                                          # NOTE: optional. default value is cpu, other value is gpu.

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
from neural_compressor.experimental import Quantization

quantizer = Quantization(args.config)
quantizer.model = graph
quantizer.eval_func = eval_func
q_model = quantizer.fit()
```

The Intel® Neural Compressor quantizer.fit() function will return a best quantized model during timeout constrain.
