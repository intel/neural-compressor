Step-by-Step
============

This document is used to list steps of reproducing TensorFlow style transfer Intel® Neural Compressor tuning zoo result.
This example can run on Intel CPUs and GPUs.

# Prerequisite

## Prerequisite

### Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```
### Install Intel Tensorflow
```shell
pip install intel-tensorflow
```
> Note: Supported Tensorflow [Version](../../../../../../README.md#supported-frameworks).

### Install Additional Dependency packages
```shell
cd examples/tensorflow/style_transfer/arbitrary_style_transfer/quantization/ptq 
pip install -r requirements.txt
```

### Install Intel Extension for Tensorflow
#### Quantizing the model on Intel GPU(Mandatory to install ITEX)
Intel Extension for Tensorflow is mandatory to be installed for quantizing the model on Intel GPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[xpu]
```
For any more details, please follow the procedure in [install-gpu-drivers](https://github.com/intel/intel-extension-for-tensorflow/blob/main/docs/install/install_for_xpu.md#install-gpu-drivers)

#### Quantizing the model on Intel CPU(Optional to install ITEX)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's not mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

> **Note**: 
> The version compatibility of stock Tensorflow and ITEX can be checked [here](https://github.com/intel/intel-extension-for-tensorflow#compatibility-table). Please make sure you have installed compatible Tensorflow and ITEX.

### 2. Prepare Pretrained model

#### Automated approach
Run the `prepare_model.py` script located in `./examples/tensorflow/style_transfer/arbitrary_style_transfer/quantization/ptq`.

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

### 3. Prepare Dataset
There are two folders named style_images and content_images in current folder. Please use these two folders to generated stylized images for test. And you can also prepare your own style_images or content_images.


# Run Command
  ```shell
  python style_tune.py --output_dir=./result --style_images_paths=./style_images --content_images_paths=./content_images --input_model=./model/model.ckpt
  ```


## Quantization Config

The Quantization Config class has default parameters setting for running on Intel CPUs. If running this example on Intel GPUs, the 'backend' parameter should be set to 'itex' and the 'device' parameter should be set to 'gpu'.

```
config = PostTrainingQuantConfig(
    device="gpu",
    backend="itex",
    ...
    )
```

## Quantization
  ```shell
  bash run_quant.sh --dataset_location=style_images/,content_images/ --input_model=./model/model.ckpt --output_model=saved_model
  ```
## Benchmark
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

Here we set the input tensor and output tensors name into *inputs* and *outputs* field. In this case we only calibration and quantize the model without tune the accuracy

### Code update

After prepare step is done, we just need add 2 lines to get the quantized model.
```python
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig
conf = PostTrainingQuantConfig(inputs=['style_input', 'content_input'],
                                outputs=['transformer/expand/conv3/conv/Sigmoid'],
                                calibration_sampling_size=[50, 100])
quantized_model = quantization.fit(args.input_graph, conf=conf, calib_dataloader=dataloader,
            eval_dataloader==dataloader)
```

The Intel® Neural Compressor quantizer.fit() function will return a best quantized model during timeout constrain.
