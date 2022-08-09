Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch ResNet50/ResNet18/ResNet101 tuning results with Intel® Neural Compressor.

# Prerequisite

### 1. Installation
We verified examples with IPEX backend on Python 3.8, recommended.

```shell
cd examples/pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/ipex
pip install -r requirements.txt
```

### 2. Install pytorch and intel-pytorch-extension

refer [intel/intel-extension-for-pytorch at icx (github.com)](https://github.com/intel/intel-extension-for-pytorch/tree/v1.8.0)

1. install PyTorch and TorchVision

   refer [PyTorch install](https://pytorch.org/get-started/locally/)
   ```shell position-relative
    pip install requirements.txt
   ```
2. Get  Intel® Extension for PyTorch* source and install
    > **Note**
    >
    > GCC9 compiler is recommended
    >

   ```shell position-relative
   python -m pip install intel_extension_for_pytorch -f https://software.intel.com/ipex-whl-stable
   ```


   ```
   # build from source for IPEX 1.12
    git clone https://github.com/intel/intel-extension-for-pytorch/
    cd intel-extension-for-pytorch
    git submodule sync && git submodule update --init --recursive
    git checkout 1279c5824f1bcb61cd8990f4148abcadf3f214a4
    git apply ../patch.patch
    python setup.py install
   ```
   > Note: Intel® Extension for PyTorch* has PyTorch version > requirement. Please check more detailed information via > the URL below.
   >
   > GCC9 compiler is recommended
   >
   > Support IPEX version >= 1.8.0, 1.12.0 version need build from source and apply patch.
### 3. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet.  The dir include below folder:

```bash
ls /path/to/imagenet
train  val
```

# Run

### 1. ResNet18 With Intel PyTorch Extension

```shell
python main.py -t -a resnet18 --ipex --pretrained /path/to/imagenet
```
or
```shell
bash run_tuning.sh --topology=resnet18_ipex --dataset_location=/path/to/imagenet
bash run_benchmark.sh --topology=resnet18_ipex --dataset_location=/path/to/imagenet --mode=benchmark/accuracy --int8=true/false
```

### 2. ResNet50 With Intel PyTorch Extension

```shell
python main.py -t -a resnet50 --ipex --pretrained /path/to/imagenet
```
or
```shell
bash run_tuning.sh --topology=resnet50_ipex --dataset_location=/path/to/imagenet
bash run_benchmark.sh --topology=resnet50_ipex --dataset_location=/path/to/imagenet --mode=benchmark/accuracy --int8=true/false
```

### 3. ResNext101_32x16d With Intel PyTorch Extension

```shell
python main.py -t -a resnext101_32x16d_wsl --hub --ipex --pretrained /path/to/imagenet
```
or
```shell
bash run_tuning.sh --topology=resnext101_32x16d_wsl_ipex --dataset_location=/path/to/imagenet
bash run_benchmark.sh --topology=resnext101_32x16d_wsl_ipex --dataset_location=/path/to/imagenet --mode=benchmark/accuracy --int8=true/false
```


# Saving model:

* Saving model:
  After tuning with Neural Compressor, we can get neural_compressor.model:

```
from neural_compressor.experimental import Quantization, common
quantizer = Quantization("./conf.yaml")
quantizer.model = common.Model(model)
nc_model = quantizer.fit()
```

Here, nc_model is Neural Compressor model class, so it has "save" API:

```python
nc_model.save("Path_to_save_configure_file")
```

Please refer to [Sample code](./main.py).

Examples of enabling Neural Compressor auto tuning on PyTorch ResNet
=======================================================

This is a tutorial of how to enable a PyTorch classification model with Neural Compressor.

# User Code Analysis

Neural Compressor supports three usages:

1. User only provide fp32 "model", and configure calibration dataset, evaluation dataset and metric in model-specific yaml config file.
2. User provide fp32 "model", calibration dataset "q_dataloader" and evaluation dataset "eval_dataloader", and configure metric in tuning.metric field of model-specific yaml config file.
3. User specifies fp32 "model", calibration dataset "q_dataloader" and a custom "eval_func" which encapsulates the evaluation dataset and metric by itself.

As ResNet18/50/101 series are typical classification models, use Top-K as metric which is built-in supported by Neural Compressor. So here we integrate PyTorch ResNet with Neural Compressor by the first use case for simplicity.

### Write Yaml Config File

    In examples directory, there is a template.yaml. We could remove most of items and only     keep mandatory item for tuning.

    ```yaml
    model:
      name: imagenet_ptq
      framework: pytorch_ipex

    quantization:
      calibration:
        sampling_size: 300
        dataloader:
          dataset:
            ImageFolder:
              root: /path/to/calibration/dataset
          transform:
            RandomResizedCrop:
              size: 224
            RandomHorizontalFlip:
            ToTensor:
            Normalize:
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]

    evaluation:
      accuracy:
        metric:
          topk: 1
        dataloader:
          batch_size: 30
          dataset:
            ImageFolder:
              root: /path/to/evaluation/dataset
          transform:
            Resize:
              size: 256
            CenterCrop:
              size: 224
            ToTensor:
            Normalize:
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
      performance:
        configs:
          cores_per_instance: 4
          num_of_instance: 7
        dataloader:
          batch_size: 1
          dataset:
            ImageFolder:
              root: /path/to/evaluation/dataset
          transform:
            Resize:
              size: 256
            CenterCrop:
              size: 224
            ToTensor:
            Normalize:
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]

    tuning:
      accuracy_criterion:
        relative:  0.01
      exit_policy:
        timeout: 0
      random_seed: 9527

    ```

    Here we choose topk built-in metric and set accuracy target as tolerating 0.01 relative     accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0     means unlimited time for a tuning config meet accuracy target.

### Prepare

The related code please refer to examples/pytorch/ipex/image_recognition/imagenet/cpu/ptq/main.py.

### Tuning With Intel PyTorch Extension

Tuning With Neural Compressor

    ```python
      from neural_compressor.experimental import Quantization, common
      quantizer = Quantization("./conf_ipex.yaml")
      quantizer.model = common.Model(model)
      nc_model = quantizer.fit()
      nc_model.save("Path_to_save_configure_file")
    ```

