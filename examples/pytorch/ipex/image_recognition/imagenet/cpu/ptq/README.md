Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch ResNet50/ResNet18/ResNet101 tuning results with Intel® Low Precision Optimization Tool(LPOT).

# Prerequisite

### 1. Installation
We verified examples with IPEX backend on Python 3.8, recommended.

```shell
pip install -r requirements.txt
```

### 2. Install pytorch and intel-pytorch-extension

refer [intel/intel-extension-for-pytorch at icx (github.com)](https://github.com/intel/intel-extension-for-pytorch/tree/v1.8.0)

1. install PyTorch1.8 and TorchVision0.9

   refer [PyTorch install](https://pytorch.org/get-started/locally/)
   ```shell position-relative
   pip3 install torch==1.8.0+cpu torchvision==0.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
   ```
2. Get Intel PyTorch Extension source and install
    > **Note**
    >
    > GCC9 compiler is recommended
    >

   ```shell position-relative
   git clone https://github.com/intel/intel-extension-for-pytorch
   cd intel-extension-for-pytorch
   git checkout v1.8.0
   git submodule sync
   git submodule update --init --recursive

   python setup.py install
   ```


### 3. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet.  The dir include below folder:

```bash
ls /path/to/imagenet
train  val
```

# Run

### 1. ResNet18 With Intel PyTorch Extension

```shell
cd examples/pytorch/ipex/image_recognition/imagenet/cpu/ptq
python main.py -t -a resnet18 --ipex --pretrained /path/to/imagenet
```
or
```shell
cd examples/pytorch/ipex/image_recognition/imagenet/cpu/ptq
bash run_tuning.sh --topology=resnet18_ipex --dataset_location=/path/to/imagenet
bash run_benchmark.sh --topology=resnet18_ipex --dataset_location=/path/to/imagenet --mode=benchmark/accuracy --int8=true/false
```

### 2. ResNet50 With Intel PyTorch Extension

```shell
cd examples/pytorch/ipex/image_recognition/imagenet/cpu/ptq
python main.py -t -a resnet50 --ipex --pretrained /path/to/imagenet
```
or
```shell
cd examples/pytorch/ipex/image_recognition/imagenet/cpu/ptq
bash run_tuning.sh --topology=resnet50_ipex --dataset_location=/path/to/imagenet
bash run_benchmark.sh --topology=resnet50_ipex --dataset_location=/path/to/imagenet --mode=benchmark/accuracy --int8=true/false
```

### 3. ResNext101_32x8d With Intel PyTorch Extension

```shell
cd examples/pytorch/ipex/image_recognition/imagenet/cpu/ptq
python main.py -t -a resnext101_32x8d --ipex --pretrained /path/to/imagenet
```
or
```shell
cd examples/pytorch/ipex/image_recognition/imagenet/cpu/ptq
bash run_tuning.sh --topology=resnext101_32x8d_ipex --dataset_location=/path/to/imagenet
bash run_benchmark.sh --topology=resnext101_32x8d_ipex --dataset_location=/path/to/imagenet --mode=benchmark/accuracy --int8=true/false
```

### 4. Mobilenet_v2 With Intel PyTorch Extension

```shell
cd examples/pytorch/ipex/image_recognition/imagenet/cpu/ptq
python main.py -t -a mobilenet_v2 --ipex --pretrained /path/to/imagenet
```
or
```shell
cd examples/pytorch/ipex/image_recognition/imagenet/cpu/ptq
bash run_tuning.sh --topology=mobilenet_v2_ipex --dataset_location=/path/to/imagenet
bash run_benchmark.sh --topology=mobilenet_v2_ipex --dataset_location=/path/to/imagenet --mode=benchmark/accuracy --int8=true/false
```
# Saving and loading model:

* Saving model:
  After tuning with LPOT, we can get LPOT.model:

```
from lpot.experimental import Quantization, common
quantizer = Quantization("./conf.yaml")
quantizer.model = common.Model(model)
lpot_model = quantizer()
```

Here, lpot_model is LPOT model class, so it has "save" API:

```python
lpot_model.save("Path_to_save_configure_file")
```

* loading model:

```python
# With IPEX
import intel_pytorch_extension as ipex 
model                 # fp32 model
model.to(ipex.DEVICE)
try:
    new_model = torch.jit.script(model)
except:
    new_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224).to(ipex.DEVICE))
ipex_config_path = os.path.join(os.path.expanduser(args.tuned_checkpoint),
                                "best_configure.json")
conf = ipex.AmpConf(torch.int8, configure_file=ipex_config_path)
with torch.no_grad():
    for i, (input, target) in enumerate(val_loader):
        with ipex.AutoMixPrecision(conf, running_mode='inference'):
            output = new_model(input.to(ipex.DEVICE))
```

Please refer to [Sample code](./main.py).

Examples of enabling LPOT auto tuning on PyTorch ResNet
=======================================================

This is a tutorial of how to enable a PyTorch classification model with LPOT.

# User Code Analysis

LPOT supports three usages:

1. User only provide fp32 "model", and configure calibration dataset, evaluation dataset and metric in model-specific yaml config file.
2. User provide fp32 "model", calibration dataset "q_dataloader" and evaluation dataset "eval_dataloader", and configure metric in tuning.metric field of model-specific yaml config file.
3. User specifies fp32 "model", calibration dataset "q_dataloader" and a custom "eval_func" which encapsulates the evaluation dataset and metric by itself.

As ResNet18/50/101 series are typical classification models, use Top-K as metric which is built-in supported by LPOT. So here we integrate PyTorch ResNet with LPOT by the first use case for simplicity.

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

1. Tuning With LPOT

    ```python
      from lpot.experimental import Quantization, common
      quantizer = Quantization("./conf_ipex.yaml")
      quantizer.model = common.Model(model)
      lpot_model = quantizer()
      lpot_model.save("Path_to_save_configure_file")
    ```

2. Saving and Run ipex model

    * Saving model

    ```python
      lpot_model.save("Path_to_save_configure_file")
    ```

    Here, lpot_model is the result of LPOT tuning. It is LPOT.model class, so it has "save"     API.

    * Run ipex model:

    ```python
    import intel_pytorch_extension as ipex 
    model                 # fp32 model
    model.to(ipex.DEVICE)
    try:
        new_model = torch.jit.script(model)
    except:
        new_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224).to(ipex.DEVICE))
    ipex_config_path = os.path.join(os.path.expanduser(args.tuned_checkpoint),
                                    "best_configure.json")
    conf = ipex.AmpConf(torch.int8, configure_file=ipex_config_path)
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            with ipex.AutoMixPrecision(conf, running_mode='inference'):
                output = new_model(input.to(ipex.DEVICE))
    ```
