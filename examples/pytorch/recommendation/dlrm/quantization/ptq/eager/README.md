Step-by-Step
============

This document is used to list steps of reproducing PyTorch DLRM tuning zoo result. and original DLRM README is in [DLRM README](https://github.com/facebookresearch/dlrm/blob/master/README.md)

> **Note**
>
> 1. PyTorch quantization implementation in imperative path has limitation on automatically execution.
> It requires to manually add QuantStub and DequantStub for quantizable ops, it also requires to manually do fusion operation.
> Intel® Neural Compressor has no capability to solve this framework limitation. Intel® Neural Compressor supposes user have done these two steps before invoking Intel® Neural Compressor interface.
> For details, please refer to https://pytorch.org/docs/stable/quantization.html
> 2. Please  ensure your PC have >370G memory to run DLRM 

# Prerequisite

### 1. Installation
  Recommend python 3.6 or higher version.

  ```shell
  # Install dependency
  cd examples/pytorch/recommendation/dlrm/quantization/ptq/eager
  pip install -r requirements.txt
  ```

### 2. Prepare Dataset

  The code supports interface with the [Criteo Terabyte Dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/)
  1. download the raw data files day_0.gz, ...,day_23.gz and unzip them.
  2. Specify the location of the unzipped text files day_0, ...,day_23, using --raw-data-file=<path/day> (the day number will be appended automatically), please refer "Run" command.

### 3. Prepare pretrained model
  Download the DLRM PyTorch weights (`tb00_40M.pt`, 90GB) from the
[MLPerf repo](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm/pytorch#more-information-about-the-model-weights)

# Run
### tune with INC
  ```shell
  cd examples/pytorch/recommendation/dlrm/quantization/ptq/eager
  bash run_tuning.sh --input_model="/path/of/pretrained/model" --dataset_location="/path/of/dataset"
  ```

### benchmark
```shell
bash run_benchmark.sh --input_model="/path/of/pretrained/model" --dataset_location="/path/of/dataset" --mode=accuracy --int8=true
```

Examples of enabling Intel® Neural Compressor
=========================

This is a tutorial of how to enable DLRM model with Intel® Neural Compressor.

# User Code Analysis

Intel® Neural Compressor supports two usages:

1. User specifies fp32 'model', calibration dataset 'q_dataloader', evaluation dataset "eval_dataloader" and metrics in tuning.metrics field of model-specific yaml config file.

2. User specifies fp32 'model', calibration dataset 'q_dataloader' and a custom "eval_func" which encapsulates the evaluation dataset and metrics by itself.

Here we use the second use case.

### Write Yaml config file
In examples directory, there is conf.yaml. We could remove most of the items and only keep mandatory item for tuning.
```yaml
model:
  name: dlrm
  framework: pytorch

device: cpu

tuning:
  accuracy_criterion:
    relative: 0.01
  exit_policy:
    timeout: 0
  random_seed: 9527
```
Here we set accuracy target as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means early stop as well as a tuning config meet accuracy target.
> **Note** : Intel® Neural Compressor does NOT support "mse" tuning strategy for pytorch framework

### prepare
PyTorch quantization requires two manual steps:

  1. Add QuantStub and DeQuantStub for all quantizable ops.
  2. Fuse possible patterns, such as Linear + Relu.

It's intrinsic limitation of PyTorch quantization imperative path. No way to develop a code to automatically do that.
The related code changes please refer to examples/pytorch/recommendation/dlrm/quantization/ptq/eager/dlrm_s_pytorch_tune.py.

### code update
After prepare step is done, we just need update run_squad_tune.py and run_glue_tune.py like below
```python
class DLRM_DataLoader(object):
    def __init__(self, loader=None):
        self.loader = loader
        self.batch_size = loader.dataset.batch_size
    def __iter__(self):
        for X_test, lS_o_test, lS_i_test, T in self.loader:
            yield (X_test, lS_o_test, lS_i_test), T
```

```python
eval_dataloader = DLRM_DataLoader(test_ld)
fuse_list = []
for i in range(0, len(dlrm.bot_l), 2):
    fuse_list.append(["bot_l.%d" % (i), "bot_l.%d" % (i + 1)])
dlrm = fuse_modules(dlrm, fuse_list)
fuse_list = []
for i in range(0, len(dlrm.top_l) - 2, 2):
    fuse_list.append(["top_l.%d" % (i), "top_l.%d" % (i + 1)])
dlrm = fuse_modules(dlrm, fuse_list)
dlrm.bot_l.insert(0, QuantStub())
dlrm.bot_l.append(DeQuantStub())
dlrm.top_l.insert(0, QuantStub())
dlrm.top_l.insert(len(dlrm.top_l) - 1, DeQuantStub())
from neural_compressor.experimental import Quantization, common
quantizer = Quantization("./conf.yaml")
quantizer.model = common.Model(dlrm)
quantizer.calib_dataloader = eval_dataloader
quantizer.eval_func = eval_func
quantizer.fit()
```

