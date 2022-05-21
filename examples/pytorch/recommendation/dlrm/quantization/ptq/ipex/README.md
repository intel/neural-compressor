Step-by-Step
============

This document is used to list steps of reproducing PyTorch DLRM tuning zoo result. and original DLRM README is in [DLRM README](https://github.com/facebookresearch/dlrm/blob/master/README.md)

> **Note**
>
> Please  ensure your PC have >370G memory to run DLRM
> IPEX version >= 1.10

# Prerequisite

### 1. Installation

PyTorch 1.11 or higher version is needed with pytorch_fx backend.

  ```shell
  # Install dependency
  cd examples/pytorch/recommendation/dlrm/quantization/ptq/ipex
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
  cd examples/pytorch/recommendation/dlrm/quantization/ptq/ipex
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

Here we used the second use case.

### Write Yaml config file

In examples directory, there is conf.yaml. We could remove most of the items and only keep mandatory item for tuning.

```yaml
model:
  name: dlrm
  framework: pytorch_ipex

device: cpu

quantization:
  calibration:
    sampling_size: 102400

tuning:
  accuracy_criterion:
    relative: 0.01
  exit_policy:
    timeout: 0
  random_seed: 9527
```

Here we set accuracy target as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means early stop as well as a tuning config meet accuracy target.
> **Note** : Intel® Neural Compressor does NOT support "mse" tuning strategy for pytorch framework

### code update

We need update dlrm_s_pytorch.py like below

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
from neural_compressor.experimental import Quantization, common

def eval_func(model):
    args.int8 = False if model.ipex_config_path is None else True
    args.int8_configure = "" \
        if model.ipex_config_path is None else model.ipex_config_path
    with torch.no_grad():
        return inference(
            args,
            model,
            best_acc_test,
            best_auc_test,
            test_ld,
            trace=args.int8
        )
assert args.inference_only, "Please set inference_only in arguments"
quantizer = Quantization("./conf_ipex.yaml")
quantizer.model = common.Model(dlrm)
quantizer.calib_dataloader = DLRM_DataLoader(train_ld)
quantizer.eval_func = eval_func
q_model = quantizer.fit()
q_model.save("/path/to/save/results")
return
```
