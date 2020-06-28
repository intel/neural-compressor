Step-by-Step
============

This document is used to list steps of reproducing PyTorch DLRM iLiT tuning zoo result.

> **Note**
>
> PyTorch quantization implementation in imperative path has limitation on automatically execution.
> It requires to manually add QuantStub and DequantStub for quantizable ops, it also requires to manually do fusion operation.
> iLiT has no capability to solve this framework limitation. iLiT supposes user have done these two steps before invoking iLiT interface.
> For details, please refer to https://pytorch.org/docs/stable/quantization.html

# Prerequisite

### 1. Installation

  ```Shell
  # Install iLiT
  pip install ilit

  # Install PyTorch
  pip install torch==1.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

  # Install sklearn
  pip install scikit-learn
  ```

### 2. Prepare Dataset

  The code supports interface with the [Criteo Terabyte Dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/)
  1. download the raw data files day_0.gz, ...,day_23.gz and unzip them Specify the location of the unzipped text files day_0, ...,day_23, using --raw-data-file=<path/day> (the day number will be appended automatically)
  2. These are then pre-processed (categorize, concat across days...) to allow using with dlrm  code
  3. The processed data is stored as .npz file in <root_dir>/input/.npz
  4. The processed file (.npz) can be used for subsequent runs with --processed-data-file=<path/  . npz>
   
### 3. Prepare pretrained model
  Corresponding pre-trained model is available under [CC-BY-NC license](https://creativecommons.org/licenses/by-nc/2.0/) and can be downloaded here [dlrm_emb64_subsample0.   875_maxindrange10M_pretrained.pt](https://dlrm.s3-us-west-1.amazonaws.com/models/tb0875_10M.pt)

# Run

  ```Shell
  cd examples/pytorch/dlrm
  ./run_and_time.sh
  ```

Examples of enabling iLiT
=========================

This is a tutorial of how to enable DLRM model with iLiT.

# User Code Analysis

iLiT supports two usages:

1. User specifies fp32 'model', calibration dataset 'q_dataloader', evaluation dataset "eval_dataloader" and metrics in tuning.metrics field of model-specific yaml config file.

2. User specifies fp32 'model', calibration dataset 'q_dataloader' and a custom "eval_func" which encapsulates the evaluation dataset and metrics by itself.

As DLRM's matrics is 'f1', so customer should provide evaluation function 'eval_func', it's suitable for the second use case.

### Write Yaml config file
In examples directory, there is conf.yaml. We could remove most of items and only keep mandotory item for tuning.
```
framework:
  - name: pytorch

device: cpu

tuning:
    accuracy_criterion:
      - relative: 0.01
    timeout: 0
    random_seed: 9527
```
Here we set accuracy target as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means early stop as well as a tuning config meet accuracy target.
> **Note** : iLiT tool don't support "mse" tuning strategy for pytorch framework

### prepare
PyTorch quantization requires two manual steps:

  1. Add QuantStub and DeQuantStub for all quantizable ops.
  2. Fuse possible patterns, such as Linear + Relu.

It's intrinsic limitation of PyTorch quantizaiton imperative path. No way to develop a code to automatically do that.
The related code changes please refer to examples/pytorch/dlrm/dlrm_s_pytorch_tune.py.

### code update
After prepare step is done, we just need update run_squad_tune.py and run_glue_tune.py like below
```
class DLRM_DataLoader(DataLoader):
    def __init__(self, loader=None):
        self.loader = loader
    def __iter__(self):
        for X_test, lS_o_test, lS_i_test, T in self.loader:
            yield (X_test, lS_o_test, lS_i_test), T
```

```
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
import ilit
tuner = ilit.Tuner("./conf.yaml")
tuner.tune(dlrm, eval_dataloader, eval_func=eval_func)
```
