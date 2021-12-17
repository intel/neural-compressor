Original DLRM README
============

Please refer [DLRM README](https://github.com/facebookresearch/dlrm/blob/master/README.md)

Step-by-Step
============

This document is used to list steps of reproducing Engine DLRM tuning zoo result.

> **Note**
>
> 1. For engine support DLRM with "dot" and "cat" interaction layer
> 2. Please  ensure your PC have >370G memory to run DLRM 

# Prerequisite

### 1. Installation

PyTorch 1.8 or higher version is needed.

  ```shell
  # Install dependency
  pip install -r requirements.txt
  ```

### 2. Prepare Dataset

  The code supports interface with the [Criteo Kaggle Display Advertising Challenge Dataset](https://ailab.criteo.com/ressources/).
    1. download the raw data files file (train.txt).
    2. This is then pre-processed (categorize, concat across days...) to allow using with dlrm code.

### 2. Prepare the DLRM ONNX model

  ```shell
  python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=32 --print-freq=1024 --test-freq=20480 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --arch-interaction-op=cat --mlperf-acc-threshold=0.77 --save-onnx --mlperf-logging 
  ```

Examples of enabling DLRM with Engine
=========================

This is a tutorial of how to enable DLRM model with Intel® Neural Compressor.

# User Code Analysis

Example based on the DLRM ONNX model with "cat" interaction layer

### Write Yaml config file
In examples directory, there is conf.yaml(for int8 quantization and conf_bf16.yaml for bf16). We could remove most of the items and only keep mandatory item for tuning.
```yaml
model:
  name: dlrm
  framework: engine

evaluation:
  accuracy:
    metric:
       ROC:
        task: dlrm
  performance:
    warmup: 5
    iteration: 10
    configs:
      num_of_instance: 1
      cores_per_instance: 28

quantization:
  dtype: bf16              # int8 for 8bit quantization
  calibration:
    sampling_size: 2000

tuning:
  accuracy_criterion:
    relative: 0.01
  exit_policy:
    timeout: 0
  random_seed: 9527
```
Here we set accuracy target as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means early stop as well as a tuning config meet accuracy target.

### RUN
After prepare step is done, we just run tune and benchmark

Accuracy run by python

```python
GLOG_minloglevel=2 python run_engine.py --tune --input_model=dlrm_s_pytorch.onnx --batch_size=32
```

Or run shell

```python
./run_tuning.sh --input_model=./dlrm_s_pytorch.onnx  --dataset_location=./input --config=./conf.yaml
```

Benchmark run by python

```python
GLOG_minloglevel=2 python run_engine.py --benchmark --input_model=dlrm_s_pytorch.onnx --batch_size=32
```

Or run shell

```python
./run_benchmark.sh --input_model=./ir  --dataset_location=./input --config=./conf.yaml --batch_size=32
```

