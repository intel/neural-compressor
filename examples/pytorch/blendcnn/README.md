Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch BlendCNN tuning(with MRPC dataset) results with Intel® Low Precision Optimization Tool.

> **Note**
>
> PyTorch quantization implementation in imperative path has limitation on automatically execution.
> It requires to manually add QuantStub and DequantStub for quantizable ops, it also requires to manually do fusion operation.
> Intel® Low Precision Optimization Tool has no capability to solve this framework limitation. Intel® Low Precision Optimization Tool supposes user have done these two steps before invoking Intel® Low Precision Optimization Tool interface.
> For details, please refer to https://pytorch.org/docs/stable/quantization.html

# Prerequisite

## 1. Installation

```Shell
cd examples/pytorch/blendcnn
pip install -r requirements.txt
pip install torch==1.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

## 2. Prepare model and Dataset

Download [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) and
[GLUE MRPC Benchmark Datasets](https://github.com/nyu-mll/GLUE-baselines)

### model

```Shell
mkdir models/ && mv uncased_L-12_H-768_A-12.zip models/
cd models/ && unzip uncased_L-12_H-768_A-12.zip

# train blend CNN from scratch
python classify.py --model_config config/blendcnn/mrpc/train.json
```

After below steps, you can find the pre-trained model weights ***model_final.pt*** at `./models/`

### dataset

After downloads dataset, you need to put dataset at `./MRPC/`, list this:

```Shell
ls MRPC/
dev_ids.tsv  dev.tsv  test.tsv  train.tsv
```

## Run

### blendcnn

```Shell
./run_tuning.sh --input_model=/PATH/TO/models/ --dataset_location=/PATH/TO/MRPC/ --output_model=/DIR/TO/INT8_MODEL/

./run_benchmark.sh --int8=true --mode=benchmark --batch_size=32
./run_benchmark.sh --int8=False --mode=benchmark --batch_size=32 --input_model=/PATH/TO/FP32_MODEL

```

Examples of enabling Intel® Low Precision Optimization Tool auto tuning on PyTorch ResNest
===========================================================================================

This is a tutorial of how to enable a PyTorch classification model with Intel® Low Precision Optimization Tool.

## User Code Analysis

Intel® Low Precision Optimization Tool supports three usages:

1. User only provide fp32 "model", and configure calibration dataset, evaluation dataset and metric in model-specific yaml config file.
2. User provide fp32 "model", calibration dataset "q_dataloader" and evaluation dataset "eval_dataloader", and configure metric in tuning.metric field of model-specific yaml config file.
3. User specifies fp32 "model", calibration dataset "q_dataloader" and a custom "eval_func" which encapsulates the evaluation dataset and metric by itself.

As ResNest series are typical classification models, use Top-K as metric which is built-in supported by Intel® Low Precision Optimization Tool. So here we integrate PyTorch ResNest with Intel® Low Precision Optimization Tool by the first use case for simplicity.

### Write Yaml config file

In examples directory, there is a template.yaml. We could remove most of items and only keep mandotory item for tuning.

```
model:                                               # mandatory. lpot uses this model name and framework name to decide where to save tuning history and deploy yaml.
  name: blendcnn
  framework: pytorch     

tuning:
  accuracy_criterion:
    relative:  0.01                                  # optional. default value is relative, other value is absolute. this example allows relative accuracy loss: 1%.
  exit_policy:
    timeout: 0                                       # optional. tuning timeout (seconds). default value is 0 which means early stop. combine with max_trials field to decide when to exit.
  random_seed: 9527                                  # optional. random seed for deterministic tuning.

```

Here we use specifies fp32 "model", calibration dataset "q_dataloader" and a custom "eval_func" which encapsulates the evaluation dataset and metric.

### prepare

PyTorch quantization requires two manual steps:

1. Add QuantStub and DeQuantStub for all quantizable ops.
2. Fuse possible patterns, such as Conv + Relu and Conv + BN + Relu.

It's intrinsic limitation of PyTorch quantizaiton imperative path. No way to develop a code to automatically do that.(Please refer [sample code](./models.py))

### code update

After prepare step is done, we just need update classify.py like below.

```
from lpot.experimental import Quantization
dataloader = Bert_DataLoader(loader=data_iter, batch_size=args.batch_size)
quantizer = Quantization(args.lpot_yaml)
quantizer.model = model
quantizer.calib_dataloader = dataloader
quantizer.eval_func = eval_func
q_model = quantizer()
```

The quantizer() function will return a best quantized model during timeout constrain.(Please refer [sample code](./classify.py))
