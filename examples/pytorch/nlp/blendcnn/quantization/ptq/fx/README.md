Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch BlendCNN tuning(with MRPC dataset) results with Intel® Neural Compressor.

> **Note**
>
> PyTorch quantization implementation in imperative path has limitation on automatically execution.
> It requires to manually add QuantStub and DequantStub for quantizable ops, it also requires to manually do fusion operation.
> Intel® Neural Compressor has no capability to solve this framework limitation. Intel® Neural Compressor supposes user have done these two steps before invoking Intel® Neural Compressor interface.
> For details, please refer to https://pytorch.org/docs/stable/quantization.html

# Prerequisite

## 1. Installation

```Shell
cd examples/pytorch/nlp/blendcnn/quantization/ptq/fx
pip install -r requirements.txt
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

./run_benchmark.sh --int8=true --mode=benchmark --batch_size=32 --input_model=/DIR/TO/INT8_MODEL/
./run_benchmark.sh --int8=False --mode=benchmark --batch_size=32 --input_model=/PATH/TO/FP32_MODEL

```

## 3. Distillation of BlendCNN with BERT-Base as Teacher

### 3.1 Fine-tune the pretrained BERT-Base model on MRPC dataset

After preparation of step 2, you can fine-tune the pretrained BERT-Base model on MRPC dataset with below steps.
```Shell
mkdir -p models/bert/mrpc
# fine-tune the pretrained BERT-Base model
python finetune.py config/finetune/mrpc/train.json
```
When finished, you can find the fine-tuned BERT-Base model weights model_final.pt at `./models/bert/mrpc/`.

### 3.2 Distilling the BlendCNN with BERT-Base

```Shell
mkdir -p models/blendcnn/
# distilling the BlendCNN
python distill.py --loss_weights 0.1 0.9
```
Follow the above steps, you will find distilled BlendCNN model weights best_model_weights.pt in `./models/blendcnn/`.
