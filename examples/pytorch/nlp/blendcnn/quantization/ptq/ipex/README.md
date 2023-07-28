Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch BlendCNN with IPEX backend tuning (with MRPC dataset) results with Intel® Neural Compressor.

# Prerequisite

## 1. Installation

```Shell
cd examples/pytorch/nlp/blendcnn/quantization/ptq/ipex
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

After downloading the datasets, you need to put datasets at `./MRPC/`, including:

```Shell
ls MRPC/
dev_ids.tsv  dev.tsv  test.tsv  train.tsv
```

## Run
### Tuning
```Shell
bash run_quant.sh --input_model=/PATH/TO/models/ --dataset_location=/PATH/TO/MRPC/ --output_model=/DIR/TO/INT8_MODEL/
```
### Benchmark
```Shell
bash run_benchmark.sh --int8=true --mode=performance --batch_size=32 --input_model=/DIR/TO/INT8_MODEL/
bash run_benchmark.sh --int8=False --mode=performance --batch_size=32 --input_model=/PATH/TO/FP32_MODEL
```
