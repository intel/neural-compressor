Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch BlendCNN distillation(with MRPC dataset) results with Intel® Neural Compressor.

# Prerequisite

## 1. Environment

```Shell
cd examples/pytorch/nlp/blendcnn/distillation/eager
pip install torch>=1.6.0 tqdm
```

## 2. Prepare model
Download [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) 

```Shell
mkdir models/ && mv uncased_L-12_H-768_A-12.zip models/
cd models/ && unzip uncased_L-12_H-768_A-12.zip
```

## 3. Prepare Datasets
Download 
[GLUE MRPC Benchmark Datasets](https://github.com/nyu-mll/GLUE-baselines), 
After downloads dataset, you need to put dataset at `./MRPC/`, list this:

```Shell
ls MRPC/
dev_ids.tsv  dev.tsv  test.tsv  train.tsv
```

# Distillation
## 1 Fine-tune the pretrained BERT-Base model on MRPC dataset
```Shell
mkdir -p models/bert/mrpc
# fine-tune the pretrained BERT-Base model
python finetune.py config/finetune/mrpc/train.json
```

## 2 Distilling the BlendCNN with BERT-Base
Now BERT-Base model weights model_final.pt at `./models/bert/mrpc/`.

```Shell
mkdir -p models/blendcnn/
# distilling the BlendCNN
python distill.py --loss_weights 0.1 0.9
```
Follow the above steps, you will find distilled BlendCNN model weights best_model_weights.pt in `./models/blendcnn/`.
