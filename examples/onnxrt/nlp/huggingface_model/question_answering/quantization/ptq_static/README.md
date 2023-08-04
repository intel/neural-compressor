Step-by-Step
============

This example load a language translation model and confirm its accuracy and speed based on [SQuAD]((https://rajpurkar.github.io/SQuAD-explorer/)) task.

# Prerequisite

## 1. Environment
```shell
pip install neural-compressor
pip install -r requirements.txt
```
> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model
Supported model identifier from [huggingface.co](https://huggingface.co/):

|                 Model Identifier                 |
|:------------------------------------------------:|
|           mrm8488/spanbert-finetuned-squadv1     |
|salti/bert-base-multilingual-cased-finetuned-squad|
|     distilbert-base-uncased-distilled-squad      |
|bert-large-uncased-whole-word-masking-finetuned-squad|
|           deepset/roberta-large-squad2           | 


```bash
python export.py --model_name_or_path=mrm8488/spanbert-finetuned-squadv1 \ # or other supported model identifier
```

## 3. Prepare Dataset
Download SQuAD dataset from [SQuAD dataset link](https://rajpurkar.github.io/SQuAD-explorer/).

# Run

## 1. Quantization

Static quantization with QOperator format:

```bash
bash run_quant.sh --input_model=/path/to/model \ # model path as *.onnx
                   --output_model=/path/to/model_tune \
                   --quant_format="QOperator"
```


## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=/path/to/model \ # model path as *.onnx
                      --batch_size=batch_size \
                      --mode=performance # or accuracy
```
