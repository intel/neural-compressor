Step-by-Step
============

This example load a language translation model and confirm its accuracy and speed based on [SQuAD]((https://rajpurkar.github.io/SQuAD-explorer/)) task.

# Prerequisite

## 1. Environment
onnx: 1.12.0  
onnxruntime: 1.13.1
> Validated framework versions can be found in main readme.

## 2. Prepare Model
Supported model identifier from [huggingface.co](https://huggingface.co/):

|                 Model Identifier                |
|:-----------------------------------------------:|
|           mrm8488/spanbert-finetuned-squadv1          |
|             salti/bert-base-multilingual-cased-finetuned-squad             |


```bash
python export.py --model_name_or_path=mrm8488/spanbert-finetuned-squadv1 \ # or other supported model identifier
```

## 3. Prepare Dataset
Download SQuAD dataset from [SQuAD dataset link](https://rajpurkar.github.io/SQuAD-explorer/).

# Run

## 1. Quantization

Quantize model with dynamic quantization:

```bash
bash run_tuning.sh --input_model=/path/to/model \ # model path as *.onnx
                   --output_model=/path/to/model_tune 
```


## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=/path/to/model \ # model path as *.onnx
                      --batch_size=batch_size \
                      --mode=performance # or accuracy
```
