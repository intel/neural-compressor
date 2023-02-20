Step-by-Step
============

This example load a language translation model and confirm its accuracy and speed based on [WikiText](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) dataset.

# Prerequisite

## 1. Environment
onnx: 1.12.0  
onnxruntime: 1.13.1
> Validated framework versions can be found in main readme.

## 2. Prepare Model

Use `export.py` script for ONNX model conversion.
Require torch==1.10.2 and transformers==3.2.0.

```shell
python export.py --model_name_or_path=Intel/gpt2-wikitext2 # or Intel/distilgpt2-wikitext2
```

## 3. Prepare Dataset
Please download [WikiText-2 dataset](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip).

# Run

## 1. Quantization

Quantize model with dynamic quantization:

```bash
bash run_tuning.sh --dataset_location=/path/to/wikitext-2-raw/wiki.valid.raw \ 
                   --input_model=path/to/model \ # model path as *.onnx
                   --output_model=path/to/model_tune
```

## 2. Benchmark

```bash
bash run_benchmark.sh --dataset_location=/path/to/wikitext-2-raw/wiki.valid.raw \ 
                      --input_model=path/to/model \ # model path as *.onnx
                      --batch_size=batch_size \
                      --mode=performance # or accuracy
```
