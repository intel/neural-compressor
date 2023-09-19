Step-by-Step
============

This example load a language translation model and confirm its accuracy and speed based on [WikiText](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) dataset.

# Prerequisite

## 1. Environment
```shell
pip install neural-compressor
pip install -r requirements.txt
```
> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model

Use `prepare_model.py` script for ONNX model conversion.
Require transformers==3.2.0.

```shell
python prepare_model.py
```

## 3. Prepare Dataset
Please download [WikiText-2 dataset](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip).

Dataset directories:

```bash
wikitext-2-raw
├── wiki.test.raw
├── wiki.train.raw
└── wiki.valid.raw
```

# Run

## 1. Quantization

Dynamic quantization:

```bash
bash run_quant.sh --dataset_location=/path/to/wikitext-2-raw/wiki.test.raw \ 
                   --input_model=path/to/model \ # model path as *.onnx
                   --output_model=path/to/model_tune # model path as *.onnx
```

## 2. Benchmark

```bash
bash run_benchmark.sh --dataset_location=/path/to/wikitext-2-raw/wiki.test.raw \ 
                      --input_model=path/to/model \ # model path as *.onnx
                      --batch_size=batch_size \
                      --mode=performance # or accuracy
```
