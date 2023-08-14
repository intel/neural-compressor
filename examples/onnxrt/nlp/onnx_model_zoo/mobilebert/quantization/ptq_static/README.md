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

Download pretrained bert model. We will refer to `vocab.txt` file.

```bash
python prepare_model.py  --input_model="bert" --output_model="."
```

Download MLPerf mobilebert model and convert it to onnx model with [tf2onnx](https://github.com/onnx/tensorflow-onnx) tool.

```bash
python prepare_model.py  --input_model="mobilebert" --output_model="mobilebert_SQuAD.onnx"
```

## 3. Prepare Dataset
Download SQuAD dataset from [SQuAD dataset link](https://rajpurkar.github.io/SQuAD-explorer/).

Dataset directories:

```bash
squad
├── dev-v1.1.json
└── train-v1.1.json
```

# Run

## 1. Quantization

Static quantization with QDQ format:

```bash
bash run_quant.sh --input_model=/path/to/model \ # model path as *.onnx
                   --output_model=/path/to/model_tune \
                   --dataset_location=/path/to/squad \
                   --quant_format='QDQ'
```

## 2. Benchmark

```bash
bash run_quant.sh --input_model=/path/to/model \ # model path as *.onnx
                   --dataset_location=/path/to/squad \
                   --batch_size=batch_size \
                   --mode=performance # or accuracy
```
