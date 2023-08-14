# Step-by-Step

This example load a language translation model and confirm its accuracy and speed based on [SQuAD](<(https://rajpurkar.github.io/SQuAD-explorer/)>) task.

# Prerequisite

## 1. Environment

```shell
pip install neural-compressor
pip install -r requirements.txt
```

> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model

Download pretrained bert model. We will refer to `vocab.txt` file.t

```bash
python prepare_model.py  --input_model="bert" --output_model="bert.zip"
```

Download BERT-Squad from [onnx model zoo](https://github.com/onnx/models/tree/master/text/machine_comprehension/bert-squad).

```bash
python prepare_model.py  --input_model="BERT-Squad" --output_model="bertsquad-12.onnx"
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

Dynamic quantization:

```bash
bash run_quant.sh --input_model=/path/to/model \ # model path as *.onnx
                   --output_model=/path/to/model_tune \
                   --dataset_location=/path/to/squad
```

## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=/path/to/model \ # model path as *.onnx
                      --dataset_location=/path/to/squad \
                      --batch_size=batch_size \
                      --mode=performance # or accuracy
```
