Step-by-Step
============

This example load a gpt-j-6B model and confirm its accuracy and speed based on [lambada](https://huggingface.co/datasets/lambada).

# Prerequisite

## 1. Environment
```shell
pip install neural-compressor
pip install -r requirements.txt
```
> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model

```bash
python -m transformers.onnx --model=EleutherAI/gpt-j-6B model/ --framework pt --opset 13 --feature=causal-lm-with-past
```

# Run

## 1. Quantization

Static quantization:

```bash
bash run_quant.sh --input_model=/path/to/model \ # model path as *.onnx
                   --output_model=/path/to/model_tune \
                   --batch_size=batch_size # optional \
                   --quant_format="QOperator" # or QDQ, optional
```

## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model \ # model path as *.onnx
                      --batch_size=batch_size # optional \
                      --mode=performance # or accuracy
```
