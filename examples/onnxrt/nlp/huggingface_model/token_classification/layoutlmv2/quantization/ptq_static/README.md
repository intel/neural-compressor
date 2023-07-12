Step-by-Step
============

This example load [LayoutLMv2](https://huggingface.co/microsoft/layoutlmv2-base-uncased) model and confirm its accuracy and speed based on [FUNSD](https://huggingface.co/datasets/nielsr/funsd) dataset.

# Prerequisite

## 1. Environment
```shell
pip install neural-compressor
pip install -r requirements.txt
```
> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare ONNX Model
Export the PyTorch model to ONNX.

```bash
# TODO replace it with custom export as the optimum not support export layoutlmv2
# fine-tuned model https://huggingface.co/nielsr/layoutlmv2-finetuned-funsd
 python export.py --torch_model_name_or_path=/fine-tuned/torch/model/name/or/path
```

# Run

## 1. Quantization

Static quantization with QOperator format:

```bash
bash run_tuning.sh --input_model=./layoutlmv2-finetuned-funsd-exported.onnx \ # onnx model path as *.onnx
                   --output_model=/path/to/model_tune \
                   --quant_format="QOperator"
```


## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=/path/to/model \ # model path as *.onnx
                      --batch_size=batch_size \
                      --mode=performance # or accuracy
```
