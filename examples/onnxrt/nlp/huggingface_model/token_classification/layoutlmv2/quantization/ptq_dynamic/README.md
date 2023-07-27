Step-by-Step
============

This example quantizes the [LayoutLMv2](https://huggingface.co/microsoft/layoutlmv2-base-uncased) model that is fine-tuned on the [FUNSD](https://huggingface.co/datasets/nielsr/funsd) dataset.

# Prerequisite

## 1. Environment
```shell
pip install neural-compressor
pip install -r requirements.txt
```
> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare ONNX Model
Export the [nielsr/layoutlmv2-finetuned-funsd](https://huggingface.co/nielsr/layoutlmv2-finetuned-funsd) model to ONNX.

```bash
# fine-tuned model https://huggingface.co/nielsr/layoutlmv2-finetuned-funsd
 python export.py --torch_model_name_or_path=/fine-tuned/torch/model/name/or/path
```
> Note: To export LayoutLMv2, please install [detectron2](https://github.com/facebookresearch/detectron2) first.

# Run

## 1. Quantization

Static quantization with QOperator format:

```bash
bash run_quant.sh --input_model=./layoutlmv2-finetuned-funsd-exported.onnx \ # onnx model path as *.onnx
                   --output_model=/path/to/model_tune \
                   --quant_format="QOperator"
```


## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=/path/to/model \ # model path as *.onnx
                      --batch_size=batch_size \
                      --mode=performance # or accuracy
```
