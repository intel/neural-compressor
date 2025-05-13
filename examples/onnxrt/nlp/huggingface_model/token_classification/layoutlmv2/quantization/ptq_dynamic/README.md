# Step-by-Step (Deprecated)

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
 python prepare_model.py  --input_model="nielsr/layoutlmv2-finetuned-funsd" --output_model="layoutlmv2-finetuned-funsd-exported.onnx"
```

> Note: To export LayoutLMv2, please install [detectron2](https://github.com/facebookresearch/detectron2) with `python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'` first. Also, you should probably install tesseract with something like: `conda install -c conda-forge tesseract`.

# Run

## 1. Quantization

Dynamic quantization:

```bash
bash run_quant.sh --input_model=./layoutlmv2-finetuned-funsd-exported.onnx \ # onnx model path as *.onnx
                   --output_model=/path/to/model_tune 
```

## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=/path/to/model \ # model path as *.onnx
                      --batch_size=batch_size \
                      --mode=performance # or accuracy
```
