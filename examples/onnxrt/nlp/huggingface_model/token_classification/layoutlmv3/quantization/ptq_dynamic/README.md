Step-by-Step
============

This example load LayoutLMv3 model and confirm its accuracy and speed based on [FUNSD](https://huggingface.co/datasets/nielsr/funsd) dataset.

# Prerequisite

## 1. Environment
```shell
pip install neural-compressor
pip install -r requirements.txt
```
> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model
Export a model to ONNX with `optimum.exporters.onnx`.

```bash
python prepare_model.py  --input_model="HYPJUDY/layoutlmv3-base-finetuned-funsd" --output_model="layoutlmv3-base-finetuned-funsd-onnx/"
```

# Run

## 1. Quantization

Dynamic quantization:

```bash
bash run_quant.sh --input_model=./layoutlmv3-base-finetuned-funsd-onnx/model.onnx \ # model path as *.onnx
                   --output_model=/path/to/model_tune
```


## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=/path/to/model \ # model path as *.onnx
                      --batch_size=batch_size \
                      --mode=performance # or accuracy
```
