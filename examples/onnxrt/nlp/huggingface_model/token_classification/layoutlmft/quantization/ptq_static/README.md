Step-by-Step
============

This example load LayoutLMv3 model and confirm its accuracy and speed based on [FUNSD](https://huggingface.co/datasets/nielsr/funsd) dataset.

# Prerequisite

## 1. Environment
```shell
pip install neural-compressor
pip install -r requirements.txt
bash install_layoutlmft.sh
```
> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model
Finetune on FUNSD

```bash
python main.py \
       --model_name_or_path microsoft/layoutlm-base-uncased \
       --output_dir ./layoutlm-base-uncased-fintuned-funsd \
       --do_train \
       --max_steps 1000 \
       --warmup_ratio 0.1 
```

Export a model to ONNX with `optimum.exporters.onnx`.

```bash
optimum-cli export onnx --model ./layoutlm-base-uncased-fintuned-funsd ./layoutlm-base-uncased-fintuned-funsd-onnx/ --task=token-classification
```

# Run

## 1. Quantization

Static quantization with QOperator format:

```bash
bash run_tuning.sh --input_model=./layoutlm-base-uncased-fintuned-funsd-onnx/model.onnx \ # model path as *.onnx
                   --output_model=/path/to/model_tune \
                   --quant_format="QOperator"
```


## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=/path/to/model \ # model path as *.onnx
                      --batch_size=batch_size \
                      --mode=performance # or accuracy
```
