Step-by-Step
============

This example quantizes the [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base) fine-tuned on the the [code defect detection](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection#codexglue----defect-detection) task.

# Prerequisite

## 1. Environment
```shell
pip install neural-compressor
pip install -r requirements.txt
```
> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).


## 2. Prepare Dataset
Run `prepare_data.sh` script to download dataset from website to `dataset` folder and pre-process it:

```shell
bash prepare_data.sh
```
## 3. Prepare Model

Fine-tuning the model on [code defect detection](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection#codexglue----defect-detection) task.
```
bash run_fine_tuning.sh --train_dataset_location=./dataset/train.jsonl --dataset_location=./dataset/valid.jsonl  --fine_tune
```

Export model to ONNX format. 
```bash
# TODO replace the model name after uploading the model to the hugging face
optimum-cli export onnx --model Intel/TBD-MODEL-NAME --task text-classification onnx_model/
```

# Run

## 1. Quantization

Static quantization with QOperator format:

```bash
bash run_quant.sh --input_model=/path/to/model \ # model path as *.onnx
                   --output_model=/path/to/model_tune \
                   --dataset_location=path/to/glue/data \
                   --quant_format="QOperator"
```

## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model \ # model path as *.onnx
                      --dataset_location=path/to/glue/data \ 
                      --batch_size=batch_size \ 
                      --mode=performance # or accuracy
```
