Step-by-Step (Deprecated)
============

This example load a RoBERTa model and confirm its accuracy and speed based on [GLUE data](https://gluebenchmark.com/). 

# Prerequisite

## 1. Environment

```shell
pip install neural-compressor
pip install -r requirements.txt
```
> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Dataset

download the GLUE data with `prepare_data.sh` script.
```shell
export GLUE_DIR=path/to/glue_data
export TASK_NAME=MRPC

bash prepare_data.sh --data_dir=$GLUE_DIR --task_name=$TASK_NAME
```

## 3. Prepare Model

Please refer to [Bert-GLUE_OnnxRuntime_quantization guide](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/notebooks/bert/Bert-GLUE_OnnxRuntime_quantization.ipynb) for detailed model export. The following is a simple example.

Use [Huggingface Transformers](https://github.com/huggingface/transformers/tree/v2.2.1) to fine-tune the model based on the [MRPC](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) example with command like:

```shell
python prepare_model.py --input_model='roberta-base' --output_model=roberta.onnx
```

# Run

## 1. Quantization

Static quantization with QDQ format:

```bash
bash run_quant.sh --input_model=path/to/model \ # model path as *.onnx
                   --output_model=path/to/model_tune \ # model path as *.onnx
                   --dataset_location=path/to/glue_data \
                   --quant_format="QDQ"
```

## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model \ # model path as *.onnx
                      --dataset_location=path/to/glue_data \
                      --batch_size=batch_size \
                      --mode=performance # or accuracy
```


