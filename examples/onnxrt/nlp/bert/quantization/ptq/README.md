# Evaluate performance of ONNX Runtime(BERT) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load a language translation model and confirm its accuracy and speed based on [GLUE data](https://gluebenchmark.com/). 

### Environment

#### Dynamic quantization environment:

onnx: 1.7.0
onnxruntime: 1.6.0+

#### Static quantization environment:

onnx: 1.9.0
onnxruntime: 1.10.0

### Prepare dataset
download the GLUE data with `prepare_data.sh` script.
```shell
export GLUE_DIR=/path/to/glue_data
export TASK_NAME=MRPC

bash prepare_data.sh --data_dir=$GLUE_DIR --task_name=$TASK_NAME
```

### Prepare model
Please refer to [Bert-GLUE_OnnxRuntime_quantization guide](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/notebooks/Bert-GLUE_OnnxRuntime_quantization.ipynb) for detailed model export.

Run the `prepare_model.sh` script


Usage:
```shell
bash prepare_model.sh --input_dir=./MRPC \
                      --task_name=$TASK_NAME \
                      --output_model=path/to/model # model path as *.onnx
```

### Quantization

Quantize model with dynamic quantization:

```bash
bash run_tuning.sh --config=bert_dynamic.yaml \ 
                   --input_model=path/to/model \ # model path as *.onnx
                   --output_model=path/to/model_tune
```

Quantize model with QLinearOps:

```bash
bash run_tuning.sh --config=bert_static.yaml \ 
                   --input_model=path/to/model \ # model path as *.onnx
                   --output_model=path/to/model_tune
```

Quantize model with QDQ mode:

```bash
bash run_tuning.sh --config=bert_qdq.yaml \ 
                   --input_model=path/to/model \ # model path as *.onnx
                   --output_model=path/to/model_tune
```

### Benchmark

```bash
bash run_tuning.sh --config=bert_dynamic.yaml \ 
                   --input_model=path/to/model \ # model path as *.onnx
                   --mode=performance # or accuracy
```
