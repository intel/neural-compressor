# Evaluate performance of ONNX Runtime(Huggingface Text Classification) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load a language translation model and confirm its accuracy and speed based on [GLUE data](https://gluebenchmark.com/). 

### Environment
Please use latest onnx and onnxruntime version.

### Prepare dataset
download the GLUE data with `prepare_data.sh` script.

```shell
export GLUE_DIR=/path/to/glue_data
export TASK_NAME=MRPC # or SST

bash prepare_data.sh --data_dir=$GLUE_DIR --task_name=$TASK_NAME
```

### Prepare model

Supported model identifier from [huggingface.co](https://huggingface.co/):

|                 Model Identifier                |
|:-----------------------------------------------:|
|           Intel/bert-base-uncased-mrpc          |
|             Intel/roberta-base-mrpc             |
|           Intel/xlm-roberta-base-mrpc           |
|            Intel/camembert-base-mrpc            |
| distilbert-base-uncased-finetuned-sst-2-english |
|         Alireza1044/albert-base-v2-sst2         |
|        Intel/MiniLM-L12-H384-uncased-mrpc       |
|      philschmid/MiniLM-L6-H384-uncased-sst2     |

```bash
python export.py --model_name_or_path=Intel/bert-base-uncased-mrpc \ # or other supported model identifier
```

### Quantization

Quantize model with dynamic quantization:

```bash
bash run_tuning.sh --config=glue_dynamic.yaml \ 
                   --input_model=path/to/model \ # model path as *.onnx
                   --output_model=path/to/model_tune \ # model path as *.onnx
                   --data_path=path/to/glue/data
```

### Benchmark

```bash
bash run_benchmark.sh --config=glue_dynamic.yaml \ 
                   --input_model=path/to/model \ # model path as *.onnx
                   --data_path=path/to/glue/data \ 
                   --mode=performance # or accuracy
```
