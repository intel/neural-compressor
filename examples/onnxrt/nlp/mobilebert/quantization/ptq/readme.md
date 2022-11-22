# Evaluate performance of ONNX Runtime(MobileBERT) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load a language translation model and confirm its accuracy and speed based on [GLUE data](https://gluebenchmark.com/). 

### Environment
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
Please refer to [Bert-GLUE_OnnxRuntime_quantization guide](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/notebooks/Bert-GLUE_OnnxRuntime_quantization.ipynb) for detailed model export. The following is a simple example.

Use [Huggingface Transformers](https://github.com/huggingface/transformers/tree/v2.2.1) to fine-tune the model based on the [MRPC](https://github.com/huggingface/transformers/tree/master/examples/text-classification#mrpc) example with command like:
```shell
export OUT_DIR=/path/to/out_dir/
python ./run_glue.py \ 
    --model_type mobilebert \
    --model_name_or_path google/mobilebert-uncased \ 
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8  \
    --per_gpu_train_batch_size=8  \
    --learning_rate 2e-5 \
    --num_train_epochs 5.0 \
    --save_steps 100000 \
    --output_dir $OUT_DIR
```
Run the `prepare_model.sh` script.

Usage:
```shell
cd examples/onnxrt/language_translation/mobilebert/

bash prepare_model.sh --input_dir=$OUT_DIR \
                      --task_name=$TASK_NAME \
                      --output_model=path/to/model # model path as *.onnx
```

### Benchmark

Dynamic quantize:

```bash
bash run_tuning.sh --config=mobilebert.yaml \ 
                   --input_model=path/to/model \ # model path as *.onnx
                   --output_model=path/to/model_tune
```

QDQ mode:

```bash
bash run_tuning.sh --config=mobilebert_qdq.yaml \ 
                   --input_model=path/to/model \ # model path as *.onnx
                   --output_model=path/to/model_tune
```

### Benchmark

```bash
bash run_benchmark.sh --config=mobilebert.yaml \ 
                      --input_model=path/to/model \ # model path as *.onnx
                      --mode=performance # or accuracy
```



