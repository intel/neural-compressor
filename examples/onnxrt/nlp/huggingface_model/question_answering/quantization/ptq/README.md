# Evaluate performance of ONNX Runtime(Huggingface Question Answering) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load a language translation model and confirm its accuracy and speed based on [SQuAD]((https://rajpurkar.github.io/SQuAD-explorer/)) task. 

### Environment
Please use latest onnx and onnxruntime version.

### Prepare dataset
You should download SQuAD dataset from [SQuAD dataset link](https://rajpurkar.github.io/SQuAD-explorer/).

### Prepare model

Supported model identifier from [huggingface.co](https://huggingface.co/):

|                 Model Identifier                |
|:-----------------------------------------------:|
|           mrm8488/spanbert-finetuned-squadv1          |
|             salti/bert-base-multilingual-cased-finetuned-squad             |


```bash
python export.py --model_name_or_path=mrm8488/spanbert-finetuned-squadv1 \ # or other supported model identifier
```

### Quantization

Dynamic quantize:

```bash
bash run_tuning.sh --input_model=/path/to/model \ # model path as *.onnx
                   --output_model=/path/to/model_tune \
                   --config=qa_dynamic.yaml
```

### Benchmark

```bash
bash run_benchmark.sh --input_model=/path/to/model \ # model path as *.onnx
                      --config=qa_dynamic.yaml
                      --mode=performance # or accuracy
```

