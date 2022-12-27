# Evaluate performance of ONNX Runtime(GPT2) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load a language translation model and confirm its accuracy and speed based on [WikiText](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) dataset. 

### Environment
onnx: 1.7.0
onnxruntime: 1.8.0
transformers: 3.2.0

### Prepare dataset
Please download [WikiText-2 dataset](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip).

### Prepare model
Use `export.py` script for ONNX model conversion. 

```shell
python export.py
```

### Quantization

```bash
bash run_tuning.sh --dataset_location=/path/to/wikitext-2-raw/ \ # NOTE: path must end with /
                   --input_model=path/to/model \ # model path as *.onnx
                   --output_model=path/to/model_tune
```

### Benchmark
```bash
bash run_benchmark.sh --dataset_location=/path/to/wikitext-2-raw/ \ # NOTE: path must end with /
                      --input_model=path/to/model \ # model path as *.onnx
                      --batch_size=batch_size \
                      --mode=performance # or accuracy
```
