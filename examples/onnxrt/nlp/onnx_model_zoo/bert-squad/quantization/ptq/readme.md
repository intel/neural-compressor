# Evaluate performance of ONNX Runtime(BERT) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load a language translation model and confirm its accuracy and speed based on [SQuAD]((https://rajpurkar.github.io/SQuAD-explorer/)) task. 

### Environment
onnx: 1.9.0
onnxruntime: 1.10.0

### Prepare dataset
You should download SQuAD dataset from [SQuAD dataset link](https://rajpurkar.github.io/SQuAD-explorer/).

### Prepare model
Download pretrained bert model. We will refer to `vocab.txt` file.

```bash
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
```

Download BERT-Squad from [onnx model zoo](https://github.com/onnx/models/tree/master/text/machine_comprehension/bert-squad).
```bash
wget https://github.com/onnx/models/raw/main/text/machine_comprehension/bert-squad/model/bertsquad-12.onnx
```

### Quantization

Dynamic quantize:

```bash
bash run_tuning.sh --input_model=/path/to/model \ # model path as *.onnx
                   --output_model=/path/to/model_tune \
                   --data_path=/path/to/SQuAD/dataset \
                   --config=bert.yaml
```

QDQ mode:

```bash
bash run_tuning.sh --input_model=/path/to/model \ # model path as *.onnx
                   --output_model=/path/to/model_tune \
                   --data_path=/path/to/SQuAD/dataset \
                   --config=bert_qdq.yaml
```

### Benchmark

```bash
bash run_benchmark.sh --input_model=/path/to/model \ # model path as *.onnx
                      --data_path=/path/to/SQuAD/dataset \
                      --config=bert.yaml
                      --mode=performance # or accuracy
```

