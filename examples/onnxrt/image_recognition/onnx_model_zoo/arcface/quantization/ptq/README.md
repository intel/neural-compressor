# Evaluate performance of ONNX Runtime(ArcFace) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load a face recognition model from [ONNX Model Zoo](https://github.com/onnx/models) and confirm its accuracy and speed based on [Refined MS-Celeb-1M](https://s3.amazonaws.com/onnx-model-zoo/arcface/dataset/faces_ms1m_112x112.zip). You need to download this dataset yourself.

### Environment
onnx: 1.11.0
onnxruntime: 1.10.0

### Prepare model
Download model from [ONNX Model Zoo](https://github.com/onnx/models)

```shell
wget https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-11.onnx
```

### Quantization

```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=arcface.yaml \
                   --data_path=/path/to/faces_ms1m_112x112/task.bin \
                   --nfolds=nfolds_num \ # optional, data type is int, default is 1
                   --output_model=path/to/save
```

### Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model \  # model path as *.onnx
                      --config=arcface.yaml \
                      --data_path=/path/to/faces_ms1m_112x112/task.bin \
                      --nfolds=nfolds_num \ # optional, data type is int, default is 1
                      --mode=performance # or accuracy
```

