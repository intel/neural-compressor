# Evaluate performance of ONNX Runtime(Ultra-lightweight face detection) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load an model converted from [ONNX Model Zoo](https://github.com/onnx/models) and confirm its accuracy and speed based on [WIDER FACE dataset (Validation Images)](http://shuoyang1213.me/WIDERFACE/). You need to download this dataset yourself.

### Environment
onnx: 1.11.0
onnxruntime: 1.10.0

### Prepare model
Download model from [ONNX Model Zoo](https://github.com/onnx/models)

```shell

wget https://github.com/onnx/models/raw/main/vision/body_analysis/ultraface/models/version-RFB-320-12.onnx
```

### Quantization

```bash
bash run_tuning.sh --input_model=path/to/model  \ # model path as *.onnx
                   --config=ultraface.yaml \ 
                   --data_path=/path/to/data \
                   --output_model=path/to/save
```

### Performance

```bash
bash run_benchmark.sh --input_model=path/to/model \  # model path as *.onnx
                      --config=ultraface.yaml \
                      --data_path=/path/to/data \
                      --mode=performance
```
