# Evaluate performance of ONNX Runtime(MNIST) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

### Environment
onnx: 1.9.0
onnxruntime: 1.8.0

### Prepare model
Download model from [ONNX Model Zoo](https://github.com/onnx/models)

```shell
wget https://github.com/onnx/models/raw/main/vision/classification/mnist/model/mnist-12.onnx
```

### Quantization
To quantize the model, run `main.py` with the path to the model:

```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=mnist.yaml \
                   --output_model=path/to/save
```

### Performance 
```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=mnist.yaml \
                   --output_model=path/to/save
```

