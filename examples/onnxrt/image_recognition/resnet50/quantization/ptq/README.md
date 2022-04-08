# Evaluate performance of ONNX Runtime(ResNet 50) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load an image classification model exported from PyTorch and confirm its accuracy and speed based on [ILSVR2012 validation Imagenet dataset](http://www.image-net.org/challenges/LSVRC/2012/downloads). You need to download this dataset yourself.

### Environment
onnx: 1.9.0
onnxruntime: 1.10.0

### Prepare model

#### ResNet 50 from torchvision
Please refer to [pytorch official guide](https://pytorch.org/docs/stable/onnx.html) for detailed model export. The following is a simple example:

```python
import torch
import torchvision
batch_size = 1
model = torchvision.models.resnet50(pretrained=True)
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
torch_out = model(x)

# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "resnet50.onnx",           # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to, please ensure at least 11.
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
```

#### ResNet 50 from MLPerf
Please refer to [MLPerf Inference Benchmarks for Image Classification and Object Detection Tasks](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection#mlperf-inference-benchmarks-for-image-classification-and-object-detection-tasks) for model details. Use [tf2onnx tool](https://github.com/onnx/tensorflow-onnx) to convert tensorflow model to onnx model.

```bash
wget https://zenodo.org/record/2535873/files/resnet50_v1.pb

python -m tf2onnx.convert --input resnet50_v1.pb --output resnet50_v1.onnx --inputs-as-nchw input_tensor:0 --inputs input_tensor:0 --outputs softmax_tensor:0 --opset 11
```

### Quantization

Quantize model with QLinearOps:

```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=resnet50_v1_5.yaml \  # or resnet50_v1_5_mlperf.yaml for ResNet50 from MLPerf
                   --output_model=path/to/save
```

Quantize model with QDQ mode:

```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=resnet50_v1_5_qdq.yaml \  # or resnet50_v1_5_mlperf_qdq.yaml for ResNet50 from MLPerf
                   --output_model=path/to/save
```

### Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model \  # model path as *.onnx
                      --config=resnet50_v1_5.yaml \  # or resnet50_v1_5_mlperf.yaml for ResNet50 from MLPerf
                      --mode=performance # or accuracy
```
