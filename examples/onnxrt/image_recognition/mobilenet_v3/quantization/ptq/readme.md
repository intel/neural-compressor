# Evaluate performance of ONNX Runtime(Mobilenet v3) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load an image classification model exported from PyTorch and confirm its accuracy and speed based on [ILSVR2012 validation Imagenet dataset](http://www.image-net.org/challenges/LSVRC/2012/downloads). You need to download this dataset yourself.

### Environment
onnx: 1.7.0
onnxruntime: 1.6.0+

### Prepare model
Use [tf2onnx tool](https://github.com/onnx/tensorflow-onnx) to convert tflite to onnx model.

```bash
wget https://github.com/mlcommons/mobile_models/blob/main/v0_7/tflite/mobilenet_edgetpu_224_1.0_float.tflite

python -m tf2onnx.convert --opset 11 --tflite mobilenet_edgetpu_224_1.0_float.tflite --output mobilenet_v3.onnx
```

### Evaluating
To evaluate the model, run `main.py` with the path to the model:

```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=mobilenet_v3.yaml \ 
                   --output_model=path/to/save
```

### Advanced 
Usually we need to bind the program to specific cores like 4 cores to get performance under real production environments.   
**for linux**
```bash
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=4
numactl --physcpubind=0-3 --membind=0 python main.py --model_path path/to/model --benchmark \ 
--tune  --config mobilenet_v3.yaml 
```

