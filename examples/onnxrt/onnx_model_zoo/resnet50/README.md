# Evaluate performance of ONNX Runtime(ResNet 50) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load an image classification model from [ONNX Model Zoo](https://github.com/onnx/models) and confirm its accuracy and speed based on [ILSVR2012 validation Imagenet dataset](http://www.image-net.org/challenges/LSVRC/2012/downloads). You need to download this dataset yourself.

### Environment
onnx: 1.7.0
onnxruntime: 1.6.0+

### Prepare model
Download model from [ONNX Model Zoo](https://github.com/onnx/models)

```shell
wget https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v1-12.onnx
```

### Quantization

```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=resnet50_v1_5.yaml \
                   --output_model=path/to/save
```

### Performance 
Usually we need to bind the program to specific cores like 4 cores to get performance under real production environments.   
**for linux**
```bash
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=4
numactl --physcpubind=0-3 --membind=0 python main.py --model_path path/to/model --config resnet50_v1_5.yaml --benchmark
```

