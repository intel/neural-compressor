# Evaluate performance of ONNX Runtime(SSD Mobilenet v1) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load an object detection model converted from Tensorflow and confirm its accuracy and speed based on [MS COCO 2017 dataset](https://cocodataset.org/#download). You need to download this dataset yourself.

### Environment
onnx: 1.7.0
onnxruntime: 1.6.0+

### Prepare model
Please refer to [Converting SSDMobilenet To ONNX Tutorial](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/ConvertingSSDMobilenetToONNX.ipynb) for detailed model converted. 

### Evaluating
To evaluate the model, run `main.py` with the path to the model:

```bash
bash run_tuning.sh --input_model=path/to/model  \ # model path as *.onnx
                   --config=ssd_mobilenet_v1.yaml \ 
                   --output_model=path/to/save
```


