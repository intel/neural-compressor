# Evaluate performance of ONNX Runtime(SSD Mobilenet v2) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load an object detection model converted from Tensorflow and confirm its accuracy and speed based on [MS COCO 2017 dataset](https://cocodataset.org/#download). You need to download this dataset yourself.

### Environment
onnx: 1.9.0
onnxruntime: 1.10.0

### Prepare model
Please refer to [Converting SSDMobilenet To ONNX Tutorial](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/ConvertingSSDMobilenetToONNX.ipynb) for detailed model converted. The following is a simple example command:

```shell
export MODEL=ssd_mobilenet_v2_coco_2018_03_29
wget http://download.tensorflow.org/models/object_detection/$MODEL.tar.gz
tar -xvf $MODEL.tar.gz

python -m tf2onnx.convert --graphdef $MODEL/frozen_inference_graph.pb --output ./$MODEL.onnx --fold_const --opset 11 --inputs image_tensor:0 --outputs num_detections:0,detection_boxes:0,detection_scores:0,detection_classes:0
```

### Quantization

Quantize model with QLinearOps:

```bash
bash run_tuning.sh --input_model path/to/model  \ # model path as *.onnx
                   --config ssd_mobilenet_v2.yaml \ 
                   --output_model path/to/save
```

Quantize model with QDQ mode:

```bash
bash run_tuning.sh --input_model path/to/model  \ # model path as *.onnx
                   --config ssd_mobilenet_v2_qdq.yaml \ 
                   --output_model path/to/save
```

### Benchmark

```bash
bash run_benchmark.sh --input_model path/to/model  \ # model path as *.onnx
                      --config ssd_mobilenet_v2.yaml \
                      --mode=performance # or accuracy
```
