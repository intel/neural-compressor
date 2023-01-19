#!/bin/bash

# convert all mlperf inference models to onnx using
# tf2onnx (https://github.com/onnx/tensorflow-onnx).
# We assume tf2onnx is already installed (pip install -U tfonnx)
#
# by default we use opset 8 but if your runtime supports it opset 10 is a better choice.

export CUDA_VISIBLE_DEVICES=""

if [ $# -lt 1 ]; then
    echo "Usage: convert.sh model_dir ..."
    exit 1
fi

model_dir=$1
shift

opts="$@"

#
# resnet50
#
python -m tf2onnx.convert --input $model_dir/resnet50_v1.pb --output $model_dir/resnet50_v1.onnx \
    --inputs-as-nchw input_tensor:0 \
    --inputs input_tensor:0 \
    --outputs ArgMax:0,softmax_tensor:0 --opset 8 $opts

#
# mobilenet
#
python -m tf2onnx.convert --input $model_dir/mobilenet_v1_1.0_224_frozen.pb --output  $model_dir/mobilenet_v1_1.0_224.onnx \
    --inputs-as-nchw input:0 \
    --inputs input:0 \
    --outputs MobilenetV1/Predictions/Reshape_1:0 --opset 8 $opts

#
# ssd_mobilenet_v1_coco
#
python -m tf2onnx.convert --input  $model_dir/ssd_mobilenet_v1_coco_2018_01_28.pb --output  $model_dir/ssd_mobilenet_v1_coco_2018_01_28.onnx \
    --inputs image_tensor:0 \
    --outputs num_detections:0,detection_boxes:0,detection_scores:0,detection_classes:0 \
    --opset 10 $opts

#
# ssd_resnet34 (from tensorflow)
#
python -m tf2onnx.convert --input  $model_dir/resnet34_tf.22.1.pb --output $model_dir/ssd_resnet34_mAP_20.2.onnx \
    --inputs image:0 --outputs detection_bboxes:0,detection_scores:0,detection_classes:0  \
    --opset 10 $opts
