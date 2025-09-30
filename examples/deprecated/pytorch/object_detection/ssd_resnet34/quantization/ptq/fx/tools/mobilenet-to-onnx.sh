#!/bin/bash

# convert mobilenet to onnx

python -c "import tf2onnx" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "We use tensorflow-onnx to convert tensorflow to onnx."
    echo "See https://github.com/onnx/tensorflow-onnx for details."    
    echo "Install with:"
    echo "pip install tf2onnx"
    echo "or"
    echo "pip install https://github.com/onnx/tensorflow-onnx"
    exit 1
fi

model=resnet50_v1
tfmodel=mobilenet_v1_1.0_224_frozen.pb
onnxmodel=mobilenet_v1_1.0_224.onnx
url=http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz
tgz=$(basename $url)

if [ ! -r $tgz ]; then
    wget  -q  $url
    tar zxvf $tgz
fi
python -m tf2onnx.convert --input $tfmodel --output $onnxmodel \
    --fold_const --opset 8 --verbose \
    --inputs-as-nchw input:0 \
    --inputs input:0 \
    --outputs MobilenetV1/Predictions/Reshape_1:0
