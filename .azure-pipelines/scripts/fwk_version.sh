#!/bin/bash

echo "export FWs version..."
test_mode=$1

if [ "$test_mode" == "coverage" ] || [ "$test_mode" == "latest" ]; then
    export tensorflow_version='2.15.0-official'
    export pytorch_version='2.2.1+cpu'
    export torchvision_version='0.17.1+cpu'
    export ipex_version='2.2.0+cpu'
    export onnx_version='1.15.0'
    export onnxruntime_version='1.17.1'
    export mxnet_version='1.9.1'
else
    export tensorflow_version='2.14.0'
    export pytorch_version='2.1.0+cpu'
    export torchvision_version='0.16.0+cpu'
    export ipex_version='2.1.0+cpu'
    export onnx_version='1.14.1'
    export onnxruntime_version='1.16.3'
    export mxnet_version='1.9.1'
fi





