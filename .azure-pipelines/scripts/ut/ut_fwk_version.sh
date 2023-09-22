#!/bin/bash

echo "export UT fwk version..."
test_mode=$1

if [ "$test_mode" == "coverage" ]; then
    export tensorflow_version='2.13.0'
    export pytorch_version='2.0.0+cpu'
    export torchvision_version='0.15.2+cpu'
    export ipex_version='2.0.1+cpu'
    export onnx_version='1.14.1'
    export onnxruntime_version='1.15.1'
    export mxnet_version='1.9.1'
else
    export tensorflow_version='2.11.0'
    export pytorch_version='1.13.0+cpu'
    export torchvision_version='0.14.0+cpu'
    export ipex_version='1.13.0+cpu'
    export onnx_version='1.13.1'
    export onnxruntime_version='1.14.1'
    export mxnet_version='1.9.1'
fi





