#!/bin/bash
set -x

echo "copy pre-train model..."
mkdir -p /tmp/.neural_compressor/inc_ut || true
cp -r /tf_dataset/ut-localfile/resnet_v2 /tmp/.neural_compressor/inc_ut || true
mkdir -p ~/.keras/datasets || true
cp -r /tf_dataset/ut-localfile/cifar-10-batches-py* ~/.keras/datasets || true
ll ~/.keras/datasets

echo "install dependencies..."
echo "tensorflow version is $tensorflow_version"
echo "itex version is $itex_version"
echo "pytorch version is $pytorch_version"
echo "torchvision version is $torchvision_version"
echo "ipex version is $ipex_version"
echo "onnx version is $onnx_version"
echo "onnxruntime version is $onnxruntime_version"
echo "mxnet version is $mxnet_version"

test_case=$1
echo "========= test case is ${test_case}"

if [[ "${tensorflow_version}" == *"-official" ]]; then
    pip install tensorflow==${tensorflow_version%-official}
elif [[ "${tensorflow_version}" == "spr-base" ]]; then
    pip install /tf_dataset/tf_binary/221212/tensorflow*.whl
    pip install protobuf==3.20.3
    pip install horovod==0.27.0
    if [[ $? -ne 0 ]]; then
      exit 1
    fi
elif [[ "${tensorflow_version}" != "" ]]; then
    pip install intel-tensorflow==${tensorflow_version}
fi

if [[ "${itex_version}" != "" ]]; then
    pip install --upgrade intel-extension-for-tensorflow[cpu]==${itex_version}
    pip install tf2onnx
fi

if [[ "${pytorch_version}" != "" ]]; then
    pip install torch==${pytorch_version} -f https://download.pytorch.org/whl/torch_stable.html
fi

if [[ "${torchvision_version}" != "" ]]; then
    pip install torchvision==${torchvision_version} -f https://download.pytorch.org/whl/torch_stable.html
fi

if [[ "${ipex_version}" == "1.13.0+cpu" ]]; then
    ipex_whl="https://github.com/intel/intel-extension-for-pytorch/releases/download/v1.13.0%2Bcpu/intel_extension_for_pytorch-1.13.0-cp38-cp38-manylinux2014_x86_64.whl"
    pip install $ipex_whl
elif [[ "${ipex_version}" == "2.0.0+cpu" ]]; then
    ipex_whl="https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/cpu/intel_extension_for_pytorch-2.0.0%2Bcpu-cp38-cp38-linux_x86_64.whl"
    pip install $ipex_whl
elif [[ "${ipex_version}" == "2.0.1+cpu" ]]; then
    ipex_whl="https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/cpu/intel_extension_for_pytorch-2.0.100%2Bcpu-cp38-cp38-linux_x86_64.whl"
    pip install $ipex_whl
elif [[ "${ipex_version}" == "2.1.0" ]]; then
    pip install /tf_dataset/pt_binary/ww32/torch-*.whl
    pip install /tf_dataset/pt_binary/ww32/torchvision-*.whl
    pip install /tf_dataset/pt_binary/ww32/intel_extension_for_pytorch-*.whl
fi

if [[ "${onnx_version}" != "" ]]; then
    pip install onnx==${onnx_version}
fi

if [[ "${onnxruntime_version}" != "" ]]; then
    pip install onnxruntime==${onnxruntime_version}
    pip install onnxruntime-extensions==0.8.0
    pip install optimum
fi

if [ "${mxnet_version}" != '' ]; then
    pip install numpy==1.23.5
    echo "re-install pycocotools resolve the issue with numpy..."
    pip uninstall pycocotools -y
    pip install --no-cache-dir pycocotools
    pip install mxnet==${mxnet_version}
fi

# install special test env requirements
# common deps
pip install transformers
pip install horovod
if [[ $(echo "${test_case}" | grep -c "others") != 0 ]];then
    pip install tf_slim xgboost accelerate==0.21.0
elif [[ $(echo "${test_case}" | grep -c "nas") != 0 ]]; then
    pip install dynast==1.5.1
elif [[ $(echo "${test_case}" | grep -c "tf pruning") != 0 ]]; then
    pip install tensorflow-addons
fi
# test deps
pip install coverage
pip install pytest

pip list
echo "[DEBUG] list pipdeptree..."
pip install pipdeptree
pipdeptree
