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
    pip install /tf_dataset/tf_binary/230928/tensorflow*.whl
    pip install cmake
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

if [[ "${ipex_version}" != "" ]]; then
    pip install intel-extension-for-pytorch=="${ipex_version%+cpu}"
fi

if [[ "${onnx_version}" != "" ]]; then
    pip install onnx==${onnx_version}
fi

if [[ "${onnxruntime_version}" != "" ]]; then
    pip install onnxruntime==${onnxruntime_version}
    if [[ "${onnxruntime_version}" == "1.14"* ]]; then
        pip install onnxruntime-extensions==0.8.0
    else
        pip install onnxruntime-extensions
    fi
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
pip install cmake
pip install transformers

if [[ $(echo "${test_case}" | grep -c "others") != 0 ]];then
    pip install tf_slim xgboost accelerate==0.21.0 peft
elif [[ $(echo "${test_case}" | grep -c "nas") != 0 ]]; then
    pip install dynast==1.6.0rc1
elif [[ $(echo "${test_case}" | grep -c "tf pruning") != 0 ]]; then
    pip install tensorflow-addons
    # Workaround
    # horovod can't be install in the env with TF and PT together
    # so test distribute cases in the env with single fw installed
    pip install horovod
fi

if [[ $(echo "${test_case}" | grep -c "api") != 0 ]] || [[ $(echo "${test_case}" | grep -c "adaptor") != 0 ]]; then
    pip install git+https://github.com/intel/auto-round.git@d02f94d4b085523df3b313863fb07f83b2989cce
fi

# test deps
pip install coverage
pip install pytest
pip install pytest-html

pip list
echo "[DEBUG] list pipdeptree..."
pip install pipdeptree
pipdeptree

# import torch before import tensorflow
if [[ $(echo "${test_case}" | grep -c "run basic api") != 0 ]] || [[ $(echo "${test_case}" | grep -c "run basic others") != 0 ]] || [[ $(echo "${test_case}" | grep -c "run basic adaptor") != 0 ]]; then
    cd /neural-compressor/test || exit 1
    find . -name "test*.py" | xargs sed -i 's/import tensorflow as tf/import torch; import tensorflow as tf/g'
    find . -name "test*.py" | xargs sed -i 's/import tensorflow.compat.v1 as tf/import torch; import tensorflow.compat.v1 as tf/g'
    find . -name "test*.py" | xargs sed -i 's/from tensorflow import keras/import torch; from tensorflow import keras/g'
fi

