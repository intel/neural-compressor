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

if [[ "${itex_version}" == "nightly" ]]; then
    pip install /tf_dataset/itex_binary/221209/intel_extension_for_tensorflow-1.1.0-cp38-cp38-linux_x86_64.whl
    pip install /tf_dataset/itex_binary/221209/intel_extension_for_tensorflow_lib-1.1.0.0-cp38-cp38-linux_x86_64.whl
elif [[ "${itex_version}" != "" ]]; then
    pip install --upgrade intel-extension-for-tensorflow[cpu]==${itex_version}
fi

if [[ "${pytorch_version}" != "" ]]; then
    pip install torch==${pytorch_version} -f https://download.pytorch.org/whl/torch_stable.html
fi

if [[ "${torchvision_version}" != "" ]]; then
    pip install torchvision==${torchvision_version} -f https://download.pytorch.org/whl/torch_stable.html
fi

if [[ "${ipex_version}" == "1.12.0+cpu" ]]; then
    ipex_whl="http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/wheels/v1.12.0/intel_extension_for_pytorch-1.12.0%2Bcpu-cp38-cp38-linux_x86_64.whl"
    pip install $ipex_whl
elif [[ "${ipex_version}" == "1.13.0+cpu" ]]; then
    ipex_whl="https://github.com/intel/intel-extension-for-pytorch/releases/download/v1.13.0%2Bcpu/intel_extension_for_pytorch-1.13.0-cp38-cp38-manylinux2014_x86_64.whl"
    pip install $ipex_whl
elif [[ "${ipex_version}" == "2.0.0+cpu" ]]; then
    ipex_whl="https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/cpu/intel_extension_for_pytorch-2.0.0%2Bcpu-cp38-cp38-linux_x86_64.whl"
    pip install $ipex_whl
fi

if [[ "${onnx_version}" != "" ]]; then
    pip install onnx==${onnx_version}
fi

if [[ "${onnxruntime_version}" != "" ]]; then
    pip install onnxruntime==${onnxruntime_version}
    pip install onnxruntime-extensions
    pip install optimum
fi

if [ "${mxnet_version}" != '' ]; then
    pip install numpy==1.23.5
    echo "re-install pycocotools resolve the issue with numpy..."
    pip uninstall pycocotools -y
    pip install --no-cache-dir pycocotools
    pip install mxnet==${mxnet_version}
fi

cd /neural-compressor/test
if [ -f "requirements.txt" ]; then
    sed -i '/^neural-compressor/d' requirements.txt
    sed -i '/^intel-tensorflow/d' requirements.txt
    sed -i '/find-links https:\/\/download.pytorch.org\/whl\/torch_stable.html/d' requirements.txt
    sed -i '/^torch/d;/^torchvision/d;/^intel-extension-for-pytorch/d' requirements.txt
    sed -i '/^mxnet-mkl/d' requirements.txt
    sed -i '/^onnx/d;/^onnxruntime/d;/^onnxruntime-extensions/d;/^optimum/d' requirements.txt
    n=0
    until [ "$n" -ge 3 ]
    do
        python -m pip install --no-cache-dir -r requirements.txt && break
        n=$((n+1))
        sleep 5
    done
    pip list
else
    echo "Not found requirements.txt file."
fi

pip install coverage
pip install pytest
