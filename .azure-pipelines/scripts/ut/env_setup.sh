#!/bin/bash

if [[ "${tensorflow_version}" == *"-official" ]]; then
    pip install tensorflow==${tensorflow_version%-official}
elif [[ "${tensorflow_version}" == "spr-base" ]]; then
    pip install /tf_dataset/tf_binary/tensorflow*.whl
    if [[ $? -ne 0 ]]; then
      exit 1
    fi
elif [[ "${tensorflow_version}" != "" ]]; then
    pip install intel-tensorflow==${tensorflow_version}
fi

if [[ "${pytorch_version}" != "" ]]; then
    pip install torch==${pytorch_version} -f https://download.pytorch.org/whl/torch_stable.html
fi

if [[ "${torchvision_version}" != "" ]]; then
    pip install torchvision==${torchvision_version} -f https://download.pytorch.org/whl/torch_stable.html
fi

if [[ "${onnx}" != "" ]]; then
    pip install onnx==${onnx_version}
fi

if [[ "${onnxruntime}" != "" ]]; then
    pip install onnxruntime==${onnxruntime_version}
    pip install onnxruntime-extensions
fi

if [[ "${mxnet}" != "" ]]; then
    pip install mxnet==${mxnet_version}
fi

cd /neural-compressor/test
if [ -f "requirements.txt" ]; then
    sed -i '/^neural-compressor/d' requirements.txt
    sed -i '/^intel-tensorflow/d' requirements.txt
    sed -i '/find-links https:\/\/download.pytorch.org\/whl\/torch_stable.html/d' requirements.txt
    sed -i '/^torch/d;/^torchvision/d' requirements.txt
    sed -i '/^mxnet-mkl/d' requirements.txt
    sed -i '/^onnx/d;/^onnxruntime/d;/^onnxruntime-extensions/d' requirements.txt
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
