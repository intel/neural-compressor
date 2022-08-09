#!/bin/bash

if [[ "${tensorflow_version}" == *"-official" ]]; then
    pip install tensorflow==${tensorflow_version%-official}
elif [[ "${tensorflow_version}" != "" ]]; then
    pip install intel-tensorflow==${tensorflow_version}
fi

pip install torch==${pytorch_version} -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==${torchvision_version} -f https://download.pytorch.org/whl/torch_stable.html

pip install onnx==${onnx_version}
pip install onnxruntime==${onnxruntime_version}

pip install mxnet==${mxnet_version}

cd /neural-compressor/test
if [ -f "requirements.txt" ]; then
    sed -i '/^neural-compressor/d' requirements.txt
    sed -i '/^intel-tensorflow/d' requirements.txt
    sed -i '/find-links https:\/\/download.pytorch.org\/whl\/torch_stable.html/d' requirements.txt
    sed -i '/^torch/d' requirements.txt
    sed -i '/^mxnet-mkl/d' requirements.txt
    sed -i '/^onnx>=/d;/^onnx==/d;/^onnxruntime>=/d;/^onnxruntime==/d' requirements.txt
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
