#!/bin/bash
set -x

echo "specify fwk version..."
export tensorflow_version='2.9.1'
export pytorch_version='1.11.0+cpu'
export torchvision_version='0.12.0'
export onnx_version='1.9.0'
export onnxruntime_version='1.10.0'
export mxnet_version='1.7.0'

echo "set up UT env..."
bash /neural-compressor/.azure-pipelines/scripts/ut/env_setup.sh

cd /neural-compressor/test || exit 1
find ./adaptor -name "test*.py" | sed 's,\.\/,python ,g' | sed 's/$/ --verbose/' > run.sh
ut_log_name=${WORKSPACE}/ut_tf_${tensorflow_version}_pt_${pytorch_version}.log

echo "cat run.sh..."
cat run.sh
echo "-------------"
bash run.sh 2>&1 | tee ${ut_log_name}

if [ $(grep -c "FAILED" ${ut_log_name}) != 0 ] || [ $(grep -c "OK" ${ut_log_name}) == 0 ];then
    exit 1
fi