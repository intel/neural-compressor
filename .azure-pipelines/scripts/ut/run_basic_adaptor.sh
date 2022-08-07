#!/bin/bash
set -x

python -c "import neural_compressor as nc;print(nc.version.__version__)"
echo "run basic adaptor"

env | sort
echo ${tensorflow_version}
echo ${pytorch_version}
export onnx_version='1.9.0'
export onnxruntime_version='1.10.0'
export mxnet_version='1.7.0'


echo "set up UT env..."
bash /neural-compressor/.azure-pipelines/scripts/ut/env_setup.sh

cd /neural-compressor/test/adaptor || exit 1
find . -name "test*.py" | sed 's,\.\/,python ,g' | sed 's/$/ --verbose/'  > run.sh
ut_log_name=${WORKSPACE}/ut_tf_${tensorflow_version}_pt_${pytorch_version}.log

echo "cat run.sh..."
cat run.sh
echo "-------------"
bash run.sh 2>&1 | tee ${ut_log_name}

if [ $(grep -c "FAILED" ${ut_log_name}) != 0 ] || [ $(grep -c "OK" ${ut_log_name}) == 0 ];then
    exit 1
fi