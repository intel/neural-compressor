#!/bin/bash
set -x
python -c "import neural_compressor as nc;print(nc.version.__version__)"
echo "run basic"

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
find . -name "test*.py" | sed 's,\.\/,python ,g' | sed 's/$/ --verbose/' > run.sh
sed -i '/ adaptor\//d' run.sh
sed -i '/ tfnewapi\//d' run.sh
sed -i '/ ux\//d' run.sh
sed -i '/ neural_coder\//d' run.sh

LOG_DIR=/neural-compressor/log_dir
ut_log_name=${LOG_DIR}/ut_tf_${tensorflow_version}_pt_${pytorch_version}.log

echo "cat run.sh..."
cat run.sh | tee ${ut_log_name}
echo "-------------"
bash run.sh 2>&1 | tee -a ${ut_log_name}

if [ $(grep -c "FAILED" ${ut_log_name}) != 0 ] || [ $(grep -c "OK" ${ut_log_name}) == 0 ];then
    exit 1
fi