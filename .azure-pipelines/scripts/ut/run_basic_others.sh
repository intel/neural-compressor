#!/bin/bash
set -x
python -c "import neural_compressor as nc;print(nc.version.__version__)"
echo "run basic others"

echo "specify fwk version..."
export tensorflow_version='2.10.0'
export pytorch_version='1.12.0+cpu'
export torchvision_version='0.13.0+cpu'
export onnx_version='1.12.0'
export onnxruntime_version='1.13.1'
export mxnet_version='1.9.1'

echo "set up UT env..."
bash /neural-compressor/.azure-pipelines/scripts/ut/env_setup.sh
lpot_path=$(python -c 'import neural_compressor; import os; print(os.path.dirname(neural_compressor.__file__))')
echo "copy pre-train model..."
mkdir -p /tmp/.neural_compressor/inc_ut || true
cp -r /tf_dataset/ut-localfile/resnet_v2 /tmp/.neural_compressor/inc_ut || true

cd /neural-compressor/test || exit 1
find . -name "test*.py" | sed 's,\.\/,coverage run --source='"${lpot_path}"' --append ,g' | sed 's/$/ --verbose/'> run.sh
sed -i '/ adaptor\//d' run.sh
sed -i '/ tfnewapi\//d' run.sh
sed -i '/ ux\//d' run.sh
sed -i '/ neural_coder\//d' run.sh
sed -i '/ ipex\//d' run.sh
sed -i '/ itex\//d' run.sh

LOG_DIR=/neural-compressor/log_dir
mkdir -p ${LOG_DIR}
ut_log_name=${LOG_DIR}/ut_tf_${tensorflow_version}_pt_${pytorch_version}.log

echo "cat run.sh..."
cat run.sh | tee ${ut_log_name}
echo "-------------"
bash run.sh 2>&1 | tee -a ${ut_log_name}
cp .coverage ${LOG_DIR}/.coverage.others
echo "list all in ${LOG_DIR}"
ls -a ${LOG_DIR}
if [ $(grep -c "FAILED" ${ut_log_name}) != 0 ] || [ $(grep -c "OK" ${ut_log_name}) == 0 ];then
    exit 1
fi