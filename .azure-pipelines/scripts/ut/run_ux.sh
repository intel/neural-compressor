#!/bin/bash
python -c "import neural_compressor as nc;print(nc.version.__version__)"
echo "run ux"

echo "specify fwk version..."
export tensorflow_version='2.10.0'
export pytorch_version='1.12.0+cpu'
export torchvision_version='0.13.0+cpu'
export onnx_version='1.12.0'
export onnxruntime_version='1.13.1'

echo "set up UT env..."
bash /neural-compressor/.azure-pipelines/scripts/ut/env_setup.sh
cd /neural-compressor/test || exit 1
find ./ux -name "test*.py" | sed 's,\.\/,python ,g' | sed 's/$/ --verbose/' > run.sh

LOG_DIR=/neural-compressor/log_dir
mkdir -p ${LOG_DIR}
ut_log_name=${LOG_DIR}/ut_tf_${tensorflow_version}_pt_${pytorch_version}.log

echo "cat run.sh..."
cat run.sh | tee ${ut_log_name}
echo "------UT start-------"
bash run.sh 2>&1 | tee -a ${ut_log_name}
echo "------UT end -------"

if [ $(grep -c "FAILED" ${ut_log_name}) != 0 ] || [ $(grep -c "OK" ${ut_log_name}) == 0 ];then
    echo "Find errors in UT test, please check the output..."
    exit 1
fi
echo "UT finished successfully! "