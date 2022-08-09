#!/bin/bash
set -x
python -c "import neural_compressor as nc;print(nc.version.__version__)"
echo "run basic adaptor tfnewapi"

echo "specify fwk version..."
export tensorflow_version='spr-base'

echo "set up UT env..."
bash /neural-compressor/.azure-pipelines/scripts/ut/env_setup.sh

cd /neural-compressor/test || exit 1
find ./tfnewapi -name "test*.py" | sed 's,\.\/,python ,g' | sed 's/$/ --verbose/' > run.sh

LOG_DIR=/neural-compressor/log_dir
ut_log_name=${LOG_DIR}/ut_tf_newapi.log

echo "cat run.sh..."
cat run.sh | tee ${ut_log_name}
echo "-------------"
bash run.sh 2>&1 | tee -a ${ut_log_name}

if [ $(grep -c "FAILED" ${ut_log_name}) != 0 ] || [ $(grep -c "OK" ${ut_log_name}) == 0 ];then
    exit 1
fi