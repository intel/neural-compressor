#!/bin/bash
python -c "import neural_compressor as nc;print(nc.version.__version__)"
test_case="run basic others"
echo "${test_case}"

echo "specify fwk version..."
source /neural-compressor/.azure-pipelines/scripts/fwk_version.sh $1

echo "set up UT env..."
bash /neural-compressor/.azure-pipelines/scripts/ut/env_setup.sh "${test_case}"
export COVERAGE_RCFILE=/neural-compressor/.azure-pipelines/scripts/ut/coverage.file
lpot_path=$(python -c 'import neural_compressor; import os; print(os.path.dirname(neural_compressor.__file__))')
cd /neural-compressor/test || exit 1
find . -name "test*.py" | sed 's,\.\/,coverage run --source='"${lpot_path}"' --append ,g' | sed 's/$/ --verbose/'> run.sh
sed -i '/ adaptor\//d' run.sh
sed -i '/ tfnewapi\//d' run.sh
sed -i '/ neural_coder\//d' run.sh
sed -i '/ itex\//d' run.sh
sed -i '/ pruning_with_pt/d' run.sh
sed -i '/ pruning_with_tf/d' run.sh
sed -i '/ quantization/d' run.sh
sed -i '/ benchmark/d' run.sh
sed -i '/ export/d' run.sh
sed -i '/ mixed_precision/d' run.sh
sed -i '/ distillation\//d' run.sh
sed -i '/ scheduler\//d' run.sh
sed -i '/ nas\//d' run.sh
sed -i '/ 3x\//d' run.sh
sed -i '/ distributed\//d' run.sh

echo "copy model for dynas..."
mkdir -p .torch/ofa_nets || true
cp -r /tf_dataset/ut-localfile/ofa_mbv3_d234_e346_k357_w1.2 .torch/ofa_nets || true

LOG_DIR=/neural-compressor/log_dir
mkdir -p ${LOG_DIR}
ut_log_name=${LOG_DIR}/ut_tf_${tensorflow_version}_pt_${pytorch_version}.log

echo "cat run.sh..."
sort run.sh -o run.sh
cat run.sh | tee ${ut_log_name}
echo "------UT start-------"
bash -x run.sh 2>&1 | tee -a ${ut_log_name}
echo "------UT end -------"

if [ $(grep -c "FAILED" ${ut_log_name}) != 0 ] || [ $(grep -c "core dumped" ${ut_log_name}) != 0 ] || [ $(grep -c "ModuleNotFoundError:" ${ut_log_name}) != 0 ] || [ $(grep -c "OK" ${ut_log_name}) == 0 ];then
    echo "Find errors in UT test, please check the output..."
    exit 1
fi
cp .coverage ${LOG_DIR}/.coverage.others
echo "UT finished successfully! "