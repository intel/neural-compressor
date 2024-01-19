#!/bin/bash
python -c "import neural_compressor as nc"
test_case="run 3x Torch"
echo "${test_case}"

# install requirements
echo "set up UT env..."
pip install -r /neural-compressor/test/3x/torch/requirements.txt
pip install coverage
pip list

export COVERAGE_RCFILE=/neural-compressor/.azure-pipelines/scripts/ut/3x/coverage.3x_pt
inc_path=$(python -c 'import neural_compressor; print(neural_compressor.__path__[0])')
cd /neural-compressor/test || exit 1
find ./3x/torch/* -name "test*.py" | sed 's,\.\/,coverage run --source='"${inc_path}"' --append ,g' | sed 's/$/ --verbose/'> run.sh

LOG_DIR=/neural-compressor/log_dir
mkdir -p ${LOG_DIR}
ut_log_name=${LOG_DIR}/ut_3x_pt.log

echo "cat run.sh..."
sort run.sh -o run.sh
cat run.sh | tee ${ut_log_name}
echo "------UT start-------"
bash -x run.sh 2>&1 | tee -a ${ut_log_name}
cp .coverage ${LOG_DIR}/.coverage

echo "------UT end -------"

if [ $(grep -c "FAILED" ${ut_log_name}) != 0 ] || [ $(grep -c "core dumped" ${ut_log_name}) != 0 ] || [ $(grep -c "ModuleNotFoundError:" ${ut_log_name}) != 0 ] || [ $(grep -c "OK" ${ut_log_name}) == 0 ];then
    echo "Find errors in UT test, please check the output..."
    exit 1
fi
echo "UT finished successfully! "