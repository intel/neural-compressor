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
cd /neural-compressor/test/3x || exit 1
grep -lrv "import pytest" --include="test*.py" ./torch | sed 's,\.\/,coverage run --source='"${inc_path}"' --append ,g' | sed 's/$/ --verbose/'> run_unittest.sh
grep -lrv "import pytest" --include="test*.py" ./common | sed 's,\.\/,coverage run --source='"${inc_path}"' --append ,g' | sed 's/$/ --verbose/'>> run_unittest.sh
grep -lr "import pytest" --include="test*.py" ./torch | sed 's,\.\/,coverage run --source='"${inc_path}"' --append -m pytest ,g' | sed 's/$/ --verbose/'> run_pytest.sh
grep -lr "import pytest" --include="test*.py" ./common | sed 's,\.\/,coverage run --source='"${inc_path}"' --append -m pytest ,g' | sed 's/$/ --verbose/'>> run_pytest.sh

LOG_DIR=/neural-compressor/log_dir
mkdir -p ${LOG_DIR}
ut_log_name=${LOG_DIR}/ut_3x_pt.log

echo "cat run_unittest.sh..."
sort run_unittest.sh -o run_unittest.sh
cat run_unittest.sh | tee ${ut_log_name}
echo "------unittest start-------"
bash -x run_unittest.sh 2>&1 | tee -a ${ut_log_name}
echo "------unittest end -------"

echo "cat run_pytest.sh..."
sort run_pytest.sh -o run_pytest.sh
cat run_pytest.sh | tee -a ${ut_log_name}
echo "------pytest start-------"
bash -x run_pytest.sh 2>&1 | tee -a ${ut_log_name}
echo "------pytest end -------"

cp .coverage ${LOG_DIR}/.coverage

# check unittest issue
if [ $(grep -c "FAILED" ${ut_log_name}) != 0 ] || [ $(grep -c "core dumped" ${ut_log_name}) != 0 ] \
|| [ $(grep -c "ModuleNotFoundError:" ${ut_log_name}) != 0 ] || [ $(grep -c "ImportError:" ${ut_log_name}) != 0 ] || [ $(grep -c "OK" ${ut_log_name}) == 0 ];then
    echo "Find errors in UT test, please check the output..."
    exit 1
fi

# check pytest issue
if [ $(grep -c 'failed,' ${ut_log_name}) != 0 ] || [ $(grep -c 'passed,' ${ut_log_name}) ]; then
    echo "Find errors in UT test, please check the output..."
    exit 1
fi

echo "UT finished successfully! "