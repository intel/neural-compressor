#!/bin/bash
python -c "import neural_compressor as nc"
test_case="run 3x Torch"
echo "${test_case}"

# install requirements
echo "set up UT env..."
pip install -r /neural-compressor/test/3x/torch/requirements.txt
pip install coverage
pip install pytest-html
pip list

export COVERAGE_RCFILE=/neural-compressor/.azure-pipelines/scripts/ut/3x/coverage.3x_pt
inc_path=$(python -c 'import neural_compressor; print(neural_compressor.__path__[0])')
cd /neural-compressor/test/3x || exit 1
#grep -lrv "import pytest" --include="test*.py" ./torch | sed 's,\.\/,coverage run --source='"${inc_path}"' --append ,g' | sed 's/$/ --verbose/'> run_unittest.sh
#grep -lrv "import pytest" --include="test*.py" ./common | sed 's,\.\/,coverage run --source='"${inc_path}"' --append ,g' | sed 's/$/ --verbose/'>> run_unittest.sh
#grep -lr "import pytest" --include="test*.py" ./torch | sed 's,\.\/,coverage run --source='"${inc_path}"' --append -m pytest --disable-warnings -v ,g' > run_pytest.sh
#grep -lr "import pytest" --include="test*.py" ./common | sed 's,\.\/,coverage run --source='"${inc_path}"' --append -m pytest --disable-warnings -v ,g'>> run_pytest.sh
rm -rf tensorflow
rm -rf onnxrt

LOG_DIR=/neural-compressor/log_dir
mkdir -p ${LOG_DIR}
ut_log_name=${LOG_DIR}/ut_3x_pt.log

# unittest and pytest has some incompatible issue, so separate the test.
#echo "cat run_unittest.sh..."
#sort run_unittest.sh -o run_unittest.sh
#cat run_unittest.sh | tee ${ut_log_name}
echo "------unittest start-------"
#bash -x run_unittest.sh 2>&1 | tee -a ${ut_log_name}
coverage run --source="${inc_path}" -m pytest ./torch --disable-warnings -v --html=report.html --self-contained-html 2>&1 | tee -a ${ut_log_name}
echo "------unittest end -------"

#if [ -s run_pytest.sh ]; then
#    echo "cat run_pytest.sh..."
#    sort run_pytest.sh -o run_pytest.sh
#    cat run_pytest.sh | tee -a ${ut_log_name}
#    echo "------pytest start-------"
#    bash -x run_pytest.sh 2>&1 | tee -a ${ut_log_name}
#    echo "------pytest end -------"
#fi

cp .coverage ${LOG_DIR}/.coverage
cp report.html ${LOG_DIR}/

ut_status="passed"
## check unittest issue
#if [ $(grep -c "FAILED" ${ut_log_name}) != 0 ] || [ $(grep -c "core dumped" ${ut_log_name}) != 0 ] \
#|| [ $(grep -c "ModuleNotFoundError:" ${ut_log_name}) != 0 ] || [ $(grep -c "ImportError:" ${ut_log_name}) != 0 ] || [ $(grep -c "OK" ${ut_log_name}) == 0 ];then
#    echo "Find errors in unittest case, please check the output..."
#    echo "Please search for 'FAILED' or 'core dumped' or 'ModuleNotFoundError:' or 'ImportError:'"
#    ut_status="failed"
#fi
#
## check pytest issue
#if [ -s run_pytest.sh ]; then
#    if [ $(grep -c '== FAILURES ==' ${ut_log_name}) != 0 ] || [ $(grep -c '== ERRORS ==' ${ut_log_name}) != 0 ] || [ $(grep -c ' passed ' ${ut_log_name}) == 0 ]; then
#        echo "Find errors in pytest case, please check the output..."
#        echo "Please search for '== FAILURES ==' or '== ERRORS =='"
#        ut_status="failed"
#    fi
#fi

if [ $(grep -c '== FAILURES ==' ${ut_log_name}) != 0 ] || [ $(grep -c '== ERRORS ==' ${ut_log_name}) != 0 ] || [ $(grep -c ' passed ' ${ut_log_name}) == 0 ]; then
    echo "Find errors in pytest case, please check the output..."
    echo "Please search for '== FAILURES ==' or '== ERRORS =='"
    ut_status="failed"
fi

if [ "$ut_status" = "failed" ]; then
    exit 1
fi

echo "UT finished successfully! "