#!/bin/bash
python -c "import neural_compressor as nc"
test_case="run ONNXRT"
echo "${test_case}"

# install requirements
echo "set up UT env..."
pip install -r /neural-compressor/test/onnxrt/requirements.txt
pip install pytest-cov
pip install pytest-html
pip list

export COVERAGE_RCFILE=/neural-compressor/.azure-pipelines/scripts/ut/coverage.ort
inc_path=$(python -c 'import neural_compressor_ort; print(neural_compressor_ort.__path__[0])')
cd /neural-compressor/test || exit 1

LOG_DIR=/neural-compressor/log_dir
mkdir -p ${LOG_DIR}
ut_log_name=${LOG_DIR}/ut_ort.log
pytest --cov="${inc_path}" -vs --disable-warnings --html=report.html --self-contained-html . 2>&1 | tee -a ${ut_log_name}

cp report.html ${LOG_DIR}/

if [ $(grep -c '== FAILURES ==' ${ut_log_name}) != 0 ] || [ $(grep -c '== ERRORS ==' ${ut_log_name}) != 0 ] || [ $(grep -c ' passed' ${ut_log_name}) == 0 ]; then
    echo "Find errors in pytest case, please check the output..."
    echo "Please search for '== FAILURES ==' or '== ERRORS =='"
    exit 1
fi

# if ut pass, collect the coverage file into artifacts
cp .coverage ${LOG_DIR}/.coverage

echo "UT finished successfully! "