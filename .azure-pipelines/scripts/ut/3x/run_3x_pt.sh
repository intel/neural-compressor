#!/bin/bash
python -c "import neural_compressor as nc"
test_case="run 3x Torch"
echo "${test_case}"

# install requirements
echo "set up UT env..."
pip install -r /neural-compressor/test/3x/torch/requirements.txt
pip install pytest-cov
pip install pytest-html
pip list

export COVERAGE_RCFILE=/neural-compressor/.azure-pipelines/scripts/ut/3x/coverage.3x_pt
inc_path=$(python -c 'import neural_compressor; print(neural_compressor.__path__[0])')
cd /neural-compressor/test/3x || exit 1
rm -rf tensorflow
rm -rf onnxrt

LOG_DIR=/neural-compressor/log_dir
mkdir -p ${LOG_DIR}
ut_log_name=${LOG_DIR}/ut_3x_pt.log
pytest --cov="${inc_path}" -vs --disable-warnings --html=report.html --self-contained-html . 2>&1 | tee -a ${ut_log_name}

cp .coverage ${LOG_DIR}/.coverage
cp report.html ${LOG_DIR}/

if [ $(grep -c '== FAILURES ==' ${ut_log_name}) != 0 ] || [ $(grep -c '== ERRORS ==' ${ut_log_name}) != 0 ] || [ $(grep -c ' passed' ${ut_log_name}) == 0 ]; then
    echo "Find errors in pytest case, please check the output..."
    echo "Please search for '== FAILURES ==' or '== ERRORS =='"
    exit 1
fi

echo "UT finished successfully! "