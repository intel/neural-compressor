#!/bin/bash
python -c "import neural_compressor as nc"
test_case="run 3x Torch Habana FP8"
echo "${test_case}"

echo "##[section]Run import check"
set -e
python -c "import neural_compressor.torch"
python -c "import neural_compressor.common"
echo "##[section]import check pass"

# install requirements
echo "##[group]set up UT env..."
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
sed -i '/^intel_extension_for_pytorch/d' /neural-compressor/test/3x/torch/requirements.txt
sed -i '/^auto_round/d' /neural-compressor/test/3x/torch/requirements.txt
cat /neural-compressor/test/3x/torch/requirements.txt
pip install -r /neural-compressor/test/3x/torch/requirements.txt
pip install pytest-cov
pip install pytest-html
pip install pytest-html-merger
echo "##[endgroup]"
pip list

export COVERAGE_RCFILE=/neural-compressor/.azure-pipelines/scripts/ut/3x/coverage.3x_pt_fp8
inc_path=$(python -c 'import neural_compressor; print(neural_compressor.__path__[0])')
LOG_DIR=/neural-compressor/log_dir
HTML_DIR=/neural-compressor/html_dir
mkdir -p ${LOG_DIR} ${HTML_DIR}

cd /neural-compressor/test/3x/torch/algorithms/fp8_quant

ut_log_name=${LOG_DIR}/ut_3x_pt_fp8.log
pytest --cov="${inc_path}" -vs --disable-warnings --html=report_1.html --self-contained-html /neural-compressor/test/3x/torch/quantization/weight_only/test_load.py 2>&1 | tee -a ${ut_log_name}
pytest --cov="${inc_path}" -vs --disable-warnings --html=report_2.html --self-contained-html /neural-compressor/test/3x/torch/quantization/weight_only/test_rtn.py 2>&1 | tee -a ${ut_log_name}
# pytest --cov="${inc_path}" -vs --disable-warnings --html=report_3.html --self-contained-html torch/quantization/weight_only/test_autoround.py 2>&1 | tee -a ${ut_log_name}
pytest --cov="${inc_path}" -vs --disable-warnings --html=report_4.html --self-contained-html . 2>&1 | tee -a ${ut_log_name}
pytest --cov="${inc_path}" -vs --disable-warnings --html=report_5.html --self-contained-html . 2>&1 | tee -a ${ut_log_name}
mv *.html ${HTML_DIR}

pytest_html_merger -i ${HTML_DIR} -o ${LOG_DIR}/report.html

if [ $(grep -c '== FAILURES ==' ${ut_log_name}) != 0 ] || [ $(grep -c '== ERRORS ==' ${ut_log_name}) != 0 ] || [ $(grep -c ' passed' ${ut_log_name}) == 0 ]; then
    echo "Find errors in pytest case, please check the output..."
    echo "Please search for '== FAILURES ==' or '== ERRORS =='"
    exit 1
fi

# if ut pass, collect the coverage file into artifacts
cp .coverage ${LOG_DIR}/.coverage

echo "UT finished successfully! "