#!/bin/bash
python -c "import neural_compressor as nc"
test_case="run 3x Torch"
echo "${test_case}"

echo "##[section]Run import check"
set -e
python -c "import neural_compressor.torch"
python -c "import neural_compressor.common"
echo "##[section]import check pass"

# install requirements
echo "##[group]set up UT env..."
export LD_LIBRARY_PATH=${HOME}/.local/lib/:$LD_LIBRARY_PATH
sed -i '/^deepspeed/d' /neural-compressor/test/3x/torch/requirements.txt
pip install -r /neural-compressor/test/3x/torch/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install pytest-cov
pip install pytest-html
pip install beautifulsoup4==4.13.5
echo "##[endgroup]"
pip list

export COVERAGE_RCFILE=/neural-compressor/.azure-pipelines/scripts/ut/3x/coverage.3x_pt
inc_path=$(python -c 'import neural_compressor; print(neural_compressor.__path__[0])')
cd /neural-compressor/test/3x || exit 1
rm -rf torch/algorithms/fp8_quant
rm -rf torch/quantization/fp8_quant
rm -rf torch/algorithms/fp8_quant_xpu

LOG_DIR=/neural-compressor/log_dir
mkdir -p ${LOG_DIR}
ut_log_name=${LOG_DIR}/ut_3x_pt.log

(find ./common -name "test*.py"; find ./torch -name "test*.py") | sed "s,\.\/,python -m pytest --cov=\"${inc_path}\" --cov-report term --html=report.html --self-contained-html  --cov-report xml:coverage.xml --cov-append -vs --disable-warnings ,g" > run.sh
cat run.sh
numactl --physcpubind="${NUMA_CPUSET:-0-15}" --membind="${NUMA_NODE:-0}" bash run.sh 2>&1 | tee ${ut_log_name}

cp report.html ${LOG_DIR}/

if [ $(grep -c '== FAILURES ==' ${ut_log_name}) != 0 ] || [ $(grep -c '== ERRORS ==' ${ut_log_name}) != 0 ] || [ $(grep -c ' passed' ${ut_log_name}) == 0 ]; then
    echo "Find errors in pytest case, please check the output..."
    echo "Please search for '== FAILURES ==' or '== ERRORS =='"
    exit 1
fi

# if ut pass, collect the coverage file into artifacts
cp .coverage ${LOG_DIR}/.coverage

echo "UT finished successfully! "