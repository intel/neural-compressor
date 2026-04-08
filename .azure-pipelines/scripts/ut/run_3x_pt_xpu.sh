#!/bin/bash
python -c "import neural_compressor as nc"
test_case="run 3x Torch with XPU"
echo "${test_case}"

echo "##[section]Run import check"
set -e
python -c "import neural_compressor.torch"
python -c "import neural_compressor.common"
echo "##[section]import check pass"

echo "##[group]set up UT env..."
export LD_LIBRARY_PATH=${HOME}/.venv/lib/:$LD_LIBRARY_PATH
uv pip install -r /neural-compressor/test/torch/requirements_xpu.txt --extra-index-url https://download.pytorch.org/whl/xpu
uv pip install pytest-cov pytest-html
uv pip list
echo "##[endgroup]"

echo "##[group]check xpu env..."
echo "ZE_AFFINITY_MASK: ${ZE_AFFINITY_MASK}"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("xpu available:", torch.xpu.is_available())
print("xpu count:", torch.xpu.device_count())
PY
echo "##[endgroup]"

export COVERAGE_RCFILE=/neural-compressor/.azure-pipelines/scripts/ut/coverage.3x_pt
inc_path=$(python -c 'import neural_compressor; print(neural_compressor.__path__[0])')
cd /neural-compressor/test || exit 1

LOG_DIR=/neural-compressor/log_dir
mkdir -p ${LOG_DIR}
ut_log_name=${LOG_DIR}/ut_3x_pt_xpu.log

find ./torch -name "test_autoround_xpu.py" | sed "s,\.\/,python -m pytest --cov=\"${inc_path}\" --cov-report term --html=report.html --self-contained-html --cov-report xml:coverage.xml --cov-append -vs --disable-warnings ,g" > run_xpu.sh
cat run_xpu.sh
numactl --physcpubind="${NUMA_CPUSET:-0-27}" --membind="${NUMA_NODE:-0}" bash run_xpu.sh 2>&1 | tee ${ut_log_name}

cp report.html ${LOG_DIR}/

set -x
if [ $(grep -c '== FAILURES ==' ${ut_log_name}) != 0 ] || [ $(grep -c '== ERRORS ==' ${ut_log_name}) != 0 ] || \
[ $(grep -c 'Killed' ${ut_log_name}) != 0 ] || [ $(grep -c 'core dumped' ${ut_log_name}) != 0 ] || [ $(grep -c ' passed' ${ut_log_name}) == 0 ]; then
    echo "##[error]Find errors in pytest case, please check the output..."
    exit 1
fi

# if ut pass, collect the coverage file into artifacts
cp .coverage ${LOG_DIR}/.coverage

echo "UT finished successfully! "