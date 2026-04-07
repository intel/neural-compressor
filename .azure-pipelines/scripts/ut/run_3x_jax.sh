#!/bin/bash
python -c "import neural_compressor as nc"
test_case="run 3x JAX"
echo "${test_case}"

echo "##[section]Run import check"
set -e
python -c "import neural_compressor.jax"
python -c "import neural_compressor.common"
echo "##[section]import check pass"

# install requirements
echo "##[group]set up UT env..."
uv pip install -r /neural-compressor/requirements_jax.txt
# Check if test/jax/requirements.txt exists and install it (optional)
if [ -f /neural-compressor/test/jax/requirements.txt ]; then
    uv pip install -r /neural-compressor/test/jax/requirements.txt
fi
uv pip install pytest-cov pytest-html pytest-html-merger beautifulsoup4==4.13.5
echo "##[endgroup]"
uv pip list

export COVERAGE_RCFILE=/neural-compressor/.azure-pipelines/scripts/ut/coverage.3x_jax
inc_path=$(python -c 'import neural_compressor; print(neural_compressor.__path__[0])')
cd /neural-compressor/test || exit 1

LOG_DIR=/neural-compressor/log_dir
mkdir -p ${LOG_DIR}
ut_log_name=${LOG_DIR}/ut_3x_jax.log

# test for jax ut
pytest --cov="${inc_path}" -vs --disable-warnings --html=report_jax.html --self-contained-html ./jax 2>&1 | tee -a ${ut_log_name}

cp report_jax.html ${LOG_DIR}/report.html

set -x
if [ $(grep -c '== FAILURES ==' ${ut_log_name}) != 0 ] || [ $(grep -c '== ERRORS ==' ${ut_log_name}) != 0 ] || \
[ $(grep -c 'Killed' ${ut_log_name}) != 0 ] || [ $(grep -c 'core dumped' ${ut_log_name}) != 0 ] || [ $(grep -c ' passed' ${ut_log_name}) == 0 ]; then
    echo "Find errors in pytest case, please check the output..."
    echo "Please search for '== FAILURES ==' or '== ERRORS =='"
    exit 1
fi

# if ut pass, collect the coverage file into artifacts
cp .coverage ${LOG_DIR}/.coverage

echo "UT finished successfully!"