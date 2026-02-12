#!/bin/bash
python -c "import neural_compressor as nc"
test_case="run 3x TensorFlow"
echo "${test_case}"

echo "##[section]Run import check"
set -e
python -c "import neural_compressor.tensorflow"
python -c "import neural_compressor.common"
echo "##[section]import check pass"

# install requirements
echo "##[group]set up UT env..."
pip install -r /neural-compressor/test/3x/tensorflow/requirements.txt
pip install pytest-cov
pip install pytest-html
pip install pytest-html-merger
pip install beautifulsoup4==4.13.5
echo "##[endgroup]"
pip list

export COVERAGE_RCFILE=/neural-compressor/.azure-pipelines/scripts/ut/3x/coverage.3x_tf
inc_path=$(python -c 'import neural_compressor; print(neural_compressor.__path__[0])')
cd /neural-compressor/test/3x || exit 1
rm -rf torch
rm -rf onnxrt
mv tensorflow/keras ../3x_keras
mv tensorflow/quantization/ptq/newapi ../3x_newapi

LOG_DIR=/neural-compressor/log_dir
mkdir -p ${LOG_DIR}
ut_log_name=${LOG_DIR}/ut_3x_tf.log

# test for tensorflow ut
pytest --cov="${inc_path}" -vs --disable-warnings --html=report_tf_quant.html --self-contained-html ./tensorflow/quantization 2>&1 | tee -a ${ut_log_name}
rm -rf tensorflow/quantization
pytest --cov="${inc_path}" --cov-append -vs --disable-warnings --html=report_tf_test_quantize_model.html --self-contained-html ./tensorflow/test_quantize_model.py 2>&1 | tee -a ${ut_log_name}
rm -rf tensorflow/test_quantize_model.py
pytest --cov="${inc_path}" --cov-append -vs --disable-warnings --html=report_tf.html --self-contained-html . 2>&1 | tee -a ${ut_log_name}

# test for tensorflow new api ut
pip uninstall tensorflow -y
pip install /tf_dataset/tf_binary/230928/tensorflow*.whl
pip install protobuf==3.20.3
pip install horovod==0.27.0
pip list
rm -rf tensorflow/*
mkdir -p tensorflow/quantization/ptq
mv ../3x_newapi tensorflow/quantization/ptq/newapi
find . -name "test*.py" | sed "s,\.\/,python -m pytest --cov=${inc_path} --cov-append -vs --disable-warnings ,g" > run.sh
cat run.sh
numactl --physcpubind="${NUMA_CPUSET:-0-15}" --membind="${NUMA_NODE:-0}" bash run.sh 2>&1 | tee -a ${ut_log_name}

# test for itex ut
rm -rf tensorflow/*
mv ../3x_keras tensorflow/keras
pip uninstall tensorflow -y
pip install intel-extension-for-tensorflow[cpu]
pytest --cov="${inc_path}" --cov-append -vs --disable-warnings --html=report_keras.html --self-contained-html ./tensorflow 2>&1 | tee -a ${ut_log_name}

mkdir -p report
mv *.html report
pytest_html_merger -i ./report -o ./report.html

cp report.html ${LOG_DIR}/

if [ $(grep -c '== FAILURES ==' ${ut_log_name}) != 0 ] || [ $(grep -c '== ERRORS ==' ${ut_log_name}) != 0 ] || [ $(grep -c ' passed' ${ut_log_name}) == 0 ]; then
    echo "Find errors in pytest case, please check the output..."
    echo "Please search for '== FAILURES ==' or '== ERRORS =='"
    exit 1
fi

# if ut pass, collect the coverage file into artifacts
cp .coverage ${LOG_DIR}/.coverage

echo "UT finished successfully! "