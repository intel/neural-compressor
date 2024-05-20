#!/bin/bash
python -c "import neural_compressor as nc"
test_case="run 3x New TF API"
echo "${test_case}"

# install requirements
echo "set up UT env..."
pip install -r /neural-compressor/test/3x/tensorflow/requirements.txt
pip install pytest-html
pip install pytest-html-merger

pip uninstall tensorflow -y
pip install /tf_dataset/tf_binary/230928/tensorflow*.whl
pip install cmake
pip install protobuf==3.20.3
pip install horovod==0.27.0
pip list

cd /neural-compressor/test/3x || exit 1
mv tensorflow/quantization/ptq/newapi ../3x_newapi
rm -rf ./*

LOG_DIR=/neural-compressor/log_dir
mkdir -p ${LOG_DIR}
ut_log_name=${LOG_DIR}/ut_3x_new_tf.log

mkdir -p tensorflow/quantization/ptq
mv ../3x_newapi tensorflow/quantization/ptq/newapi

pytest -vs --disable-warnings --html=report_new_tf_quant_one_case.html --self-contained-html ./tensorflow/quantization/ptq/newapi/test_big_saved_model.py 2>&1 | tee -a ${ut_log_name}
rm -rf tensorflow/quantization/ptq/newapi/test_big_saved_model.py
pytest -vs --disable-warnings --html=report_new_tf_quant.html --self-contained-html ./tensorflow/quantization/ptq/newapi 2>&1 | tee -a ${ut_log_name}

mkdir -p report
mv *.html report
pytest_html_merger -i ./report -o ./report.html

cp report.html ${LOG_DIR}/

if [ $(grep -c '== FAILURES ==' ${ut_log_name}) != 0 ] || [ $(grep -c '== ERRORS ==' ${ut_log_name}) != 0 ] || [ $(grep -c ' passed' ${ut_log_name}) == 0 ]; then
    echo "Find errors in pytest case, please check the output..."
    echo "Please search for '== FAILURES ==' or '== ERRORS =='"
    exit 1
fi

echo "UT finished successfully! "