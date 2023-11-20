#!/bin/bash
source /neural-compressor/.azure-pipelines/scripts/change_color.sh
python -c "import neural_compressor as nc;print(nc.version.__version__)"
echo "run itrex ut..."

# prepare itrex
git clone https://github.com/intel/intel-extension-for-transformers.git /intel-extension-for-transformers
bash /intel-extension-for-transformers/.github/workflows/script/prepare_env.sh
bash /intel-extension-for-transformers/.github/workflows/script/install_binary.sh

# prepare test env
pip install -r /intel-extension-for-transformers/tests/requirements.txt
LOG_DIR=/neural-compressor/log_dir
mkdir -p ${LOG_DIR}
ut_log_name=${LOG_DIR}/ut_itrex.log

# run unit test
pip install pytest
pytest -v -s --log-cli-level=WARNING --junitxml=test-ITREX.xml /intel-extension-for-transformers/tests -k "not test_tf*" 2>&1 | tee -a ${ut_log_name}

xml_data=$(cat test-ITREX.xml)
errors=$(echo "$xml_data" | grep -o 'errors="[0-9]*"' | sed 's/errors="\([0-9]*\)"/\1/')
failures=$(echo "$xml_data" | grep -o 'failures="[0-9]*"' | sed 's/failures="\([0-9]*\)"/\1/')

echo "Errors: $errors"
echo "Failures: $failures"

if [ "$errors" -ne 0 ] || [ "$failures" -ne 0 ]; then
    echo "unit test failed!"
    exit 1
else
    echo "unittest succeed!"
fi


if [ $(grep -c "FAILED" ${ut_log_name}) != 0 ] || [ $(grep -c "core dumped" ${ut_log_name}) != 0 ] || [ $(grep -c "ModuleNotFoundError:" ${ut_log_name}) != 0 ] || [ $(grep -c "OK" ${ut_log_name}) == 0 ];then
    echo "Find errors in UT test, please check the output..."
    exit 1
fi
echo "UT finished successfully! "