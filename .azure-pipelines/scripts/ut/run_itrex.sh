#!/bin/bash
set -xe
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
find . -name "test*.py" | grep -v "test_tf" | sed 's,\.\/,python ,g' | sed 's/$/ --verbose/' >run.sh

# run UT
$BOLD_YELLOW && echo "cat run.sh..." && $RESET
cat run.sh | tee ${ut_log_name}
$BOLD_YELLOW && echo "------UT start-------" && $RESET
bash -x run.sh 2>&1 | tee -a ${ut_log_name}
$BOLD_YELLOW && echo "------ UT end -------" && $RESET

if [ $(grep -c "FAILED" ${ut_log_name}) != 0 ] || [ $(grep -c "core dumped" ${ut_log_name}) != 0 ] || [ $(grep -c "ModuleNotFoundError:" ${ut_log_name}) != 0 ] || [ $(grep -c "OK" ${ut_log_name}) == 0 ];then
    echo "Find errors in UT test, please check the output..."
    exit 1
fi
echo "UT finished successfully! "