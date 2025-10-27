#!/bin/bash

echo "Enable Conda Env."
source ${ONEAPI_ROOT}/intelpython/python3.9/etc/profile.d/conda.sh
conda activate env_inc
export PYTHONPATH=$(find $CONDA_PREFIX -type d -name "site-packages" | head -n 1)
bash run_sample.sh
