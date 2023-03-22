#!/bin/bash

echo "Enable Conda Env."
source ${ONEAPI_ROOT}/intelpython/python3.9/etc/profile.d/conda.sh
conda activate env_inc
bash run_sample.sh
