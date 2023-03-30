#!/bin/bash

echo "Enable Conda Env."
source /opt/intel/oneapi/setvars.sh
conda activate env_inc
export PYTHONPATH=$(find $CONDA_PREFIX -type d -name "site-packages" | head -n 1)
cd scripts
bash run_sample.sh
