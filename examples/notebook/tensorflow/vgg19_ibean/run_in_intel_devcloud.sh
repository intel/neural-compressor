#!/bin/bash

echo "Enable Conda Env."
source /glob/development-tools/versions/oneapi/2022.1.1/oneapi/intelpython/python3.9/etc/profile.d/conda.sh
conda activate env_inc
bash run_sample.sh
