#!/bin/bash

source ${ONEAPI_ROOT}/intelpython/python3.9/etc/profile.d/conda.sh
ENV_NAME=env_inc
conda deactivate
conda env remove -n $ENV_NAME -y 
conda create -n $ENV_NAME python=3.9 -y
conda activate $ENV_NAME
which python3.9
python3.9 -m pip install -r requirements.txt
