#!/bin/bash

ENV_NAME=env_inc
conda deactivate
conda env remove -n $ENV_NAME
conda create -n $ENV_NAME -c ${ONEAPI_ROOT}/conda_channel python=`python -V| awk '{print $2}'` -y
conda activate $ENV_NAME
pip install tensorflow neural-compressor runipy notebook ipykernel matplotlib
pip install tensorflow-hub tensorflow-datasets
python -m ipykernel install --user --name $ENV_NAME

