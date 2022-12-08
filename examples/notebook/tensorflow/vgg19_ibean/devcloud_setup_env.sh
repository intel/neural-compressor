#!/bin/bash

ENV_NAME=env_inc
conda deactivate
conda env remove -n $ENV_NAME
conda create -n $ENV_NAME -c ${ONEAPI_ROOT}/conda_channel python=`python -V| awk '{print $2}'` -y
conda activate $ENV_NAME
conda install -n $ENV_NAME tensorflow python-flatbuffers -c intel -y
conda install -n $ENV_NAME neural-compressor runipy notebook ipykernel matplotlib -c conda-forge  -c intel -y
conda install -n $ENV_NAME tensorflow-hub tensorflow-datasets -y
python -m ipykernel install --user --name $ENV_NAME

