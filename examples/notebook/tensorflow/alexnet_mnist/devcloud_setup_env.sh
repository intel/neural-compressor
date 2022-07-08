#!/bin/bash

ENV_NAME=env_inc
conda deactivate
conda env remove -n $ENV_NAME
conda create -n $ENV_NAME -c ${ONEAPI_ROOT}/conda_channel python=`python -V| awk '{print $2}'` -y
conda activate $ENV_NAME
conda install -n $ENV_NAME -c ${ONEAPI_ROOT}/conda_channel tensorflow python-flatbuffers -y
conda install -n $ENV_NAME -c ${ONEAPI_ROOT}/conda_channel neural-compressor -y --offline
conda install -n $ENV_NAME runipy notebook -y
