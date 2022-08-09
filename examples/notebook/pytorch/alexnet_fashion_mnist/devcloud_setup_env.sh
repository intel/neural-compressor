#!/bin/bash

ENV_NAME=env_inc
conda deactivate
conda env remove -n $ENV_NAME
conda create -n $ENV_NAME -c ${ONEAPI_ROOT}/conda_channel python=`python -V| awk '{print $2}'` -y
conda activate $ENV_NAME
conda install -n $ENV_NAME -c ${ONEAPI_ROOT}/conda_channel pytorch torchvision torchaudio cpuonly -c intel -c pytorch -y
conda install -n $ENV_NAME -c ${ONEAPI_ROOT}/conda_channel neural-compressor -y --offline
conda install -c anaconda scipy
conda install -n $ENV_NAME runipy notebook -y
