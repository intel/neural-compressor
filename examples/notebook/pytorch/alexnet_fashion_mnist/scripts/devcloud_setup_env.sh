#!/bin/bash

source /opt/intel/oneapi/setvars.sh
ENV_NAME=env_inc
conda deactivate
conda env remove -n $ENV_NAME 
conda create -n $ENV_NAME python=3.9 pip pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 cpuonly neural-compressor=2.0 matplotlib -y -c conda-forge -c pytorch -c intel
