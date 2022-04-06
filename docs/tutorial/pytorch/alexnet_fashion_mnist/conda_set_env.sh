#!/bin/bash

ENV_NAME=env_inc
conda deactivate
conda env remove -n $ENV_NAME
conda create -n $ENV_NAME python=3.9 -c intel -y
conda activate $ENV_NAME
conda install -n $ENV_NAME pytorch  -c intel -y
conda install -n $ENV_NAME neural-compressor runipy notebook ipykernel -c conda-forge  -c intel -y
python -m ipykernel install --user --name $ENV_NAME
