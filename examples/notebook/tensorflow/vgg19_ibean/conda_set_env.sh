#!/bin/bash

ENV_NAME=env_inc
conda deactivate
conda env remove -n $ENV_NAME
conda create -n $ENV_NAME python=3.9 -y
conda activate $ENV_NAME
conda install pip -y
pip install --upgrade pip
pip install matplotlib
pip install tensorflow==2.10.0 neural-compressor runipy notebook ipykernel
pip install tensorflow_hub tensorflow_datasets
python -m ipykernel install --user --name $ENV_NAME
