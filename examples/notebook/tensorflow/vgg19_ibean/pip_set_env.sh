#!/bin/bash

ENV_NAME=env_inc
deactivate
rm -rf $ENV_NAME
python -m venv $ENV_NAME
source $ENV_NAME/bin/activate
pip install --upgrade pip
pip install matplotlib 
pip install tensorflow==2.10.0 neural-compressor runipy notebook ipykernel
pip install tensorflow_hub tensorflow_datasets
python -m ipykernel install --user --name $ENV_NAME
