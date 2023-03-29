#!/bin/bash

ENV_NAME=pip_env_inc
deactivate
rm -rf $ENV_NAME
python3 -m venv $ENV_NAME
source $ENV_NAME/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt


