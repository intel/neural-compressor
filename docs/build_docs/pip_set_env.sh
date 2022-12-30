#!/bin/bash

ENV_NAME=env_sphinx
deactivate
rm -rf $ENV_NAME
python -m venv $ENV_NAME
source $ENV_NAME/bin/activate
pip install --upgrade pip
pip install -r sphinx-requirements.txt

