#!/bin/bash

cd /neural_compressor
python -m pip install --no-cache-dir -r requirements.txt
python setup.py sdist bdist_wheel
pip install dist/neural_compressor-*.whl
pip list