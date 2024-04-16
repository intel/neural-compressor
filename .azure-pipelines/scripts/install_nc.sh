#!/bin/bash

echo -e "\n Install Neural Compressor ... "
cd /neural-compressor

python -m pip install --no-cache-dir -r requirements.txt
python setup.py bdist_wheel
pip install dist/neural_compressor*.whl --force-reinstall

echo -e "\n pip list after install Neural Compressor ... "
pip list
