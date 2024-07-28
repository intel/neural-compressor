#!/bin/bash

echo -e "\n Install Neural Compressor ... "
cd /neural-compressor
if [[ $1 = *"3x_pt"* ]]; then
    if [[ $1 != *"3x_pt_fp8"* ]]; then
        echo -e "\n Install torch CPU ... "
        pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
    fi
    python -m pip install --no-cache-dir -r requirements_pt.txt
    python setup.py pt bdist_wheel
    pip install --no-deps dist/neural_compressor*.whl --force-reinstall
elif [[ $1 = *"3x_tf"* ]]; then
    python -m pip install --no-cache-dir -r requirements_tf.txt
    python setup.py tf bdist_wheel
    pip install dist/neural_compressor*.whl --force-reinstall
else
    python -m pip install --no-cache-dir -r requirements.txt
    python setup.py bdist_wheel
    pip install dist/neural_compressor*.whl --force-reinstall
fi

echo -e "\n pip list after install Neural Compressor ... "
pip list
