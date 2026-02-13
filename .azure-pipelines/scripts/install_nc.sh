#!/bin/bash

echo -e "##[group]Install Neural Compressor ... "
cd /neural-compressor
if [[ $1 = *"3x_pt"* ]]; then
    python -m pip install --no-cache-dir -r requirements_pt.txt
    if [[ $1 = *"3x_pt_fp8"* ]]; then
        pip uninstall neural_compressor_3x_pt -y || true
        python setup.py pt bdist_wheel
    else
        echo -e "\n Install torch CPU ... "
        pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cpu
        pip install auto-round-lib==0.10.2.1 # mapping torch and auto-round version
        python -m pip install --no-cache-dir -r requirements.txt
        python setup.py bdist_wheel
    fi
    pip install --no-deps dist/neural_compressor*.whl --force-reinstall
elif [[ $1 = *"3x_tf"* ]]; then
	pip install tensorflow==2.19.0
    python -m pip install --no-cache-dir -r requirements.txt
    python -m pip install --no-cache-dir -r requirements_tf.txt
    python setup.py bdist_wheel
    pip install dist/neural_compressor*.whl --force-reinstall
else
    python -m pip install --no-cache-dir -r requirements.txt
    python setup.py bdist_wheel
    pip install dist/neural_compressor*.whl --force-reinstall
fi

echo -e "\n pip list after install Neural Compressor ... "
echo "##[endgroup]"
pip list
