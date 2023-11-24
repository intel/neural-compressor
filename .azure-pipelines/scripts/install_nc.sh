#!/bin/bash

echo -e "\n Install Neural Compressor ... "
cd /neural-compressor
if [ $1 ]; then
    if [[ $1 = *"3x_pt" ]]; then
        python -m pip install --no-cache-dir -r requirements_pt.txt
        python setup.py pt bdist_wheel
        pip install dist/neural_compressor*.whl
    elif [[ $1 = *"3x_tf" ]]; then
        python -m pip install --no-cache-dir -r requirements_tf.txt
        python setup.py tf bdist_wheel
        pip install dist/neural_compressor*.whl
    fi
else
    python -m pip install --no-cache-dir -r requirements.txt
    python setup.py 2x bdist_wheel
    pip install dist/neural_compressor*.whl
fi

echo -e "\n pip list after install Neural Compressor ... "
pip list
