#!/bin/bash
set -x
echo -e "##[group]Install Neural Compressor ... "
cd /neural-compressor

if ! command -v uv &> /dev/null; then
    pip install uv
    export UV_SYSTEM_PYTHON=1
fi

if [[ $1 = *"3x_pt"* ]]; then
    uv pip install --no-cache-dir -r requirements_pt.txt
    if [[ $1 = *"hpu"* ]]; then
        uv pip uninstall neural_compressor_3x_pt -y || true
    elif [[ $1 = *"xpu"* ]]; then
        echo -e "\n Install torch XPU ... "
        uv pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/xpu
        uv pip install torch==2.11.0 auto-round-lib # mapping torch and auto-round version
    else
        echo -e "\n Install torch CPU ... "
        uv pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cpu
        uv pip install torch==2.11.0 auto-round-lib # mapping torch and auto-round version
    fi
    python setup.py pt bdist_wheel
    uv pip install --no-deps dist/neural_compressor*.whl --force-reinstall
elif [[ $1 = *"3x_tf"* ]]; then
	uv pip install tensorflow==2.19.0
    uv pip install --no-cache-dir -r requirements_tf.txt
    python setup.py tf bdist_wheel
    uv pip install dist/neural_compressor*.whl --force-reinstall
elif [[ $1 = *"3x_jax"* ]]; then
    uv pip install --no-cache-dir -r requirements_jax.txt
    python setup.py jax bdist_wheel
    uv pip install dist/neural_compressor*.whl --force-reinstall
else
    python setup.py bdist_wheel
    uv pip install dist/neural_compressor*.whl --force-reinstall
fi

echo -e "\n pip list after install Neural Compressor ... "
uv pip list

echo "##[endgroup]"
