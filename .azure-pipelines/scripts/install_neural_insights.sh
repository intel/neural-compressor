#!/bin/bash

echo "Install Neural Insights ... "
cd /neural-compressor
python -m pip install --no-cache-dir -r neural_insights/requirements.txt
python setup.py neural_insights bdist_wheel
pip install dist/neural_insights*.whl
pip list