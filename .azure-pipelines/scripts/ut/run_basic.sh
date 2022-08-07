#!/bin/bash
set -x
python -c "import neural_compressor as nc;print(nc.version.__version__)"
echo "run basic"