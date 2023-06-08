#!/bin/bash

echo "Install Open MPI ..."
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz
gunzip -c openmpi-4.1.5.tar.gz | tar xf -
cd openmpi-4.1.5
./configure --prefix=/usr/local
make all install
MPI_DIR=/usr/local/lib/openmpi
export LD_LIBRARY_PATH=$MPI_DIR/lib:$LD_LIBRARY_PATH

echo "Install Neural Solution ... "
cd /neural-compressor
python -m pip install --no-cache-dir -r neural_solution/requirements.txt
python setup.py neural_solution sdist bdist_wheel
pip install dist/neural_solution*.whl
pip list