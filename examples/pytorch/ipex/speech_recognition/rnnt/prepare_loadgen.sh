WORKDIR=$1
pushd .
cd $WORKDIR
echo Current directory is $PWD
echo Using gcc=`which gcc`
echo GCC version should >= 9
gcc --version
CC=`which gcc`

# install pytorch
echo "Install pytorch/ipex"
export LD_LIBRARY_PATH=$WORKDIR/local/lib:$LD_LIBRARY_PATH
CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

cd $WORKDIR
echo "Install loadgen"
git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
cd mlperf_inference
git checkout r1.1
git log -1
git submodule update --init --recursive
cd loadgen
CFLAGS="-std=c++14" python setup.py install

popd
