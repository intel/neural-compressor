pushd .
echo "Install loadgen"
git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
cd mlperf_inference
git checkout r2.1
git log -1
git submodule update --init --recursive
cd loadgen
CFLAGS="-std=c++14" python setup.py install
popd
