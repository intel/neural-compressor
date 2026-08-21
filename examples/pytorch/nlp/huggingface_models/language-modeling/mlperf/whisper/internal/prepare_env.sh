#!/bin/bash

echo "Preparing wheels for XPU MLPerf workflow container"

mkdir -p /workspace/dist

# Clone internal repos
mkdir -p third_party && pushd third_party
# rm -rf ipex-gpu-internal vllm-internal

# ipex
git clone https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-gpu -b releases/2.8.10+xpu_rc ipex-gpu-internal
# vllm
git clone https://github.com/intel-sandbox/vllm-xpu # -b release/2509/vllm-xpu-0.10.1.1 vllm-internal
popd

# vllm
pip uninstall -y vllm
cd /workspace/third_party/vllm-internal
git checkout 5774b0a1d
pip install -r requirements/xpu.txt
VLLM_TARGET_DEVICE=xpu python setup.py bdist_wheel
cp dist/*.whl /workspace/dis
cd ..
# rm -rf vllm-internal

# ipex internal
export BUILD_WITH_CPU=OFF
export BUILD_SEPARATE_OPS=ON
export USE_AOT_DEVLIST='bmg'
export TORCH_XPU_ARCH_LIST='bmg'
# Use all threads if there is enough memory
export MAX_JOBS=$(awk -v threads="$(nproc)" '/^Mem:/{mem=int($7/2); print(threads < mem ? threads : mem)}' <(free -g))

cd /workspace/third_party/ipex-gpu-internal
# Inside ipex-gpu source directory
pip uninstall -y intel-extension-for-pytorch
pip install -r requirements.txt
git submodule sync
git submodule update --init --recursive

python setup.py bdist_wheel
cp dist/*.whl /workspace/dist/
pip install dist/*.whl
cd ..
# rm -rf ipex-gpu-internal

pushd third_party
git clone https://github.com/intel/auto-round.git
cd autoround
# git checkout mlperf-awq
python setup.py install
popd

# Remove internal repos. Already built whls
pushd third_party
# rm -rf ipex-gpu-internal vllm-internal
popd

echo "Wheels prepared in dist/ directory"
