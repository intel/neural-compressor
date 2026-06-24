pip install -r requirements.txt

mkdir -p /workspace/build_envs

cd /workspace/build_envs

pip install setuptools --upgrade
pip install packaging --upgrade
pip install -U "huggingface_hub[cli]"
git clone -b mxfp4 https://github.com/mengniwang95/vllm-fork.git
cd vllm-fork
VLLM_USE_PRECOMPILED=1 pip install . -vvv --no-build-isolation
cd ..

git clone https://github.com/mlcommons/inference.git mlperf-inference && cd mlperf-inference && cd loadgen && \
python3 -m pip install .

pip install auto-round==0.9.0
