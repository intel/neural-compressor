pip install -r requirements.txt
pip install setuptools --upgrade
pip install packaging --upgrade
pip install -U "huggingface_hub[cli]"
git clone -b mxfp4 https://github.com/mengniwang95/vllm-fork.git
cd vllm-fork
VLLM_USE_PRECOMPILED=1 pip install . -vvv --no-build-isolation
cd ..
