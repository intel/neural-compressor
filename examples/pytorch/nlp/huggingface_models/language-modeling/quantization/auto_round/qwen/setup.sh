pip install -r requirements.txt
pip install setuptools --upgrade
pip install packaging --upgrade
pip install -U "huggingface_hub[cli]"
# Intall vllm
git clone -b ds-fp8kv  --single-branch --quiet https://github.com/yiliu30/vllm-fork.git && cd vllm-fork
VLLM_USE_PRECOMPILED=1 pip install --editable . -v
cd ..
# Install auto-round
git clone -b ds-fp8kv --single-branch --quiet https://github.com/intel/auto-round.git && cd auto-round
pip install -e . -v
cd ..
# Uninstall flash_attn to avoid conflicts
pip uninstall flash_attn -y