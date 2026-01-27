pip install -r requirements.txt
pip install setuptools --upgrade
pip install packaging --upgrade
pip install transformers==4.57.3
pip install -U "huggingface_hub[cli]"
# Intsall auto-round
git clone -b fix-ds-attn https://github.com/intel/auto-round.git && cd auto-round
pip install -e .
cd ..
# Install vllm
git clone -b fused-moe-ar --single-branch --quiet https://github.com/yiliu30/vllm-fork.git && cd vllm-fork
VLLM_USE_PRECOMPILED=1 pip install . -v
cd ..
# Uninstall flash_attn to avoid conflicts
pip uninstall flash_attn -y