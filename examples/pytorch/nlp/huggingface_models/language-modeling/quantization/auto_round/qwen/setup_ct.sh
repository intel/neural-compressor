pip install -r requirements.txt
pip install setuptools --upgrade
pip install packaging --upgrade
pip install -U "huggingface_hub[cli]"

# Install vllm
VLLM_TAGS="tags/v0.19.1"
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout $VLLM_TAGS
VLLM_USE_PRECOMPILED=1 pip install . -v
cd ..
# Install vllm-qdq-plugin
git clone https://github.com/yiliu30/vllm-qdq-plugin.git
pip install -e vllm-qdq-plugin/
cd ..
# Uninstall flash_attn to avoid conflicts
pip uninstall flash_attn -y
pip install lm_eval["ruler"]