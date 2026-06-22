#!/bin/bash
set -e

uv pip install -U pip setuptools_rust setuptools_scm
uv pip install -U evalscope lm_eval transformers datasets
uv pip install compressed-tensors --no-deps
bash <(curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/main/tools/install_deepgemm.sh)
VLLM_USE_PRECOMPILED=1 uv pip install git+https://github.com/xin3he/vllm-fork.git@support_deepseekv4_mxfp --no-build-isolation