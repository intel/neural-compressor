#!/bin/bash
set -e

uv pip install -U pip setuptools_rust setuptools_scm
uv pip install -U evalscope lm_eval[api] lm-eval["ruler"] transformers datasets
uv pip install git+https://github.com/intel/auto-round.git@main
uv pip install compressed-tensors --no-deps
VLLM_USE_PRECOMPILED=1 uv pip install git+https://github.com/xin3he/vllm.git@support_deepseekv4_mxfp --no-build-isolation
# reference: https://github.com/xin3he/vllm/blob/support_deepseekv4_mxfp/tools/install_deepgemm.sh 
uv pip install git+https://github.com/deepseek-ai/DeepGEMM.git@891d57b4db1071624b5c8fa0d1e51cb317fa709f --no-build-isolation