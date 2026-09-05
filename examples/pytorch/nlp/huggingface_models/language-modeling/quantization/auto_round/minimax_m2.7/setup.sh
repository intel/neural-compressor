#!/bin/bash
set -e

uv pip install -U pip setuptools_rust setuptools_scm
uv pip install -U evalscope lm_eval[api] transformers datasets
uv pip install git+https://github.com/intel/auto-round.git@main
uv pip install compressed-tensors --no-deps
uv pip install vllm
