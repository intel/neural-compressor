#!/bin/bash
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh --model_path=<path_to_quantized_model> --scheme=mxfp4

set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

export DEFAULT_SCHEME="mxfp8"

export ATTENTION_BACKEND_FP8KV="FLASHINFER"
# fp8 attention is only supported through the TRITON_ATTN backend for LLMC format
export ATTENTION_BACKEND_FP8ATTN="TRITON_ATTN"
export FLASHINFER_WORKSPACE_SIZE=2147483648
export ROPE_SCALING_JSON='{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
export SKIP_ROPE_SCALING_PATTERN="Qwen3-235B-A22B"
# B200 special env
export NLTK_ALLOW_PROXIED_URLOPEN=1

bash "${REPO_ROOT}/benchmark/lm_eval/run_lm_eval.sh" "$@"
