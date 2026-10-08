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

# DeepSeek uses MLA, which needs the MLA flashinfer backend.
export ATTENTION_BACKEND_FP8KV="FLASHINFER_MLA"
export ATTENTION_BACKEND_FP8ATTN="FLASHINFER_MLA"
export FLASHINFER_WORKSPACE_SIZE=1073741824
#FIXME: (yiliu30) remove this env once we have fixed the pynccl issues
export NCCL_NVLS_ENABLE=0
for arg in "$@"; do
    if [[ "$arg" == "--scheme=mxfp8" ]]; then
        export VLLM_DISABLE_PYNCCL=1
    fi
done

bash "${REPO_ROOT}/benchmark/lm_eval/run_lm_eval.sh" "$@"
