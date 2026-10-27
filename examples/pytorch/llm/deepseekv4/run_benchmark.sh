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

# Usage: CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh --model_path=<path_to_quantized_model> [--tasks=<tasks>]
#
# Tasks are split into two paths:
#   - RULER (*ruler*, niah_*) and LongBench (*longbench*) tasks: benchmark/lm_eval/run_lm_eval.sh
#   - all other tasks: benchmark/evalscope/run_evalscope.sh
# --batch_size, --gpu_memory_utilization and --ruler_max_pos only go to lm_eval.

set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

TASKS="aime26,gpqa_diamond,piqa,hellaswag,gsm8k,mmlu_pro,math_500,mmlu"
MODEL_NAME=""
LM_EVAL_ARGS=()
EVALSCOPE_ARGS=()
for arg in "$@"; do
    case $arg in
        --tasks=*)
            TASKS="${arg#*=}"
            ;;
        --batch_size=*|--gpu_memory_utilization=*|--ruler_max_pos=*)
            LM_EVAL_ARGS+=("$arg")
            ;;
        --model_path=*|--static_kv_dtype=*|--static_attention_dtype=*)
            [[ "$arg" == --model_path=* ]] && MODEL_NAME=$(basename "${arg#*=}")
            LM_EVAL_ARGS+=("$arg")
            EVALSCOPE_ARGS+=("$arg")
            ;;
        *)
            EVALSCOPE_ARGS+=("$arg")
            ;;
    esac
done

LM_EVAL_TASKS=""
EVALSCOPE_TASKS=""
IFS=',' read -ra task_array <<< "$TASKS"
for task in "${task_array[@]}"; do
    if [[ "$task" == *ruler* || "$task" == niah_* || "$task" == *longbench* ]]; then
        LM_EVAL_TASKS="${LM_EVAL_TASKS:+$LM_EVAL_TASKS,}$task"
    else
        EVALSCOPE_TASKS="${EVALSCOPE_TASKS:+$EVALSCOPE_TASKS,}$task"
    fi
done

export SAFETENSORS_FAST_GPU=1
export EXTRA_SERVE_ARGS="--trust-remote-code --kv-cache-dtype fp8 --block-size 256 --attention_config.use_fp4_indexer_cache=True --no-enable-flashinfer-autotune"
# Only the original (non-quantized) checkpoints use expert parallel with the DeepGEMM mega MoE backend.
if [[ "$MODEL_NAME" == "DeepSeek-V4-Flash" || "$MODEL_NAME" == "DeepSeek-V4-Pro" ]]; then
    EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS} --enable-expert-parallel --moe-backend deep_gemm_mega_moe"
fi
export EXTRA_LM_EVAL_MODEL_ARGS=",trust_remote_code=True"

# Override the MiniMax-oriented defaults of the evalscope driver; empty values disable the option.
export DEFAULT_NUM_RUNS=1
export DEFAULT_TEMPERATURE=0
export DEFAULT_TOP_P=""
export DEFAULT_MAX_TOKENS=""
export DEFAULT_MAX_MODEL_LEN=""
export DEFAULT_SERVED_MODEL_NAME=""
if [[ "$MODEL_NAME" == *"DeepSeek-V4-Pro"* ]]; then
    export DEFAULT_MAX_MODEL_LEN=1048576
    export DEFAULT_EVAL_BATCH_SIZE=10
fi

if [[ -n "$EVALSCOPE_TASKS" ]]; then
    bash "${REPO_ROOT}/benchmark/evalscope/run_evalscope.sh" "${EVALSCOPE_ARGS[@]}" --tasks="$EVALSCOPE_TASKS"
fi
if [[ -n "$LM_EVAL_TASKS" ]]; then
    bash "${REPO_ROOT}/benchmark/lm_eval/run_lm_eval.sh" "${LM_EVAL_ARGS[@]}" --tasks="$LM_EVAL_TASKS"
fi
