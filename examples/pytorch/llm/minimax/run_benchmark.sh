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

# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh --model_path=<path_to_quantized_model> [--tasks=<tasks>]
#
# Tasks are split into two paths:
#   - RULER (*ruler*, niah_*) tasks: benchmark/lm_eval/run_lm_eval.sh
#   - all other tasks: benchmark/evalscope/run_evalscope.sh
# --gpu_memory_utilization and --ruler_max_pos only go to lm_eval.

set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

export DEFAULT_TASKS="gpqa_diamond,aime25,gsm8k,piqa"
export DEFAULT_NUM_RUNS=5
export DEFAULT_SERVED_MODEL_NAME="minimax-m2.7"
export DEFAULT_MAX_MODEL_LEN=102400
export DEFAULT_EVAL_BATCH_SIZE=256
export EVAL_TIMEOUT=3600

# Thinking mode with reasoning_effort=max.
export DEFAULT_TEMPERATURE=1.0
export DEFAULT_TOP_P=0.95
export DEFAULT_MAX_TOKENS=64000
export EXTRA_GENERATION_CONFIG='{"extra_body": {"chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "max"}}}'

export EXTRA_SERVE_ARGS="--trust-remote-code --tool-call-parser minimax_m2 --enable-auto-tool-choice --reasoning-parser minimax_m2 --max-num-seqs 1024 --max-num-batched-tokens 32768 --enable-chunked-prefill"
export ATTENTION_BACKEND_FP8KV="FLASHINFER"
export ATTENTION_BACKEND_FP8ATTN="FLASHINFER"

TASKS="$DEFAULT_TASKS"
LM_EVAL_ARGS=()
EVALSCOPE_ARGS=()
for arg in "$@"; do
    case $arg in
        --tasks=*)
            TASKS="${arg#*=}"
            ;;
        --gpu_memory_utilization=*|--ruler_max_pos=*)
            LM_EVAL_ARGS+=("$arg")
            ;;
        --model_path=*|--static_kv_dtype=*|--static_attention_dtype=*)
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
    if [[ "$task" == *ruler* || "$task" == niah_* ]]; then
        LM_EVAL_TASKS="${LM_EVAL_TASKS:+$LM_EVAL_TASKS,}$task"
    else
        EVALSCOPE_TASKS="${EVALSCOPE_TASKS:+$EVALSCOPE_TASKS,}$task"
    fi
done

if [[ -n "$EVALSCOPE_TASKS" ]]; then
    bash "${REPO_ROOT}/benchmark/evalscope/run_evalscope.sh" "${EVALSCOPE_ARGS[@]}" --tasks="$EVALSCOPE_TASKS"
fi
if [[ -n "$LM_EVAL_TASKS" ]]; then
    bash "${REPO_ROOT}/benchmark/lm_eval/run_lm_eval.sh" "${LM_EVAL_ARGS[@]}" --tasks="$LM_EVAL_TASKS"
fi
