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

# Usage: CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh --model_path=<path_to_quantized_model> [--tasks=<tasks>]
#
# Tasks are split into two paths:
#   - aime26, gpqa_diamond and math_500: benchmark/evalscope/run_evalscope.sh
#   - all other tasks: benchmark/lm_eval/run_lm_eval.sh
# All other options are passed to both drivers.

set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

TASKS="hellaswag,piqa,mmlu,gsm8k"
EVALSCOPE_TASK_LIST="aime26,gpqa_diamond,math_500"
COMMON_ARGS=()
for arg in "$@"; do
    case $arg in
        --tasks=*)
            TASKS="${arg#*=}"
            ;;
        *)
            COMMON_ARGS+=("$arg")
            ;;
    esac
done

LM_EVAL_TASKS=""
EVALSCOPE_TASKS=""
IFS=',' read -ra task_array <<< "$TASKS"
for task in "${task_array[@]}"; do
    task="${task//[[:space:]]/}"
    [[ -z "$task" ]] && continue
    if [[ ",${EVALSCOPE_TASK_LIST}," == *",${task},"* ]]; then
        EVALSCOPE_TASKS="${EVALSCOPE_TASKS:+$EVALSCOPE_TASKS,}$task"
    else
        LM_EVAL_TASKS="${LM_EVAL_TASKS:+$LM_EVAL_TASKS,}$task"
    fi
done

if [[ -n "$LM_EVAL_TASKS" ]]; then
    bash "${REPO_ROOT}/benchmark/lm_eval/run_lm_eval.sh" "${COMMON_ARGS[@]}" --tasks="$LM_EVAL_TASKS"
fi
if [[ -n "$EVALSCOPE_TASKS" ]]; then
    bash "${REPO_ROOT}/benchmark/evalscope/run_evalscope.sh" "${COMMON_ARGS[@]}" --tasks="$EVALSCOPE_TASKS"
fi
