#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
	cat <<'EOF'
Usage:
	bash run_swebenchpro.sh [OPTIONS]

Run mini-SWE-agent on SWE-bench Pro using an already-running vLLM server, then
evaluate the generated patches with the official local-Docker evaluator.

Options:
	--host HOST          vLLM host (default: 127.0.0.1)
	--port PORT          vLLM port (default: 8888)
	--served-name NAME   Served model ID (default: discover from /v1/models)
	--num-tasks N        Run the first N instances
	--slice START:END    Run an explicit dataset slice
	--workers N          Parallel agent workers (default: 2)
	--eval-workers N     Parallel evaluation workers (default: --workers)
	--step-limit N       Maximum model calls per instance (default: 250)
	--pull-timeout N     Docker image pull/start timeout in seconds (default: 600)
	--tag TAG            Run tag (default: UTC timestamp)
	--redo               Re-run existing generation and evaluation results
	--skip-eval          Generate patches without local evaluation
	--block-network      Disable container networking during local evaluation
	--keep-images        Keep SWE-bench Pro Docker images after the run
	-h, --help           Show this help message

Environment:
	AGENT_MAX_TOKENS     Maximum tokens per model response (default: 8192)
	VLLM_API_KEY         API key sent to vLLM (default: EMPTY)
	VLLM_NO_PROXY        Hosts that bypass HTTP proxies (default: vLLM host)
	VLLM_WAIT_TIMEOUT    Readiness timeout in seconds (default: 300)
	VLLM_PID_FILE        vLLM PID file (default: logs/vllm_<PORT>.pid)
	OPENAI_BASE_URL      Override the OpenAI-compatible API base URL
EOF
}

stop_vllm() {
	local pid_file="${VLLM_PID_FILE:-${LOG_DIR}/vllm_${VLLM_PORT}.pid}"
	stop_vllm_from_pid_file "${pid_file}"
}

remove_swebench_pro_images() {
	[[ "${KEEP_IMAGES}" == false && -f "${IMAGE_LIST}" ]] || return
	mapfile -t images < <(sort -u "${IMAGE_LIST}" | grep -E '^jefzda/sweap-images:' || true)
	[[ ${#images[@]} -gt 0 ]] || return
	log "Removing ${#images[@]} SWE-bench Pro images"
	docker image rm -f "${images[@]}" >/dev/null 2>&1 || \
		warn "Some SWE-bench Pro images could not be removed"
}

cleanup() {
	remove_swebench_pro_images
	stop_vllm
}

require_positive_integer() {
	local name="$1"
	local value="$2"
	[[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer: ${value}"
}

WORKERS=2
EVAL_WORKERS=""
STEP_LIMIT=250
PULL_TIMEOUT=600
NUM_TASKS=""
SLICE_ARG=""
SERVED_MODEL_NAME=""
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
REDO=false
SKIP_EVAL=false
BLOCK_NETWORK=false
KEEP_IMAGES=false

while [[ $# -gt 0 ]]; do
	case "$1" in
		--host)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			VLLM_HOST="$2"
			shift 2
			;;
		--port)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			VLLM_PORT="$2"
			shift 2
			;;
		--served-name)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			SERVED_MODEL_NAME="$2"
			shift 2
			;;
		--num-tasks)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			NUM_TASKS="$2"
			shift 2
			;;
		--slice)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			SLICE_ARG="$2"
			shift 2
			;;
		--workers)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			WORKERS="$2"
			shift 2
			;;
		--eval-workers)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			EVAL_WORKERS="$2"
			shift 2
			;;
		--step-limit)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			STEP_LIMIT="$2"
			shift 2
			;;
		--pull-timeout)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			PULL_TIMEOUT="$2"
			shift 2
			;;
		--tag)
			[[ $# -ge 2 ]] || die "$1 requires a value"
			RUN_TAG="$2"
			shift 2
			;;
		--redo)
			REDO=true
			shift
			;;
		--skip-eval)
			SKIP_EVAL=true
			shift
			;;
		--block-network)
			BLOCK_NETWORK=true
			shift
			;;
		--keep-images)
			KEEP_IMAGES=true
			shift
			;;
		-h | --help)
			usage
			exit 0
			;;
		*)
			die "Unknown argument: $1"
			;;
	esac
done

[[ -z "${NUM_TASKS}" || -z "${SLICE_ARG}" ]] || \
	die "--num-tasks and --slice cannot be used together"
require_positive_integer "--workers" "${WORKERS}"
require_positive_integer "--step-limit" "${STEP_LIMIT}"
require_positive_integer "--pull-timeout" "${PULL_TIMEOUT}"
[[ -z "${EVAL_WORKERS}" ]] || require_positive_integer "--eval-workers" "${EVAL_WORKERS}"
[[ -z "${NUM_TASKS}" ]] || require_positive_integer "--num-tasks" "${NUM_TASKS}"
[[ -z "${SLICE_ARG}" || "${SLICE_ARG}" =~ ^[0-9]*:[0-9]*$ ]] || \
	die "--slice must use START:END format: ${SLICE_ARG}"

init_benchmark_paths
init_vllm_endpoint
RUN_TAG="$(sanitize_run_tag "${RUN_TAG}")"
readonly SWEBENCH_PRO_DIR="${BENCHMARK_DIR}/SWE-bench_Pro-os"
readonly OUT_DIR="${AGENT_DIR}/results/swebench_pro_${RUN_TAG}"
readonly PREDS_JSON="${OUT_DIR}/preds.json"
readonly PATCHES_JSON="${OUT_DIR}/patches.json"
readonly INSTANCES_CSV="${OUT_DIR}/instances.csv"
readonly IMAGE_LIST="${OUT_DIR}/images.txt"
readonly EVAL_OUT="${OUT_DIR}/evaluation"
readonly RUNTIME_CONFIG="${OUT_DIR}/swebench_pro.yaml"
readonly LOG_FILE="${LOG_DIR}/swebench_pro_${RUN_TAG}.log"

require_file "${SWEBENCH_PRO_DIR}/swe_bench_pro_eval.py"
require_file "${AGENT_DIR}/pyproject.toml"
require_file "${AGENT_DIR}/src/minisweagent/run/extra/swebench.py"
require_command mini-extra
require_command python
require_command docker
mkdir -p "${OUT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1
trap cleanup EXIT

python -c 'import datasets, yaml' 2>/dev/null || \
	die "Generation dependencies are unavailable; run: bash setup_swebenchpro.sh"
if [[ "${SKIP_EVAL}" == false ]]; then
	python -c 'import docker, pandas' 2>/dev/null || \
		die "Local evaluation dependencies are unavailable; run: bash setup_swebenchpro.sh"
fi

wait_for_vllm "${VLLM_WAIT_TIMEOUT:-300}"
SERVED_MODEL_NAME="$(discover_vllm_model "${SERVED_MODEL_NAME}")"
EVAL_WORKERS="${EVAL_WORKERS:-${WORKERS}}"
readonly EVAL_PREFIX="$(sanitize_run_tag "${SERVED_MODEL_NAME}_step${STEP_LIMIT}_${RUN_TAG}")"

SLICE_SPEC="${SLICE_ARG}"
[[ -n "${SLICE_SPEC}" || -z "${NUM_TASKS}" ]] || SLICE_SPEC="0:${NUM_TASKS}"
SLICE_OPTIONS=()
[[ -z "${SLICE_SPEC}" ]] || SLICE_OPTIONS=(--slice "${SLICE_SPEC}")
REDO_OPTIONS=()
[[ "${REDO}" == false ]] || REDO_OPTIONS=(--redo-existing)

python - "${RUNTIME_CONFIG}" "${SERVED_MODEL_NAME}" "${OPENAI_BASE_URL}" \
	"${VLLM_API_KEY}" "${STEP_LIMIT}" "${PULL_TIMEOUT}" <<'PY'
import os
import sys

import yaml
from minisweagent.config import builtin_config_dir

destination, model, base_url, api_key, step_limit, pull_timeout = sys.argv[1:]
config = yaml.safe_load((builtin_config_dir / "extra" / "swebench.yaml").read_text())
config["agent"]["step_limit"] = int(step_limit)
config.setdefault("environment", {})["pull_timeout"] = int(pull_timeout)
config.setdefault("run", {})["env_startup_command"] = (
    "git clone https://github.com/{{ repo }}.git . && {{ before_repo_set_cmd }}"
)

old_submit = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached"
new_submit = (
    "cd /testbed && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && "
    "git diff -- . ':(exclude)test/**' ':(exclude)tests/**'"
)
template = config["agent"]["instance_template"]
if old_submit not in template:
    raise RuntimeError("Expected submission command not found in the mini-SWE-agent config")
config["agent"]["instance_template"] = template.replace(old_submit, new_submit)
config["model"] = {
    "model_name": model,
    "cost_tracking": "ignore_errors",
    "model_kwargs": {
        "api_base": base_url,
        "api_key": api_key,
        "drop_params": True,
        "temperature": 0.0,
        "max_tokens": int(os.getenv("AGENT_MAX_TOKENS", "8192")),
    },
}
with open(destination, "w") as file:
    yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
print(f"Runtime config: {destination}")
PY

log "Running SWE-bench Pro with model ${SERVED_MODEL_NAME}"
(
	cd "${AGENT_DIR}"
	mini-extra swebench \
		--subset ScaleAI/SWE-bench_Pro \
		--split test \
		--model "${SERVED_MODEL_NAME}" \
		--workers "${WORKERS}" \
		--output "${OUT_DIR}" \
		--config "${RUNTIME_CONFIG}" \
		"${SLICE_OPTIONS[@]}" \
		"${REDO_OPTIONS[@]}"
)

require_file "${PREDS_JSON}"
python - "${PREDS_JSON}" "${PATCHES_JSON}" "${EVAL_PREFIX}" <<'PY'
import json
import re
import sys

source, destination, prefix = sys.argv[1:]
with open(source) as file:
    predictions = json.load(file)

error_pattern = re.compile(
    r"(?i)(traceback|exception|calledprocesserror|no space left on device|"
    r"not a git repository|error response from daemon)"
)


def normalize_patch(text):
    text = (text or "").strip()
    fence = "`" * 3
    if text.startswith(fence) and text.endswith(fence):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
	if not text or "*** Begin Patch" in text:
        return ""
    git_index = text.find("diff --git ")
    if git_index >= 0:
        text = text[git_index:].lstrip()
        return text if "--- a/" in text and "+++ b/" in text else ""
	if error_pattern.search(text):
		return ""
    match = re.search(r"(?m)^--- [^\n]+\n\+\+\+ [^\n]+", text)
    return text[match.start():].lstrip() if match and "@@" in text[match.start():] else ""


def touched_files(patch):
    files = [match.group(2) for match in re.finditer(r"(?m)^diff --git a/(.*?) b/(.*?)$", patch)]
    return files or [match.group(1) for match in re.finditer(r"(?m)^\+\+\+ b/(.*?)$", patch)]


def is_test_file(path):
    parts = path.split("/")
    name = parts[-1]
    return (
        any(part in {"test", "tests"} for part in parts[:-1])
        or name.startswith("test_")
        or name.endswith(("_test.py", ".test.js", ".spec.js", ".test.ts", ".spec.ts", "_test.go"))
    )


patches = []
invalid = 0
test_changes = 0
items = predictions.items() if isinstance(predictions, dict) else enumerate(predictions)
for key, record in items:
    if not isinstance(record, dict):
        record = {"model_patch": str(record)}
    instance_id = record.get("instance_id") or (key if isinstance(key, str) else None)
    if not instance_id:
        continue
    patch = ""
    for field in ("model_patch", "patch", "prediction", "completion", "response", "output"):
        if isinstance(record.get(field), str) and (candidate := normalize_patch(record[field])):
            patch = candidate
            break
    if patch and any(is_test_file(path) for path in touched_files(patch)):
        patch = ""
        test_changes += 1
    if not patch:
        invalid += 1
    patches.append({"instance_id": instance_id, "patch": patch, "prefix": prefix})

with open(destination, "w") as file:
    json.dump(patches, file, indent=2)
print(f"Wrote {len(patches)} patches to {destination}")
print(f"Invalid or empty patches: {invalid}")
print(f"Patches touching test files: {test_changes}")
PY

python - "${INSTANCES_CSV}" "${IMAGE_LIST}" "${SLICE_SPEC}" <<'PY'
import sys

import pandas as pd
from datasets import load_dataset

csv_path, image_path, slice_spec = sys.argv[1:]
instances = list(load_dataset("ScaleAI/SWE-bench_Pro", split="test"))
if slice_spec:
    bounds = [int(value) if value else None for value in slice_spec.split(":")]
    instances = instances[slice(*bounds)]
if not instances:
	raise RuntimeError(f"Dataset slice selected no instances: {slice_spec or 'all'}")
pd.DataFrame(instances).to_csv(csv_path, index=False)
images = sorted(
    {f"jefzda/sweap-images:{item['dockerhub_tag']}" for item in instances if item.get("dockerhub_tag")}
)
with open(image_path, "w") as file:
    file.write("\n".join(images) + ("\n" if images else ""))
print(f"Wrote {len(instances)} instances to {csv_path}")
print(f"Tracked {len(images)} Docker images in {image_path}")
PY

if [[ "${SKIP_EVAL}" == true ]]; then
	log "Skipping local SWE-bench Pro evaluation"
	log "Patches: ${PATCHES_JSON}"
	log "Log: ${LOG_FILE}"
	exit 0
fi

EVAL_OPTIONS=(--use_local_docker)
[[ "${REDO}" == false ]] || EVAL_OPTIONS+=(--redo)
[[ "${BLOCK_NETWORK}" == false ]] || EVAL_OPTIONS+=(--block_network)

log "Evaluating patches with ${EVAL_WORKERS} workers"
(
	cd "${SWEBENCH_PRO_DIR}"
	python swe_bench_pro_eval.py \
		--raw_sample_path "${INSTANCES_CSV}" \
		--patch_path "${PATCHES_JSON}" \
		--output_dir "${EVAL_OUT}" \
		--scripts_dir run_scripts \
		--dockerhub_username jefzda \
		--num_workers "${EVAL_WORKERS}" \
		"${EVAL_OPTIONS[@]}"
)

readonly REPORT_FILE="${EVAL_OUT}/eval_results.json"
require_file "${REPORT_FILE}"
python - "${REPORT_FILE}" <<'PY'
import json
import sys

with open(sys.argv[1]) as file:
    results = json.load(file)
passed = sum(bool(value) for value in results.values())
total = len(results)
accuracy = passed / total * 100 if total else 0.0
print(f"Local accuracy: {passed}/{total} = {accuracy:.1f}%")
for instance_id, resolved in sorted(results.items(), key=lambda item: (not item[1], item[0])):
    print(f"  {'PASS' if resolved else 'FAIL'}  {instance_id}")
PY

log "Patches: ${PATCHES_JSON}"
log "Report: ${REPORT_FILE}"
log "Log: ${LOG_FILE}"
