# Agent Benchmarks

Scripts for serving models with vLLM and running agent benchmarks against its OpenAI-compatible API.

## vLLM environment setup

Run the setup script inside an existing uv, Conda, or Docker environment that provides Python and supports `uv pip install`. The script does not create or activate an environment.

```bash
bash setup_vllm.sh
```

The script installs the pinned standard vLLM release from PyPI.

Values in `versions.env` are defaults. Set the corresponding environment
variable before running a setup or benchmark script to override a version,
commit, or image for that invocation.

## Start the vLLM server

```bash
CUDA_VISIBLE_DEVICES=0 bash start_vllm_serve.sh MODEL [VLLM_OPTIONS...]
```

The command returns after starting the server with `nohup` in a detached
background process group. Arguments after `MODEL` are passed directly to
`vllm serve`. Output is retained under `logs/vllm_<MODEL>_<TIMESTAMP>.log`, with
the process ID saved next to it and in `logs/vllm_<PORT>.pid`. Set `VLLM_LOG_DIR`
to use another log directory, `VLLM_LOG_FILE` to specify an exact log path, or
`VLLM_PID_FILE` to override the active-server PID file.

Common settings:

- Port: `8888`, unless the user passes `--port`
- Served model name: `gpt-3.5-turbo`
- `--trust-remote-code`
- `--enable-auto-tool-choice` for every model

For example:

`--tool-call-parser` must match the model's tool-call format so vLLM can return structured `tool_calls`.
A wrong parser may cause `RepeatedFormatError`.
The default is `hermes`. When the model name or path contains `Qwen3.6`, the
script automatically adds `--tool-call-parser qwen3_coder` and
`--reasoning-parser qwen3` unless those options were explicitly supplied:

```bash
CUDA_VISIBLE_DEVICES=0 bash start_vllm_serve.sh \
    Qwen3/Qwen3.6-27B \
    --tensor-parallel-size 1 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3
```

The script prints the generated log and PID-file paths. To follow the server
output, run `tail -f` on the printed log path. Benchmark runners leave the
shared vLLM server running whether they succeed or fail.

Stop the server explicitly when it is no longer needed:

```bash
bash start_vllm_serve.sh --stop --port 8888
```

The stop command uses `logs/vllm_<PORT>.pid` by default. Set the same
`VLLM_LOG_DIR` used to start the server, or set `VLLM_PID_FILE` to the exact PID
file, when using a custom location.

## Terminal and Multimodal Benchmarks

`run_terminal_bench.sh` runs Terminal-Bench 2.0 and 2.1 with Harbor and
Terminus-2. `run_multimodal_bench.sh` runs MMMU, MMMU-Pro, SimpleVQA, and
OmniDocBench 1.5 with lmms-eval. Both runners connect to an existing
OpenAI-compatible vLLM endpoint. Use separate environments because Harbor and
lmms-eval have different dependencies.

### Set up the environments

Create and activate a Terminal-Bench environment, then install the pinned
Harbor release:

```bash
conda create -n terminal-bench python=3.12 pip
conda activate terminal-bench
bash setup_terminal_bench.sh
```

Use another environment for multimodal evaluation:

```bash
conda create -n multimodal-bench python=3.12 pip
conda activate multimodal-bench
bash setup_multimodal_bench.sh
```

The multimodal setup clones the lmms-eval revision pinned in `versions.env`,
applies the OpenAI API compatibility patch, and installs the checkout. The
patch forwards vLLM sampling extensions such as `top_k`, prevents an HF token
from being printed in logs, and selects the Qwen prompt for MMMU and MMMU-Pro
when the `openai` backend is used. It also adds the Mini tasks backed by the
`jia0160/multimodal-benchmarks-mini` dataset at the revision pinned in
`versions.env`.

### Start the model server

Start one shared multimodal-capable server. Do not pass
`--language-model-only`, because that disables the vision encoder.

```bash
MODEL_PATH=/path/to/model
SERVED_MODEL_NAME=model-name
CUDA_VISIBLE_DEVICES=0,1 bash start_vllm_serve.sh \
  "${MODEL_PATH}" \
  --port 8002 \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 1 \
  --enable-prefix-caching
```

The same server can handle Terminal-Bench. A text-only server started with
`--language-model-only` can be used when running only Terminal-Bench.

### Run smoke and full evaluations

Activate the corresponding environment before invoking its runner. Alternatively,
pass a Conda prefix through `--env-prefix`. `--mini` selects one Terminal-Bench
task or the fixed 90-sample Mini dataset for each multimodal benchmark.

```bash
bash run_terminal_bench.sh \
  --benchmark terminal-bench-2.1 \
  --port 8002 \
  --mini

bash run_multimodal_bench.sh \
  --benchmark mmmu \
  --port 8002 \
  --mini
```

The multimodal Mini dataset is selected offline rather than taking the first N
rows at runtime. Its `mmmu`, `mmmu_pro`, `simplevqa`, and `omnidocbench`
configurations each contain 90 samples. MMMU and MMMU-Pro use three samples
from each of 30 subjects while covering question and image characteristics.
SimpleVQA balances nine tasks, nine topics, and Chinese/English samples.
OmniDocBench balances nine data sources and covers language, layout, table, and
formula characteristics. The dataset repository and immutable revision are
recorded in `versions.env` and the patched lmms-eval task definitions.

Each runner accepts `all` for the benchmarks it owns:

```bash
bash run_terminal_bench.sh \
  --benchmark all \
  --port 8002 \
  --workers 1 \
  --retry-attempts 3

bash run_multimodal_bench.sh \
  --benchmark all \
  --port 8002 \
  --workers 1 \
  --retry-attempts 3

bash run_multimodal_bench.sh \
  --benchmark all \
  --port 8002 \
  --mini
```

Terminal-Bench uses `terminal-bench@2.0` and the fixed
`terminal-bench/terminal-bench-2-1@6` revision, with five trials per task by
default. MMMU, MMMU-Pro, and OmniDocBench use thinking mode with 32,768 output
tokens and `temperature=1.0`, `top_p=0.95`, `top_k=20`, and
`presence_penalty=1.5`. SimpleVQA uses its deterministic task defaults with
thinking disabled. Results are written under `outputs/terminal-bench/` and
`outputs/multimodal-bench/`.

Create a normalized summary from completed Harbor and lmms-eval result files:

```bash
python lib/benchmark_data.py benchmark-report \
  --terminal-result outputs/terminal-bench/JOB/result.json \
  --lmms-result outputs/multimodal-bench/MODEL/TIMESTAMP_results.json \
  --model MODEL_NAME \
  --output outputs/benchmark-report.json
```

Pass each option more than once to combine multiple runs. The report records
the benchmark, model, primary metric as a percentage, sample count, failed
sample count when available, and source result path.

### Resume failed or interrupted runs

Harbor persists each Terminal-Bench job under its jobs directory. Resume an
interrupted job without rerunning completed trials by passing the job directory
that contains `config.json`:

```bash
bash run_terminal_bench.sh \
  --resume-job outputs/terminal-bench/terminal-bench-2.1-RUN_TIMESTAMP \
  --port 8002
```

Harbor removes trials with `CancelledError` before resuming by default. Pass
`--retry-error-type TYPE` one or more times to remove and retry completed trials
with specific exception types. For a new run, `--max-retries N` retries each
trial when Harbor encounters an exception; `--retry-attempts N` remains a
process-level retry for runner failures. `--attempts` controls independent
trials per task for pass-rate estimation and is not a failure retry count.

For multimodal evaluation, assign a stable run ID from the first invocation and
reuse it with `--resume`. A completion marker is written only after an entire
benchmark succeeds, so a resumed `all` run skips completed benchmarks and reruns
only the benchmark that was interrupted:

```bash
bash run_multimodal_bench.sh \
  --benchmark all \
  --run-id qwen-eval-1 \
  --port 8002

bash run_multimodal_bench.sh \
  --benchmark all \
  --run-id qwen-eval-1 \
  --resume \
  --port 8002
```

Runs with an ID store results, completion markers, and the default lmms-eval
response cache under `outputs/multimodal-bench/RUN_ID/`. The cache reuses only
successful deterministic responses, so it provides sample-level recovery for
SimpleVQA. MMMU, MMMU-Pro, and OmniDocBench use sampling and therefore rerun the
current benchmark after interruption; lmms-eval intentionally does not cache
those responses. The completion marker also records the model, endpoint, and
run options, and resume fails instead of skipping when they differ.

### Clean up artifacts

Terminal-Bench passes Harbor's `--delete` option, so trial containers are
removed after completion while pulled or built Docker images remain cached for
later runs. Harbor job directories contain the configuration, trajectories,
and result files required by `--resume-job`; remove a job directory only after
it no longer needs to be resumed. Docker images are shared host resources and
are not deleted automatically. Inspect them with `docker image ls` and remove
only confirmed unused images according to the host's cleanup policy.

Multimodal evaluation does not create containers or images. Remove a completed
`outputs/multimodal-bench/RUN_ID/` directory to delete its results, completion
markers, and response cache. Removing the cache is safe but prevents response
reuse on a subsequent rerun.

## SWE-Verified and SWE-Verified Mini

### SWE-Verified environment setup

```bash
bash setup_swe_verified.sh
```

The setup script:

1. Clones mini-SWE-agent v2.4.6 into `mini-swe-agent/`.
2. Applies `patches/swebench_verified_per_instance_cleanup.patch` so each instance reliably removes its Docker container during cleanup.
3. Installs mini-SWE-agent to generate predictions, the pinned SWE-bench 4.1.0 harness for local evaluation, and datasets for loading SWE-bench Verified and Verified Mini.


### Run SWE-Verified

```bash
bash run_swe_verified.sh \
  --port 8888 \
  --step-limit 250 \
  --tag qwen36_27b_full
```

To run the 50-instance
[SWE-bench Verified Mini](https://evalscope.readthedocs.io/zh-cn/latest/benchmarks/swe_bench_verified_mini.html)
dataset instead, select it through the same runner:

```bash
bash run_swe_verified.sh \
  --dataset verified-mini \
  --port 8888 \
  --step-limit 250 \
  --tag qwen36_27b_mini
```

Generation loads the original Hugging Face dataset
`MariusHobbhahn/swe-bench-verified-mini`, which is mirrored by EvalScope as
`evalscope/swe-bench-verified-mini`. Evaluation uses the canonical Verified
dataset plus the selected Mini instance IDs, so it remains compatible with the
official SWE-bench harness and prebuilt images.

The runner connects to the existing shared vLLM server and leaves it running
when the benchmark exits, whether the benchmark succeeds or fails. Generation
runs as a single continuous process across the whole selection instead of
sequential batches, so the vLLM server always has work queued. As instances
finish generating, they are gathered into chunks of `--eval-chunk-size` and
evaluated with the local harness in the background while generation continues
for the remaining instances, overlapping the CPU/Docker-bound evaluation with
GPU-bound generation instead of alternating between the two. A chunk's Docker
images are removed once its instances are evaluated, so evaluation reuses the
images pulled during generation while disk usage remains bounded. Verified Mini
retains images by default because its smaller image set is practical to reuse.
Pass `--keep-images` to retain images explicitly for a full Verified run. Any
remaining instances are drained into a final, possibly smaller chunk once
generation finishes. With `--skip-eval`, images are removed as each chunk is
claimed according to the same image policy. An independent watchdog
checks the vLLM health endpoint during generation and stops the run after three
consecutive failures by default. The aggregate report is refreshed after every
completed evaluation chunk, so completed results remain available if the run
is interrupted before final cleanup.

To resume an interrupted run, reuse the same `--tag` and the same selection.
mini-SWE-agent skips instances already present in the generation output, and
already-evaluated instances are tracked separately so they are not
re-evaluated:

```bash
bash run_swe_verified.sh \
  --port 8888 \
  --num-tasks 100 \
  --tag qwen36_27b_full
```

To re-run a selection from scratch, use a new `--tag` instead of reusing an
existing one.

Use `--retry-errors` with the same tag to retry cases in `error_ids`. Existing
valid patches are evaluated again without regeneration; invalid submissions are
removed from `generation/preds.json` and generated again. Add
`--retry-empty-patches` to regenerate cases in `empty_patch_ids`. Before retry
state is changed, the runner saves `report.json`, `generation/preds.json`,
`claimed_ids.txt`, `eval_report_list.txt`, and a retry plan under
`retries/retry_<TIMESTAMP>_<PID>/`. New chunk reports override the old
classification for the same instance in the aggregate report. Set
`--retry-attempts N` to enable both retry modes and run up to N retry rounds.
After every round, the runner compares aggregate accuracy with the preceding
round and stops when accuracy no longer improves. It also stops early when both
retry categories are empty.

| Option | Default | Description |
| --- | --- | --- |
| `--dataset NAME` | `verified` | Dataset selection: `verified` (500 instances) or `verified-mini` (50 instances) |
| `--host HOST` | `127.0.0.1` | vLLM host |
| `--port PORT` | `8888` | vLLM port |
| `--served-name NAME` | discovered | Model ID exposed by vLLM |
| `--num-tasks N` | all | Run the first N instances |
| `--slice START:END` | all | Run an explicit slice; cannot be combined with `--num-tasks` |
| `--workers N` | `16` | Parallel mini-SWE-agent workers |
| `--eval-workers N` | `8` | Parallel local harness workers |
| `--step-limit N` | `250` | Maximum model calls per instance |
| `--pull-timeout N` | `600` | Docker image pull/start timeout in seconds |
| `--eval-chunk-size N` | `24` | Finished instances gathered before dispatching an evaluation chunk |
| `--poll-interval N` | `60` | Seconds between checks for newly finished instances |
| `--health-interval N` | `30` | Seconds between vLLM health checks |
| `--health-failures N` | `3` | Consecutive failed health checks before stopping the run |
| `--tag TAG` | UTC timestamp | Output and log label |
| `--retry-errors` | disabled | Re-evaluate valid error patches and regenerate invalid error submissions |
| `--retry-empty-patches` | disabled | Regenerate and evaluate previously empty patches |
| `--retry-attempts N` | `1` | Enable error and empty-patch retries for up to N rounds; stop early when accuracy no longer improves or both categories are empty |
| `--skip-eval` | disabled | Generate predictions without local evaluation |
| `--keep-images` | enabled for Verified Mini | Keep benchmark Docker images after each evaluation chunk |

Outputs:

- mini-SWE-agent results: `mini-swe-agent/results/swe_verified_<TAG>/`
- Live generation output: `mini-swe-agent/results/swe_verified_<TAG>/generation/`
- Per-chunk evaluation artifacts: `mini-swe-agent/results/swe_verified_<TAG>/eval_chunks/`
- Harness predictions: `mini-swe-agent/results/swe_verified_<TAG>/preds.jsonl`
- Local evaluation report with resolved counts and accuracy:
  `mini-swe-agent/results/swe_verified_<TAG>/report.json`
- Log: `logs/swe_verified_<TAG>.log`

For Verified Mini, the same layout uses the `swe_verified_mini_<TAG>` prefix
for the result directory and log file.


## SWE-bench Pro

### SWE-bench Pro environment setup

```bash
bash setup_swebenchpro.sh
```

The setup script clones the pinned `scaleapi/SWE-bench_Pro-os` repository and
its mini-SWE-agent submodule, applies support for the benchmark's Docker Hub
images and reliable container cleanup, and installs the generation and local
Docker evaluation dependencies into the currently activated environment using
uv. Create and activate the environment before running the setup script.

### Run SWE-bench Pro

```bash
bash run_swebenchpro.sh \
  --port 8888 \
  --step-limit 250 \
  --tag qwen36_27b_pro
```

The runner connects to an existing shared vLLM server and leaves it running
when the benchmark exits. Generation runs continuously across the full
selection. As instances finish, their predictions are gathered into evaluation
chunks while generation continues, overlapping CPU/Docker evaluation with GPU
inference. Each chunk receives a matching instance CSV and normalized patch
file, and its Docker images are removed after evaluation. The final chunk may
contain fewer instances than `--eval-chunk-size`. An independent watchdog
stops the run if the vLLM health endpoint repeatedly fails. The aggregate
report is refreshed after every completed evaluation chunk. Host proxy variables
(`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, `NO_PROXY`, and lowercase variants)
are forwarded to generation containers so repository cloning and dependency
installation can use the same network path as the host.

Resume an interrupted run by reusing the same `--tag` and selection. Existing
generation results and already-claimed evaluations are skipped. Use a new tag
to run the selection from scratch.

| Option | Default | Description |
| --- | --- | --- |
| `--host HOST` | `127.0.0.1` | vLLM host |
| `--port PORT` | `8888` | vLLM port |
| `--served-name NAME` | discovered | Model ID exposed by vLLM |
| `--num-tasks N` | all 731 | Run the first N instances |
| `--slice START:END` | all | Run an explicit slice; cannot be combined with `--num-tasks` |
| `--workers N` | `8` | Parallel mini-SWE-agent workers |
| `--eval-workers N` | `4` | Parallel local evaluator workers |
| `--step-limit N` | `250` | Maximum model calls per instance |
| `--pull-timeout N` | `1800` | Docker image pull/start timeout in seconds |
| `--command-timeout N` | `600` | In-container command timeout in seconds |
| `--eval-chunk-size N` | `12` | Finished instances gathered before dispatching an evaluation chunk |
| `--poll-interval N` | `60` | Seconds between checks for newly finished instances |
| `--health-interval N` | `30` | Seconds between vLLM health checks |
| `--health-failures N` | `3` | Consecutive failed health checks before stopping the run |
| `--tag TAG` | UTC timestamp | Output and log label |
| `--skip-eval` | disabled | Generate patches without local evaluation |
| `--block-network` | disabled | Disable network access in evaluation containers |
| `--keep-images` | disabled | Keep benchmark Docker images after each evaluation chunk |

Outputs are grouped under
`SWE-bench_Pro-os/mini-swe-agent/results/swebench_pro_<TAG>/`, including
`preds.json`, normalized `patches.json`, the selected instance CSV, evaluation
artifacts, and `report.json`. Per-chunk artifacts are retained
under `eval_chunks/`, while the top-level files contain the merged results. The
top-level report includes evaluated, resolved, and unresolved counts, accuracy,
resolved/unresolved ID lists, and the per-instance result mapping. The combined
run log is written to `logs/swebench_pro_<TAG>.log`.


## MCP-Atlas

### MCP-Atlas environment setup

Use a separate Python environment from vLLM and the SWE benchmarks when
possible, then run:

```bash
bash setup_mcp_atlas.sh
```

The setup script clones a pinned MCP-Atlas revision, creates `mcp-atlas/.env`
from the upstream template, installs Python and TypeScript dependencies, builds
the agent harness, and pulls the pinned `ghcr.io/scaleapi/mcp-atlas:1.2.7`
sandbox image. It also patches the LLM judge so its request timeout follows
`EVAL_LLM_TIMEOUT_MS` or `LLM_TIMEOUT_MS` instead of being fixed at 60 seconds.
MCP-Atlas requires Node.js 20 or newer; when the system Node.js
is older, setup downloads a pinned workspace-local Node.js 20 runtime under
`.tools/`. The 20 no-key MCP servers work without additional credentials.
Add optional server credentials to `mcp-atlas/.env` to enable key-gated
servers.

### Run MCP-Atlas

With vLLM already serving the model:

```bash
bash run_mcp_atlas.sh \
  --port 8888 \
  --tag qwen36_27b_mcp_full
```

This runs all 500 tasks. By default the runner starts and owns the MCP sandbox
and TypeScript harness, evaluates the model, then uses the same served model as
the LLM judge. Older large tool results are compacted before later model calls
to keep long trajectories within the model context window. Tool output remains
uncapped to match the public runner's default evaluation configuration. Use
`--skip-score` to generate responses only.
The runner stops the MCP services it started but leaves the shared vLLM server
running. It also removes the MCP sandbox image after the run to release disk
space; pass `--keep-image` to retain it for the next run. Host proxy variables
are forwarded to the sandbox because its MCP servers may install packages when
they start. The MCP SDK normally filters proxy variables from stdio child
processes, so the runner explicitly allows its `uvx` and `npx` server processes
to inherit them. It also pins `uvx` to the MCP SDK version validated with the
pinned MCP servers, preventing dependency updates from breaking them. The
runner passes the LLM API origin without a trailing `/v1`
because the MCP harness and judge append `/v1/chat/completions` themselves. An
independent watchdog stops generation or scoring after repeated vLLM health
failures. The sandbox and harness use fixed host ports `1984` and `3001`.

| Option | Default | Description |
| --- | --- | --- |
| `--host HOST` | `127.0.0.1` | vLLM host |
| `--port PORT` | `8888` | vLLM port |
| `--workers N` | `10` | Parallel benchmark tasks |
| `--score-workers N` | `10` | Parallel judge requests |
| `--num-tasks N` | all 500 | Run the first N tasks |
| `--timeout N` | `1800` | Per-task timeout in seconds |
| `--health-interval N` | `30` | Seconds between vLLM health checks |
| `--health-failures N` | `3` | Consecutive failed health checks before stopping the run |
| `--skip-health-check` | disabled | Skip the enabled-server online check |
| `--skip-score` | disabled | Generate responses without scoring |
| `--keep-image` | disabled | Keep the sandbox image after the run |

Outputs are grouped under `mcp-atlas/outputs/run_<TAG>/`: `outputs.csv`,
`run_config.json`, `harness.log`, `sandbox.log`, and the `scored/` reports. The
combined run log is written to `logs/mcp_atlas_<TAG>.log`.

Set `EVAL_LLM_TIMEOUT_MS` to override the judge request timeout. It defaults to
`LLM_TIMEOUT_MS`, which defaults to `600000` milliseconds.
