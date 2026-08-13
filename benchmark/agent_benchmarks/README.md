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

## SWE-Verified

### SWE-Verified environment setup

```bash
bash setup_swe_verified.sh
```

The setup script:

1. Clones mini-SWE-agent v2.4.6 into `mini-swe-agent/`.
2. Applies `patches/swebench_verified_per_instance_cleanup.patch` so each instance reliably removes its Docker container during cleanup.
3. Installs mini-SWE-agent to generate predictions and the official SWE-bench harness to evaluate those predictions locally. It also installs datasets for loading SWE-bench Verified.


### Run SWE-Verified

```bash
bash run_swe_verified.sh \
  --port 8888 \
  --step-limit 250 \
  --tag qwen36_27b_full
```

The runner connects to the existing shared vLLM server and leaves it running
when the benchmark exits, whether the benchmark succeeds or fails. Generation
runs as a single continuous process across the whole selection instead of
sequential batches, so the vLLM server always has work queued. As instances
finish generating, they are gathered into chunks of `--eval-chunk-size` and
evaluated with the local harness in the background while generation continues
for the remaining instances, overlapping the CPU/Docker-bound evaluation with
GPU-bound generation instead of alternating between the two. A chunk's Docker
images are removed once its instances are evaluated, so evaluation reuses the
images pulled during generation while disk usage remains bounded. Any
remaining instances are drained into a final, possibly smaller chunk once
generation finishes. With `--skip-eval`, images are removed as each chunk is
claimed unless `--keep-images` is also specified. An independent watchdog
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

| Option | Default | Description |
| --- | --- | --- |
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
| `--skip-eval` | disabled | Generate predictions without local evaluation |
| `--keep-images` | disabled | Keep benchmark Docker images after each evaluation chunk |

Outputs:

- mini-SWE-agent results: `mini-swe-agent/results/swe_verified_<TAG>/`
- Live generation output: `mini-swe-agent/results/swe_verified_<TAG>/generation/`
- Per-chunk evaluation artifacts: `mini-swe-agent/results/swe_verified_<TAG>/eval_chunks/`
- Harness predictions: `mini-swe-agent/results/swe_verified_<TAG>/preds.jsonl`
- Local evaluation report with resolved counts and accuracy:
  `mini-swe-agent/results/swe_verified_<TAG>/report.json`
- Log: `logs/swe_verified_<TAG>.log`


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
the LLM judge. Use `--skip-score` to generate responses only.
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
