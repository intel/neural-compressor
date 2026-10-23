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


## SWE-bench Pro V2

### SWE-bench Pro environment setup

```bash
bash setup_swebenchpro.sh
```

The setup script clones `scaleapi/SWE-bench_Pro-os` at the V2 release commit
pinned in `versions.env`, verifies its published SHA256 manifest, and installs
Harbor 0.24.0 into the activated **Python 3.12+** environment using uv.
Docker with the Compose plugin is required for local runs. The official V2
task directories contain the instructions, sanitized GHCR image references,
and hidden-test verifiers; Hugging Face's mutable default is never loaded.

**Migration from V1:** move an existing `SWE-bench_Pro-os` checkout aside (or
set `BENCHMARK_DIR` to a new directory) before setup. Setup refuses to silently
overwrite another revision. V1's 731 tasks, `jefzda/sweap-images` images,
`swe_bench_pro_eval.py`, and mini-SWE-agent submodule are not used by V2.
V1 and V2 scores are not comparable.

### Run SWE-bench Pro

```bash
bash run_swebenchpro.sh \
  --port 8888 \
  --step-limit 250 \
  --tag qwen36_27b_pro
```

The runner uses the official V2 locked protocol:

1. A thin subclass of official `locked_mini_swe:LockedMiniSwe` runs the pinned mini-SWE-agent in each
  sanitized image, with a 50-minute task budget. Harbor enforces the task's
  agent-phase network isolation, allowing only the model endpoint host.
  Setup remains online. The official agent captures `agent/model.patch`;
  verification is disabled in this potentially modified sandbox.
  The subclass only installs `litellm[mcp]==1.106.2`: the upstream bare
  LiteLLM installation otherwise fails on its first completion with a missing
  MCP import. Patch capture, timeout enforcement, and execution are inherited
  unchanged from the official locked agent.
2. `patch_replay:PatchReplayAgent` applies the captured diff in a **new**
  sandbox and runs the unchanged official V2 verifier. Only these regrade
  rewards determine the reported accuracy. Missing trials, infrastructure
  failures, and missing verifier rewards remain in the selected denominator
  and make the runner exit nonzero after writing its report.

The vLLM server is left running. For local Docker, bind it to `0.0.0.0`:
host loopback URLs are mapped to the Docker bridge gateway. If using a proxy,
ensure this gateway bypasses the proxy. Override `--agent-base-url` if the
endpoint needs another sandbox-reachable address. Modal requires configured
credentials and a remotely reachable model endpoint, not host loopback.
Only the model host is allowlisted. Host proxy settings are forwarded for
online setup and verification, but the proxy host is **not** allowlisted
during agent execution; the model endpoint bypasses the proxy. Environments
requiring an outbound proxy should export it before running the script.
Generation exceptions or missing captured patches abort before scoring and
are saved in `generation_errors.json`; genuinely empty captured patches
remain valid submissions and are replayed as empty patches.

Tasks are selected in sorted task-directory order from the pinned tree;
`--subset hard` selects the official HARD-51 before slicing. Each run needs a
new tag. Old V1 resume/chunking/health-watchdog/image-deletion flags are not
supported; Harbor owns container lifecycle and enforces phase timeouts.
Docker images are retained for reuse, not removed from a shared host.
`--skip-eval` captures patches without claiming a benchmark score. Keep both
Harbor jobs when publishing results and follow the upstream V2 network and
git-history probe instructions before a full benchmark run.

| Option | Default | Description |
| --- | --- | --- |
| `--host HOST` | `127.0.0.1` | vLLM host |
| `--port PORT` | `8888` | vLLM port |
| `--served-name NAME` | discovered | Model ID exposed by vLLM |
| `--num-tasks N` | all 642 | Run the first N sorted V2 tasks |
| `--slice START:END` | all | Run an explicit slice; cannot be combined with `--num-tasks` |
| `--agent-base-url URL` | derived | Sandbox-reachable OpenAI API endpoint including `/v1` |
| `--environment NAME` | `docker` | Harbor backend: `docker` or `modal` |
| `--subset NAME` | `full` | V2 full (642) or hard (51), before slicing |
| `--workers N` | `8` | Parallel locked-agent trials |
| `--eval-workers N` | `4` | Parallel pristine-sandbox replay trials |
| `--step-limit N` | `250` | Maximum model calls per instance |
| `--tag TAG` | UTC timestamp | Output and log label |
| `--skip-eval` | disabled | Generate patches without a score |

Outputs are grouped under `outputs/swebench_pro_v2_<TAG>/`:

- `selection.json`: V2 release commit, subset, selected IDs, and agent limits
- `tasks/`: unchanged copies of the selected official task directories
- `agent.yaml`: runtime agent configuration (default 8192 output tokens)
- `jobs/generation/`: agent trajectories, trial results, and captured patches
- `jobs/regrade/`: pristine-sandbox verifier artifacts and Harbor results
- `report.json`: fresh-sandbox accuracy, per-instance results, and error IDs

Logs are written to `logs/swebench_pro_v2_<TAG>.log`. Harbor job configs may
contain the model API credentials supplied to the agent: keep artifacts private
and redact secrets before publishing them.

Dependency-light regression tests (no model, Docker, or downloads required):

```bash
python -m unittest discover -s tests -p 'test_swebench_pro_v2.py' -v
```

Official protocol: <https://github.com/scaleapi/SWE-bench_Pro-os/blob/66f92766bba642462d4bbe5479e83f91f9211862/v2/README.md>.


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
