# Benchmark Runner

Two scripts for running SWE-bench Pro, SWE-bench Verified, and MCP-Atlas benchmarks against a local vLLM inference server.

```
setup_agent.sh   — one-time environment setup (clone repos, create venvs, pull Docker image)
run_agent.sh     — benchmark runner
```

---

## Requirements

- Python 3.10 (via `uv`)
- [`uv`](https://github.com/astral-sh/uv) — `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Node.js + npm (for MCP-Atlas harness)
- Docker (for MCP-Atlas sandbox and SWE-bench Pro evaluation)
- [vLLM](https://github.com/vllm-project/vllm) (installed automatically into each venv)
- HuggingFace access to `ScaleAI/SWE-bench_Pro` and `ScaleAI/MCP-Atlas`

### SWE-bench Verified only

- `sb-cli` token (`SWEBENCH_API_KEY` env var) for remote submission

---

## Directory layout

After setup, `BENCHMARK_DIR` will contain:

```
BENCHMARK_DIR/
├── setup_agent.sh
├── run_agent.sh
├── patches/
│   └── swebench_pro_image.patch   # Docker image fix for SWE-bench Pro
├── SWE-bench_Pro-os/              # cloned by setup (swebp)
│   └── mini-swe-agent/            # git submodule — v1 agent
├── mini-swe-agent/                # cloned by setup (swe-verified)
├── mcp-atlas/                     # cloned by setup (mcp-atlas)
├── .venv-swebp/                   # Python venv for swebp
├── .venv-swe/                     # Python venv for swe-verified
├── .venv-mcp/                     # Python venv for mcp-atlas
└── logs/
```

---

## Setup

Run once per benchmark task (or `all` to set up everything):

```bash
cd /path/to/BENCHMARK_DIR
BENCHMARK_DIR=$PWD bash setup_agent.sh [swebp|swe-verified|mcp-atlas|all]
```

`BENCHMARK_DIR` defaults to `$PWD` if not set.

### SWE-bench Pro patch

Before running setup, place the Docker image patch at:

```
BENCHMARK_DIR/patches/swebench_pro_image.patch
```

This patch redirects the evaluator to use the correct Docker images for SWE-bench Pro instances.

---

## Running benchmarks

```bash
cd /path/to/BENCHMARK_DIR
bash run_agent.sh --task TASK [OPTIONS]
```

### Common options

| Option | Default | Description |
|--------|---------|-------------|
| `--task TASK` | `swebp` | `swebp` \| `swe-verified` \| `mcp-atlas` |
| `--port N` | `8888` | vLLM API port |
| `--model PATH` | — | Model path; launches vLLM automatically when set |
| `--max-model-len N` | `262144` | vLLM `max_model_len` |
| `--served-name NAME` | `gpt-3.5-turbo` | vLLM `served-model-name` |
| `--tag NAME` | timestamp | Label for output directory and logs |
| `--skip-serve` | — | Skip vLLM launch and readiness wait |

### SWE-bench Pro (`--task swebp`)

| Option | Default | Description |
|--------|---------|-------------|
| `--num-tasks N` | all | Limit to first N instances |
| `--slice S:E` | — | Exact slice, e.g. `0:20` |
| `--workers N` | `2` | Parallel agent workers |
| `--step-limit N` | `250` | Max steps per instance |
| `--redo` | — | Re-run already-completed instances |

Outputs written to `SWE-bench_Pro-os/mini-swe-agent/results/run_<TAG>/`.

Local Docker evaluation runs automatically after the agent finishes.

### SWE-bench Verified (`--task swe-verified`)

| Option | Default | Description |
|--------|---------|-------------|
| `--num-tasks N` | all | Limit to first N instances |
| `--slice S:E` | — | Exact slice |
| `--workers N` | `2` | Parallel agent workers |
| `--step-limit N` | `250` | Max steps per instance |

Requires `SWEBENCH_API_KEY` env var for remote submission via `sb-cli`.

Outputs written to `mini-swe-agent/results/swe_verified_<TAG>/`.

### MCP-Atlas (`--task mcp-atlas`)

| Option | Default | Description |
|--------|---------|-------------|
| `--num-tasks N` | all 500 | Limit to first N tasks |
| `--workers N` | `5` | Parallel eval concurrency |
| `--sandbox-port P` | `1984` | MCP sandbox Docker port |
| `--harness-port P` | `3001` | Agent harness port |
| `--skip-sandbox` | — | Skip Docker sandbox startup |

Outputs written to `mcp-atlas/outputs/run_<TAG>/`.

The script starts the MCP sandbox Docker container and Node.js agent harness automatically. Any stale harness process pointing to a different directory is replaced.

---

## Examples

### Smoke test — SWE-bench Pro (10 instances, vLLM on port 8888)

```bash
CUDA_VISIBLE_DEVICES=0 bash run_agent.sh \
    --task swebp \
    --model /path/to/model \
    --port 8888 \
    --num-tasks 10 \
    --workers 2 \
    --step-limit 100 \
    --tag smoke_swebp
```

### Full run — SWE-bench Verified (vLLM already running)

```bash
SWEBENCH_API_KEY=<your_key> \
CUDA_VISIBLE_DEVICES=1 bash run_agent.sh \
    --task swe-verified \
    --port 8881 \
    --skip-serve \
    --workers 4 \
    --tag run_verified_01
```

### Smoke test — MCP-Atlas

```bash
CUDA_VISIBLE_DEVICES=0 bash run_agent.sh \
    --task mcp-atlas \
    --model /path/to/model \
    --port 8882 \
    --num-tasks 10 \
    --workers 5 \
    --tag smoke_mcp
```

If vLLM is already running and the sandbox/harness are up:

```bash
bash run_agent.sh \
    --task mcp-atlas \
    --port 8882 \
    --skip-serve \
    --skip-sandbox \
    --num-tasks 10 \
    --workers 5 \
    --tag smoke_mcp2
```

---

## Environment variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `BENCHMARK_DIR` | both scripts | Root directory (defaults to `$PWD`) |
| `SWEBENCH_API_KEY` | swe-verified | API key for `sb-cli submit` |
| `CUDA_VISIBLE_DEVICES` | run_agent.sh | GPU selection for vLLM |
| `GPU_MEM_UTIL` | run_agent.sh | vLLM `gpu-memory-utilization` (default `0.7`) |
| `VLLM_ENGINE_READY_TIMEOUT_S` | run_agent.sh | vLLM startup timeout in seconds (default `1800`) |
| `VLLM_WAIT_RETRIES` | run_agent.sh | Health-check retries before giving up (default `180`) |

---

## Notes

- **Multiple tasks simultaneously**: use different ports and `CUDA_VISIBLE_DEVICES` for each task.
- **Resuming a run**: pass `--redo` (swebp) or re-run with the same `--tag`; already-completed instances are skipped.
- **vLLM served model name**: the default `gpt-3.5-turbo` works with all three tasks without extra config. Change with `--served-name`.
