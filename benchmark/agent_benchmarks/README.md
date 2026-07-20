# Agent Benchmarks

Scripts for evaluating language models on SWE-bench Pro, SWE-bench Verified, and MCP-Atlas using a local vLLM inference server.

```
setup_agent.sh   — one-time environment setup (clone repos, install packages, pull Docker image)
run_agent.sh     — benchmark runner
```

---

## Requirements

- Python environment already activated (conda or venv) with Python 3.10+
- Node.js + npm (for MCP-Atlas agent harness)
- Docker (for MCP-Atlas sandbox and SWE-bench Pro local evaluation)
- vLLM (installed by `setup_agent.sh`)

`setup_agent.sh` installs vLLM in one of two modes:

- `MODEL_NAME` contains `DeepSeek-V4`
  - installs the DeepSeek-V4-compatible vLLM fork
  - aligns the CUDA toolchain before and after install
  - keeps the environment on the validated CUDA 13.2.86 toolchain
- otherwise
  - installs generic `vllm`, `torch`, `torchaudio`, and `torchvision`
  - does not pin those package versions

SWE-bench Verified also requires:
- `SWEBENCH_API_KEY` env var for remote submission via `sb-cli`

---

## Directory layout after setup

```
BENCHMARK_DIR/
├── setup_agent.sh
├── run_agent.sh
├── patches/
│   └── swebench_pro_image.patch   # required before setup swebp
├── SWE-bench_Pro-os/              # cloned by setup (swebp)
│   └── mini-swe-agent/            # git submodule — v1 agent
├── mini-swe-agent/                # cloned by setup (swe-verified)
├── mcp-atlas/                     # cloned by setup (mcp-atlas)
└── logs/
```

`BENCHMARK_DIR` defaults to `$PWD`. Set it explicitly if running from elsewhere:
```bash
BENCHMARK_DIR=/path/to/agent_benchmarks bash setup_agent.sh swebp
```

---

## Setup

Run **once per task** inside the activated environment for that task:

```bash
cd /path/to/agent_benchmarks
MODEL_NAME=DeepSeek-V4-Flash bash setup_agent.sh swebp
MODEL_NAME=DeepSeek-V4-Flash bash setup_agent.sh swe-verified
MODEL_NAME=DeepSeek-V4-Flash bash setup_agent.sh mcp-atlas
```

`MODEL_NAME` can also be passed as the second positional argument:

```bash
bash setup_agent.sh swebp DeepSeek-V4-Flash
```

> **Note for `swebp`**: place the Docker image patch at `patches/swebench_pro_image.patch` before running setup.

---

## Running benchmarks

```bash
cd /path/to/agent_benchmarks
bash run_agent.sh --task TASK [OPTIONS]
```

### Common options

| Option | Default | Description |
|--------|---------|-------------|
| `--task TASK` | `swebp` | `swebp` \| `swe-verified` \| `mcp-atlas` |
| `--port N` | `8888` | vLLM API port |
| `--model PATH` | — | Model path; launches vLLM automatically when set |
| `--scheme NAME` | `FP8` | vLLM serving scheme; used by DeepSeek-V4 model presets |
| `--max-model-len N` | `262144` | vLLM `max_model_len` |
| `--served-name NAME` | `gpt-3.5-turbo` | vLLM `served-model-name` |
| `--tag NAME` | timestamp | Label for output directory and logs |
| `--skip-serve` | — | Skip vLLM launch and readiness wait |

### vLLM serving options (env vars)

Set before calling the script:

| Variable | Default | Description |
|----------|---------|-------------|
| `KV_CACHE_DTYPE` | `fp8` | `--kv-cache-dtype` |
| `BLOCK_SIZE` | `256` | `--block-size` |
| `TENSOR_PARALLEL_SIZE` | `1` | `--tensor-parallel-size` |
| `GPU_MEM_UTIL` | `0.9` | `--gpu-memory-utilization` |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `1800` | vLLM startup timeout (seconds) |
| `VLLM_WAIT_RETRIES` | `180` | Health-check retries (× 10s each) |

### DeepSeek-V4 serving presets

For model paths whose basename contains `DeepSeek-V4-Flash` or `DeepSeek-V4-Pro`:

- `--scheme FP8`
  - adds `SAFETENSORS_FAST_GPU=1`
  - adds `--moe-backend deep_gemm_mega_moe`
  - adds `--enable-expert-parallel`
- `--scheme MXFP4`
  - adds `SAFETENSORS_FAST_GPU=1`
  - adds `--moe-backend cutlass`

All `vllm serve` launches from `run_agent.sh` add:

- `--trust-remote-code`
- `--enable-auto-tool-choice`
- `--tool-call-parser hermes`

### SWE-bench Pro (`--task swebp`)

| Option | Default | Description |
|--------|---------|-------------|
| `--num-tasks N` | all | Limit to first N instances |
| `--slice S:E` | — | Exact slice, e.g. `0:20` |
| `--workers N` | `2` | Parallel agent workers |
| `--step-limit N` | `250` | Max steps per instance |
| `--redo` | — | Re-run already-completed instances |

Outputs: `SWE-bench_Pro-os/mini-swe-agent/results/run_<TAG>/`

Local Docker evaluation runs automatically after the agent finishes.

### SWE-bench Verified (`--task swe-verified`)

| Option | Default | Description |
|--------|---------|-------------|
| `--num-tasks N` | all | Limit to first N instances |
| `--slice S:E` | — | Exact slice |
| `--workers N` | `2` | Parallel agent workers |
| `--step-limit N` | `250` | Max steps per instance |

Requires `SWEBENCH_API_KEY` for remote submission via `sb-cli`.

Outputs: `mini-swe-agent/results/swe_verified_<TAG>/`

### MCP-Atlas (`--task mcp-atlas`)

| Option | Default | Description |
|--------|---------|-------------|
| `--num-tasks N` | all 500 | Limit to first N tasks |
| `--workers N` | `5` | Parallel eval concurrency |
| `--sandbox-port P` | `1984` | MCP sandbox Docker port |
| `--harness-port P` | `3001` | Agent harness port |
| `--skip-sandbox` | — | Skip Docker sandbox startup |

Outputs: `mcp-atlas/outputs/run_<TAG>/`

The script starts the MCP sandbox Docker container and Node.js agent harness automatically.
Any stale harness process pointing to a different directory or vLLM port is replaced.

---

## Examples

### SWE-bench Pro — DeepSeek-V4-Flash FP8 smoke test

```bash
cd /path/to/agent_benchmarks
MODEL_NAME=DeepSeek-V4-Flash bash setup_agent.sh swebp

bash run_agent.sh \
    --task       swebp \
    --model      /path/to/DeepSeek-V4-Flash \
    --scheme     FP8 \
    --port       8888 \
    --num-tasks  10 \
    --workers    2 \
    --tag        smoke_swebp
```

### SWE-bench Verified — DeepSeek-V4-Flash MXFP4 smoke test

```bash
cd /path/to/agent_benchmarks
MODEL_NAME=DeepSeek-V4-Flash bash setup_agent.sh swe-verified

export SWEBENCH_API_KEY=<your_key>
bash run_agent.sh \
    --task       swe-verified \
    --model      /path/to/DeepSeek-V4-Flash \
    --scheme     MXFP4 \
    --port       8888 \
    --num-tasks  10 \
    --workers    2 \
    --tag        smoke_verified
```

### MCP-Atlas — DeepSeek-V4-Flash MXFP4 smoke test

```bash
cd /path/to/agent_benchmarks
MODEL_NAME=DeepSeek-V4-Flash bash setup_agent.sh mcp-atlas

bash run_agent.sh \
    --task      mcp-atlas \
    --model     /path/to/DeepSeek-V4-Flash \
    --scheme    MXFP4 \
    --port      8888 \
    --num-tasks 10 \
    --workers   5 \
    --tag       smoke_mcp
```

If vLLM, sandbox, and harness are already running on the default port `8888`:

```bash
bash run_agent.sh \
    --task         mcp-atlas \
    --port         8888 \
    --skip-serve \
    --skip-sandbox \
    --num-tasks    10 \
    --workers      5 \
    --tag          smoke_mcp2
```

---

## Notes

- **One environment per task** — each task has its own dependencies; run `setup_agent.sh` inside the correct activated environment.
- **Multiple tasks simultaneously** — use different `--port` values and `CUDA_VISIBLE_DEVICES` for each.
- **Resuming a run** — use `--redo` (swebp) or the same `--tag`; already-completed instances are skipped automatically.
- **Logs** — all output is tee'd to `logs/agent_<task>_<tag>.log`.
