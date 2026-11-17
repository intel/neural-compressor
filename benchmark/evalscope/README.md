# EvalScope

Two drivers live here:

- `run_evalscope.sh` starts a local `vllm serve` from a model path, evaluates
  every task `--num_runs` times, and summarizes the scores with
  `eval_summary.py`. Use it when the model is served on the same host.
- `run_evalscope_7r.sh` connects to an existing OpenAI-compatible endpoint,
  discovers the model id and `reasoning_effort`, runs the complete task list N
  times, and writes a per-run + aggregate suite. Use it for remote services or
  when several benchmark runners share one vLLM server.

## Endpoint mode: `run_evalscope_7r.sh`

```bash
export EVALSCOPE_BIN=/path/to/evalscope
./run_evalscope_7r.sh \
  --host <vllm-host> \
  --port <vllm-port> \
  --runs 7 \
  --tag demo_evalscope_7r
```

Defaults:

| Option | Default | Description |
| --- | --- | --- |
| `--host` / `--port` | `127.0.0.1:8000` | vLLM endpoint |
| `--api-url` | derived | Full API base, for example `http://host:8000/v1` |
| `--served-name` | discovered | Model id from `/v1/models` |
| `--reasoning-effort` | discovered | `reasoning_effort`; falls back to `xhigh` |
| `--datasets` | `aime26 math_500 gpqa_diamond` | Space-separated EvalScope tasks |
| `--runs` | `7` | Independent full-suite runs |
| `--temperature` | `1.0` | Sampling temperature |
| `--top-p` | `0.95` | Nucleus sampling |
| `--max-tokens` | `131072` | Per-request generation limit |
| `--timeout` | `18000` | Per-request timeout in seconds |
| `--eval-batch-size` | `16` | EvalScope concurrency |
| `--root` | `<script dir>/results` | Results root |
| `--evalscope` | `EVALSCOPE_BIN` or `evalscope` | EvalScope executable |
| `--dry-run` | off | Print the resolved configuration and exit |

`reasoning_effort` is read from model metadata (`reasoning_effort`,
`reasoning.effort`, or `default_reasoning_effort`) or inferred from the model
id/root path. Override it explicitly when the endpoint metadata is incomplete.

## Outputs

Each run writes raw EvalScope reports. The suite aggregates every task and
writes:

```text
<root>/<tag>_suite/
├── state.json
├── summary.json
├── summary.md
├── run01.log ...
└── runs/<tag>_runNN/evalscope/<timestamp>/reports/
```

`summary.json` contains per-task scores for every run, the overall score, and
the mean and standard deviation across completed runs.
