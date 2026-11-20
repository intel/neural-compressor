# HumanEval+

Run [HumanEval+](https://github.com/evalplus/evalplus) against an
OpenAI-compatible vLLM endpoint.

## Setup

```bash
cd benchmark/humaneval_plus
python3 -m venv .venv
.venv/bin/pip install evalplus==0.3.1
.venv/bin/python scripts/patch_evalplus.py
```

The patch raises `max_new_tokens` to 4096 and makes the OpenAI decoder tolerate
`content=None` / retry empty completions, which reasoning-style models need.
The first run downloads HumanEval+ v0.1.10 from the EvalPlus release and caches
it under `~/.cache/evalplus`.

## Independent repeated runs

`run_humaneval_plus_independent.sh` runs N **separate full benchmark
invocations**, one sample per problem per run. This is the mode to use for a
7-run mean/std report.

```bash
./run_humaneval_plus_independent.sh \
  --host <vllm-host> \
  --port <vllm-port> \
  --runs 7 \
  --tag demo_humaneval_indep
```

Each run writes its own EvalPlus output and `state.json`; the suite aggregates
pass@1, passed counts, mean and standard deviation.

## N samples per problem

`run_humaneval_plus.sh --runs N` means N samples **inside one EvalPlus
invocation**, not N sequential benchmark jobs. Use it when you want pass@1 over
multiple samples per problem.

```bash
./run_humaneval_plus.sh \
  --host <vllm-host> \
  --port <vllm-port> \
  --runs 7 \
  --temperature 0.6 \
  --tag demo_humaneval_7s
```

## Options

Run either script with `--help` for the full list. Common options:

| Option | Default | Description |
| --- | --- | --- |
| `--host HOST` | `127.0.0.1` | vLLM host |
| `--port PORT` | `8000` | vLLM port |
| `--served-name NAME` | discovered | Model id exposed by `/v1/models` |
| `--runs N` | `7` | Independent runs (independent script) or samples per problem |
| `--temperature T` | `0.6` | Sampling temperature |
| `--root DIR` | `<script dir>/results` | Results root |
| `--expected N` | `164` | Tasks per run |
| `--mini` | off | Use the HumanEval+ Mini test set |
| `--evalplus PATH` | `<script dir>/.venv/bin/evalplus.evaluate` | EvalPlus executable |
| `--parallel N` | half CPU | EvalPlus evaluation workers |
| `--wait-timeout N` | `300` | vLLM readiness timeout in seconds |

## Outputs

- Single run: `<root>/<tag>/` with `state.json`, EvalPlus artifacts and
  `report.json`.
- Independent suite: `<root>/<tag>_suite/` with `state.json`, `summary.json`,
  `summary.md`, per-run logs and `runs/<tag>_runNN/` containing `state.json`,
  `summary.*` and `evalplus/`.
