# Benchmarks

This folder contains benchmark drivers for quantized LLM evaluation.

| Directory | Benchmark | Mode |
| --- | --- | --- |
| `agent_benchmarks/` | SWE-bench Verified / Verified Mini, SWE-bench Pro, Terminal-Bench, MCP-Atlas, multimodal benchmarks | Agent runs, optional repeated suites |
| `evalscope/` | EvalScope tasks (AIME 2026, MATH-500, GPQA Diamond, ...) | Local vLLM server or existing OpenAI-compatible endpoint |
| `humaneval_plus/` | HumanEval+ | Existing OpenAI-compatible endpoint, independent repeated runs |
| `ifbench/` | IFBench (precise instruction following) | Existing OpenAI-compatible endpoint, repeated suites |
| `lm_eval/` | lm-evaluation-harness | Local vLLM server |

Most endpoint-mode runners use the same optional conventions:

- `--host` / `--port` or `--api-url` select the OpenAI-compatible endpoint.
- `--served-name` pins the model id; when omitted, it is discovered from
  `/v1/models`.
- `--runs N` repeats the complete benchmark N times and writes per-run details
  plus an aggregate `summary.json` / `summary.md`.
- `--tag` names the run; use a fresh tag for every retry.
- A suite directory contains `state.json` with the exact command, per-run
  commands, status, timings and metrics. Any dashboard that understands this
  state schema (for example TaskWatch) can discover the run without reading
  logs.
