# MiniMax-M2.7 AutoRound (INC prepare/convert)

This example demonstrates model-free quantization of [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7) via the INC API:

```python
from neural_compressor.torch.quantization import AutoRoundConfig, prepare, convert

config = AutoRoundConfig(
    model_free=True,
    scheme="MXFP8",
    layer_config={"block_sparse_moe": {"scheme": "MXFP4"}},
    export_format="llm_compressor",
    output_dir="/path/to/output",
)
model = "MiniMaxAI/MiniMax-M2.7"
model = prepare(model, config)
model = convert(model)
```

## Requirements

Install dependencies before running quantization or evaluation:

```bash
bash setup.sh
```

Or manually:

```bash
uv pip install -U pip setuptools_rust setuptools_scm
uv pip install -U evalscope lm_eval[api] transformers datasets
uv pip install git+https://github.com/intel/auto-round.git@main
uv pip install compressed-tensors --no-deps
uv pip install vllm
```

## Quick Start

### 1. Quantize

```bash
cd examples/pytorch/llm/minimax
bash run_quant.sh \
  --dtype=mxfp4_mixed \
  --input_model=MiniMaxAI/MiniMax-M2.7 \
  --output_model=~/models/minimax-m2.7-mxfp
```

To additionally quantize the KV cache / attention to static FP8 (requires `transformers==4.57.6`):

```bash
bash run_quant.sh \
  --dtype=mxfp4_mixed \
  --input_model=MiniMaxAI/MiniMax-M2.7 \
  --output_model=~/models/minimax-m2.7-mxfp-fp8kv \
  --static_kv_dtype=fp8 \
  --static_attention_dtype=fp8
```

> Note: `--static_kv_dtype` / `--static_attention_dtype` require loading the real model, so `model_free` is automatically disabled when either is set.

### 2. Serve + Evaluate

`run_benchmark.sh` is a thin wrapper over the shared drivers. RULER tasks (`*ruler*`, `niah_*`) run with `benchmark/lm_eval/run_lm_eval.sh`; all other tasks run with `benchmark/evalscope/run_evalscope.sh`, which starts `vllm serve`, runs each task `--num_runs` times with evalscope and prints a score table. Tensor parallel size is inferred from `CUDA_VISIBLE_DEVICES`.

```bash
CUDA_VISIBLE_DEVICES=3,4,5,6 bash run_benchmark.sh \
  --model_path=~/models/minimax-m2.7-mxfp \
  --port=8001 \
  --tasks=gpqa_diamond,aime25,gsm8k,piqa,hellaswag,live_code_bench
```

By default each task is repeated 5 times and the trimmed mean is reported (drop max/min, average the remaining 3); `hellaswag`, `piqa`, `mmlu` and `gsm8k` always run once. Use `--num_runs=1` for a single pass:

```bash
CUDA_VISIBLE_DEVICES=3,4,5,6 bash run_benchmark.sh \
  --model_path=~/models/minimax-m2.7-mxfp \
  --num_runs=1
```

For a model quantized with static FP8 KV cache / attention:

```bash
CUDA_VISIBLE_DEVICES=3,4,5,6 bash run_benchmark.sh \
  --model_path=~/models/minimax-m2.7-mxfp-fp8kv \
  --static_kv_dtype=fp8 \
  --static_attention_dtype=fp8
```

RULER long-context evaluation (default max position `131072`):

```bash
CUDA_VISIBLE_DEVICES=3,4,5,6 bash run_benchmark.sh \
  --model_path=~/models/minimax-m2.7-mxfp \
  --tasks=ruler
```

Equivalent vLLM command used by `run_benchmark.sh` for evalscope tasks:

```bash
CUDA_VISIBLE_DEVICES=3,4,5,6 vllm serve ~/models/minimax-m2.7-mxfp \
  --port 8001 \
  --tensor-parallel-size 4 \
  --served-model-name minimax-m2.7 \
  --max-model-len 102400 \
  --trust-remote-code \
  --tool-call-parser minimax_m2 \
  --enable-auto-tool-choice \
  --reasoning-parser minimax_m2 \
  --max-num-seqs 1024 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill
```

## CLI Arguments

### `run_quant.sh`

- `--dtype`: quantization preset.
  - `mxfp4_mixed`: `scheme=MXFP8` globally + `block_sparse_moe` layers use `scheme=MXFP4`
  - `mxfp8`: `scheme=MXFP8` globally
  - `mxfp4`: `scheme=MXFP4` globally
- `--input_model`: HF model name or local model path.
- `--output_model`: output directory.
- `--export_format`: `auto_round` or `llm_compressor` (default: `llm_compressor`).
- `--static_kv_dtype`: static KV cache data type, e.g. `fp8` (default: `auto`).
- `--static_attention_dtype`: static attention data type, e.g. `fp8` (default: `auto`).

### `run_benchmark.sh`

- `--model_path`: quantized model path for vLLM and evalscope (required).
- `--tasks`: comma-separated tasks (default: `gpqa_diamond,aime25,gsm8k,piqa`). RULER tasks (`*ruler*`, `niah_*`) run with lm_eval, the others with evalscope.
- `--num_runs`: number of times each task is evaluated (default: `5`). `hellaswag`, `piqa`, `mmlu` and `gsm8k` always run once (override the list with the `SINGLE_RUN_TASKS` environment variable). With 3 or more runs, the highest and lowest scores are dropped and the remaining (kept) scores are averaged; otherwise all scores are averaged. After each task a result table (every run, kept/dropped, mean of kept) is printed, and at the end a summary table lists every run of each task (dropped scores are marked with `(D)`) and the mean of kept scores, plus the average over all tasks.
- `--temperature`: generation temperature (default: `1.0`).
- `--top_p`: generation top_p (default: `0.95`).
- `--max_tokens`: max generated tokens (default: `64000`).
- `--max_model_len`: vLLM max context length (default: `102400`).
- `--served_model_name`: served model name alias (default: `minimax-m2.7`).
- `--eval_batch_size`: evalscope concurrent requests (default: `256`).
- `--port`: vLLM API port (default: `$SERVER_PORT` or `8001`; the next free port is used when not specified and busy).
- `--static_kv_dtype`: vLLM `kv_cache_dtype`, e.g. `fp8` (default: `auto`).
- `--static_attention_dtype`: set to `fp8` to enable FP8 attention (implies `kv_cache_dtype=fp8`).
- `--skip_serve=true`: skip starting vLLM and reuse an existing endpoint on `--port` (evalscope tasks only).
- lm_eval (RULER) options: `--ruler_max_pos` (default: `131072`), `--gpu_memory_utilization` (default: `0.90`).

Outputs are written under `<model_name>-tp<N>-evalscope/` in the current directory: the vLLM log, the evaluation log, and per-run evalscope outputs in `outputs_<timestamp>/<task>/run<i>/`. RULER results are written under `<model_name>-tp<N>-eval/`.

## Evaluation Generation Config

All tasks use thinking mode with `reasoning_effort=max`:

```json
{
  "temperature": 1.0,
  "top_p": 0.95,
  "n": 1,
  "extra_body": {
    "chat_template_kwargs": {
      "enable_thinking": true,
      "reasoning_effort": "max"
    }
  },
  "max_tokens": 64000
}
```

`live_code_bench` additionally sets `subset_list: ["v6"]`.

## Notes

- This flow requires:
  - `config` is `AutoRoundConfig`
  - `config.model_free=True`
  - `model` passed to `prepare/convert` is a `str` (model path or HF model name)
- The example uses `reloading=False` by default and saves quantized artifacts to `--output_model`.
