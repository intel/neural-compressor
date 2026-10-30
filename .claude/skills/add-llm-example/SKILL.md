---
name: add-llm-example
description: Add a new LLM quantization example under examples/pytorch/llm (quantize.py, run_quant.sh, run_benchmark.sh, README). Use when onboarding a new model family such as Qwen, DeepSeek, Llama, MiniMax, Kimi or GLM, or when an existing example's scripts need to follow the shared CLI conventions.
---

# Add a new LLM quantization example

Every example under `examples/pytorch/llm/<model>/` exposes the same CLI so that
users and CI can drive any model the same way. Follow the layout below.

```
examples/pytorch/llm/<model>/
├── quantize.py        # AutoRoundConfig presets + prepare/convert
├── run_quant.sh       # quantization entry point
├── run_benchmark.sh   # thin wrapper over the shared lm-eval driver
├── requirements.txt
└── README.md
```

## CLI conventions

All shell scripts accept **only** `--key=value` form. Do not add `--key value`
variants. Unknown flags must print usage and exit non-zero.

### run_quant.sh

| Flag | Default | Notes |
| --- | --- | --- |
| `--dtype=` | required | preset key, e.g. `mxfp4`, `mxfp8`, `nvfp4` |
| `--input_model=` | required | HF model ID or local path |
| `--output_model=` | required | output directory |
| `--export_format=` | `llm_compressor` | `auto_round` or `llm_compressor` |
| `--static_kv_dtype=` | `auto` | `fp8` enables static KV quantization |
| `--static_attention_dtype=` | `auto` | `fp8` enables static attention quantization |

`auto` means "not set" — only forward the flag to `quantize.py` when the value
differs from `auto`. Collect optional flags in a bash **array**, never a string:

```bash
EXTRA_ARGS=()
if [[ "$STATIC_KV_DTYPE" != "auto" ]]; then
  EXTRA_ARGS+=(--static_kv_dtype "$STATIC_KV_DTYPE")
fi
python quantize.py ... "${EXTRA_ARGS[@]}"
```

Under `set -e`, do not use `[[ cond ]] && CMD` as the last statement of a block —
a false condition exits the script. Use `if`.

Variable naming: `DTYPE`, `INPUT_MODEL`, `OUTPUT_MODEL`, `EXPORT_FORMAT`,
`STATIC_KV_DTYPE`, `STATIC_ATTENTION_DTYPE`.

### quantize.py

Argument names mirror the rest of the repo, not the shell flag names:

| Shell flag | `quantize.py` argument |
| --- | --- |
| `--input_model=` | `--model_name_or_path` |
| `--output_model=` | `--export_path` |
| `--dtype=` | `--dtype` |
| `--export_format=` | `--export_format` |

Static quantization dtypes use `choices=["fp8", "float8_e4m3fn"]` and
`default=None`.

Structure: a module-level preset dict keyed by `--dtype`, a `build_config(args)`
returning `AutoRoundConfig`, then `prepare` / `convert`.

```python
_PRESET_CONFIG = {
    "mxfp4": {"scheme": "MXFP4", "layer_config": None},
    "mxfp8": {"scheme": "MXFP8", "layer_config": None},
}
```

Notes:
- `llm_compressor` export does not support `_RCEIL` schemes — strip the suffix
  when `export_format != "auto_round"`.
- Set `iters=0` for model-free / RTN-style presets.
- Static KV or attention quantization needs the real model, so `model_free`
  must be `False` when either is set.

### run_benchmark.sh

This is a **thin wrapper** (20–40 lines) over
`benchmark/lm_eval/run_lm_eval.sh`. It must not parse user arguments — pass
`"$@"` straight through. It only exports model-family specific settings:

```bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

export DEFAULT_SCHEME="mxfp8"
export ATTENTION_BACKEND_FP8KV="FLASHINFER"

bash "${REPO_ROOT}/benchmark/lm_eval/run_lm_eval.sh" "$@"
```

Hooks understood by the shared driver:

| Variable | Purpose |
| --- | --- |
| `DEFAULT_SCHEME` | default value for `--scheme` |
| `ATTENTION_BACKEND_FP8KV` | backend when `--static_kv_dtype=fp8` (MLA models use `FLASHINFER_MLA`) |
| `ATTENTION_BACKEND_FP8ATTN` | backend when `--static_attention_dtype=fp8` (`TRITON_ATTN` takes a different code path) |
| `FLASHINFER_WORKSPACE_SIZE` | `VLLM_AR_FLASHINFER_WORKSPACE_BUFFER_SIZE` |
| `EXTRA_SERVE_ARGS` | extra flags for `vllm serve` |
| `EXTRA_LM_EVAL_MODEL_ARGS` | extra `,key=value` pairs for `--model_args` |
| `ROPE_SCALING_JSON` | enables rope scaling on `vllm serve` |
| `SKIP_ROPE_SCALING_PATTERN` | model name substring that disables rope scaling |

Anything task-driven (new benchmark suite, chat-template task, long-context
routing) belongs in the shared driver, **not** in the wrapper. Anything that
depends on the model architecture belongs in the wrapper.

Do not re-implement argument parsing, tensor-parallel detection, server
lifecycle, or the `--scheme` environment matrix — the shared driver owns them.
Tensor parallel size is always inferred from `CUDA_VISIBLE_DEVICES`.

## README

Mirror the structure of `examples/pytorch/llm/qwen/README.md`: quantization
commands per dtype, then evaluation commands. Use `--key=value` in every example
command, and prefix eval commands with `CUDA_VISIBLE_DEVICES=...`.

## Validation

```bash
bash -n examples/pytorch/llm/<model>/run_quant.sh
bash -n examples/pytorch/llm/<model>/run_benchmark.sh
python -c "import ast; ast.parse(open('examples/pytorch/llm/<model>/quantize.py').read())"
```

Smoke-test argument routing without a GPU by putting a stub `lm_eval` on `PATH`:

```bash
mkdir -p /tmp/fakebin && printf '#!/bin/bash\necho "[lm_eval] $*"\n' > /tmp/fakebin/lm_eval
chmod +x /tmp/fakebin/lm_eval
PATH=/tmp/fakebin:$PATH CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh --model_path=<dir>
```

New files under `neural_compressor/` need the Apache 2.0 header; example files
follow the same convention. Run `pre-commit run --files <changed files>` before
committing.
