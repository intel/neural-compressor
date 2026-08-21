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
cd examples/pytorch/nlp/huggingface_models/language-modeling/quantization/auto_round/minimax_m2.7
bash run_quant.sh \
  --dtype=mxfp4_mixed \
  --input_model=MiniMaxAI/MiniMax-M2.7 \
  --output_model=~/models/minimax-m2.7-mxfp
```

### 2. Serve + Evaluate

```bash
CUDA_VISIBLE_DEVICES=3,4,5,6 bash run_evalscope.sh \
  --model ~/models/minimax-m2.7-mxfp \
  --tp 4 \
  --port 8001 \
  --tasks gpqa_diamond,aime25,gsm8k,piqa,hellaswag,live_code_bench
```

Equivalent vLLM command used inside `run_evalscope.sh`:

```bash
CUDA_VISIBLE_DEVICES=3,4,5,6 vllm serve ~/models/minimax-m2.7-mxfp \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --tool-call-parser minimax_m2 \
  --enable-auto-tool-choice \
  --reasoning-parser minimax_m2 \
  --served-model-name minimax-m2.7 \
  --max-model-len 102400 \
  --max-num-seqs 1024 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --port 8001
```

## CLI Arguments

### `run_quant.sh`

- `--dtype`: quantization preset.
  - `mxfp4_mixed`: `scheme=MXFP8` globally + `block_sparse_moe` layers use `scheme=MXFP4`
  - `mxfp8`: `scheme=MXFP8` globally
  - `mxfp4`: `scheme=MXFP4` globally
- `--input_model`: HF model name or local model path.
- `--output_model`: output directory.
- `--format`: `auto_round` or `llm_compressor` (default: `llm_compressor`).

### `run_evalscope.sh`

- `--model`: quantized model path for vLLM and evalscope.
- `--port`: vLLM API port (default: `8001`).
- `--temp`: generation temperature (default: `1.0`).
- `--tp`: tensor parallel size for vLLM (default: `4`).
- `--max-model-len`: max context length (default: `102400`).
- `--served-model-name`: served model name alias (default: `minimax-m2.7`).
- `--tasks`: comma-separated subset of `gpqa_diamond,aime25,gsm8k,piqa,hellaswag,live_code_bench` (default: all).
- `--skip_serve`: skip starting vLLM (use existing endpoint on the same `--port`).

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
