# DeepSeek V4 AutoRound (INC prepare/convert)

This example demonstrates model-free quantization through INC API:

```python
from neural_compressor.torch.quantization import AutoRoundConfig, prepare, convert

config = AutoRoundConfig(
    model_free=True,
    scheme="MXFP4",
    ignore_layers="compressor,indexer.weights_proj",
    export_format="llm_compressor",
    output_dir="/path/to/output",
)
model = "/path/or/hf_model_name"
model = prepare(model, config)
model = convert(model)
```

## Requirements

Install dependencies before running quantization or evaluation:

```bash
uv pip install -U pip
uv pip install -U "git+https://github.com/intel/auto-round.git@main"
uv pip install -U evalscope lm_eval transformers datasets
uv pip install compressed-tensors --no-deps
bash <(curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/main/tools/install_deepgemm.sh)
uv pip install setuptools_rust setuptools_scm
VLLM_USE_PRECOMPILED=1 uv pip install git+https://github.com/xin3he/vllm-fork.git@support_deepseekv4_mxfp --no-build-isolation
```

## Quick Start

```bash
cd examples/pytorch/nlp/huggingface_models/language-modeling/quantization/auto_round/deepseekv4
bash run_quant.sh \
  --dtype=mxfp4_mixed \
  --input_model=/workspace/models/deepseek-ai/DeepSeek-V4-Flash \
  --output_model=/workspace/models/deepseek-ai/DeepSeek-V4-Flash-MXFP4-Mixed
```

Then run serving + evaluation in one command:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_evalscope.sh \
  --model /workspace/models/deepseek-ai/DeepSeek-V4-Flash-MXFP4-Mixed \
  --tp 2 \
  --port 8009 \
  --tasks piqa,hellaswag,gsm8k,mmlu_pro,math_500,mmlu,aime26,gpqa_diamond,ruler_qa_squad
  --temp 1.0
```

Equivalent vLLM defaults inside `run_evalscope.sh`:

```bash
SAFETENSORS_FAST_GPU=1 CUDA_VISIBLE_DEVICES=0,1 vllm serve <model> \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tensor-parallel-size 2 \
  --attention_config.use_fp4_indexer_cache=True \
  --port 8009 \
  --no-enable-flashinfer-autotune
```

If model basename is exactly `DeepSeek-V4-Flash` or `DeepSeek-V4-Pro` (without extra suffix),
`run_evalscope.sh` will also add (automatically):

```bash
--enable-expert-parallel --moe-backend deep_gemm_mega_moe
```

Mixed preset example:

```bash
bash run_quant.sh \
  --dtype=mxfp4_mixed \
  --input_model=/workspace/models/deepseek-ai/DeepSeek-V4-Flash \
  --output_model=/workspace/models/deepseek-ai/DeepSeek-V4-Flash-MXFP8
```

## CLI Arguments

- `--dtype`: quantization preset.
  - `mxfp4`: `scheme=MXFP4`
  - `mxfp4_mixed`: `scheme=MXFP8` + `layer_config={"ffn.experts": {"bits": 4, "data_type": "mx_fp"}}`
  - `mxfp8`: `scheme=MXFP8`
  - `w4a16`: `scheme=W4A16` + `layer_config={"wo_a": {"bits": 16}}`
- `--input_model`: HF model name or local model path.
- `--output_model`: output directory.
- `--format`: `auto_round` or `llm_compressor` (default: `llm_compressor`).
- `--ignore_layers`: comma-separated layer patterns (default: `compressor,indexer.weights_proj`).

`run_evalscope.sh` arguments:

- `--model`: model path for vLLM and evalscope.
- `--port`: vLLM API port (default: `8009`).
- `--temp`: generation temperature used by evalscope (default: `0`).
- `--skip_serve`: skip starting vLLM (use existing endpoint on the same `--port`).
- `--tp`: tensor parallel size for vLLM (default: `2`).
- `--kv-cache-dtype`: kv cache dtype for vLLM (default: `fp8`).
- `--block-size`: vLLM block size (default: `256`).

## Notes

- This flow is enabled only when:
  - `config` is `AutoRoundConfig`
  - `config.model_free=True`
  - `model` passed to `prepare/convert` is a `str` (model path or model name)
- The example uses `reloading=False` by default and saves quantized artifacts to `--output_model`.


---

## Agent Evaluation (run_agent.sh)

Run SWE-bench Pro, SWE-bench Verified, or MCP-Atlas against the quantized model
via a local vLLM server.

### One-time setup

```bash
# From within an activated environment (conda/venv):
BENCHMARK_DIR=$PWD bash setup_agent.sh [swebp|swe-verified|mcp-atlas]
```

Place the SWE-bench Pro Docker image patch at `patches/swebench_pro_image.patch`
before running `setup_agent.sh swebp`.

### Run

```bash
# SWE-bench Pro — smoke test (10 instances, TP=2)
CUDA_VISIBLE_DEVICES=0,1 TENSOR_PARALLEL_SIZE=2 \
bash run_agent.sh \
  --task      swebp \
  --model     /path/to/DeepSeek-V4-Flash \
  --port      8888 \
  --num-tasks 10 \  # only run 10 cases.
  --workers   2 \
  --tag       smoke_swebp

# SWE-bench Verified — full run (vLLM already running)
SWEBENCH_API_KEY=<your_key> \
bash run_agent.sh \
  --task      swe-verified \
  --port      8888 \
  --skip-serve \
  --workers   4 \
  --tag       run_verified_01

# MCP-Atlas — smoke test
CUDA_VISIBLE_DEVICES=0,1 TENSOR_PARALLEL_SIZE=2 \
bash run_agent.sh \
  --task      mcp-atlas \
  --model     /path/to/DeepSeek-V4-Flash \
  --port      8889 \
  --workers   5 \
  --tag       smoke_mcp
```
If the model basename is exactly `DeepSeek-V4-Flash` or `DeepSeek-V4-Pro`,
`--enable-expert-parallel --moe-backend deep_gemm_mega_moe` are added automatically.

### Arguments

`run_agent.sh` common arguments:

- `--task`: `swebp` | `swe-verified` | `mcp-atlas`
- `--model PATH`: model path; launches vLLM automatically when set
- `--port N`: vLLM API port (default: `8888`)
- `--served-name NAME`: vLLM served-model-name (default: `gpt-3.5-turbo`)
- `--max-model-len N`: vLLM max_model_len (default: `262144`)
- `--tag NAME`: label for output directory and logs
- `--skip-serve`: skip vLLM launch (use already-running server)

SWE-bench Pro / Verified:

- `--workers N`: parallel agent workers (default: `2`)
- `--step-limit N`: max steps per instance (default: `250`)
- `--num-tasks N`: limit to first N instances
- `--slice S:E`: exact slice, e.g. `0:20`
- `--redo`: re-run already-completed instances (swebp only)

MCP-Atlas:

- `--workers N`: parallel eval concurrency (default: `5`)
- `--num-tasks N`: limit to first N tasks
- `--sandbox-port P`: MCP sandbox Docker port (default: `1984`)
- `--harness-port P`: agent harness port (default: `3001`)
- `--skip-sandbox`: skip Docker sandbox startup
