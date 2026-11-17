# Kimi / GLM AutoRound (INC prepare/convert)

This example demonstrates model-free quantization and evaluation for Kimi and GLM models.

## Quantization

The quantization flow is aligned with INC `prepare/convert` model-free API. The script
automatically detects the model type from `--input_model` name and applies the appropriate
quantization config:

| Model Type | Detection | Scheme | ignore_layers | layer_config |
|------------|-----------|--------|---------------|--------------|
| Kimi | name contains `kimi` | MXFP4 | `shared_experts,self_attn,mlp.gate_proj,mlp.up_proj,mlp.down_proj` | — |
| GLM | name contains `glm` | BF16 (base) + MXFP4 (experts) | — | `{"mlp.experts": {"scheme": "MXFP4"}}` |

### Quick Start

```bash
cd examples/pytorch/llm/kimi-glm

# Kimi
bash run_quant.sh \
  --dtype=mxfp4 \
  --input_model=moonshotai/Kimi-K2.6 \
  --output_model=/workspace/models/moonshotai/Kimi-K2.6-MXFP4

# GLM
bash run_quant.sh \
  --dtype=mxfp4 \
  --input_model=zai-org/GLM-5.2 \
  --output_model=/workspace/models/zai-org/GLM-5.2-MXFP4
```

Equivalent Python command:

```bash
python quantize.py \
  --dtype mxfp4 \
  --model_name_or_path moonshotai/Kimi-K2.6 \
  --export_path /workspace/models/moonshotai/Kimi-K2.6-MXFP4 \
  --model_type kimi \
  --export_format llm_compressor
```

## Evaluation

`run_benchmark.sh` is a thin wrapper over the shared drivers. Each task is routed to one path:
- `aime26`, `gpqa_diamond` and `math_500` run with `benchmark/evalscope/run_evalscope.sh` (starts `vllm serve` and evaluates through the OpenAI-compatible API with `evalscope`).
- All other tasks run with `benchmark/lm_eval/run_lm_eval.sh` (vLLM backend through `lm_eval`).

### Benchmark Quick Start

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh \
  --model_path=/workspace/models/moonshotai/Kimi-K2.6-MXFP4/Kimi-K2.6-mxfp-w4g32
```

Equivalent default command:

```bash
VLLM_QDQ=1 lm_eval --model vllm \
  --model_args pretrained=/workspace/models/moonshotai/Kimi-K2.6-MXFP4/Kimi-K2.6-mxfp-w4g32,tensor_parallel_size=8,data_parallel_size=1,max_model_len=8192,trust_remote_code=True,load_format=fastsafetensors \
  --tasks gsm8k,mmlu,piqa,hellaswag \
  --batch_size auto
```

Evaluate with evalscope:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh \
  --model_path=/workspace/models/moonshotai/Kimi-K2.6-MXFP4/Kimi-K2.6-mxfp-w4g32 \
  --tasks=aime26,gpqa_diamond,math_500
```

The evalscope path serves the model with `vllm serve <model> --tensor-parallel-size <N> --trust-remote-code --load-format fastsafetensors`.
`VLLM_QDQ=1` is only set on GPUs other than SM 10.x.

Arguments (tensor parallel size is inferred from `CUDA_VISIBLE_DEVICES`):

- `--model_path`: model path for vLLM (required).
- `--tasks`: comma-separated tasks (default: `gsm8k,mmlu,piqa,hellaswag`).
- `--static_kv_dtype` / `--static_attention_dtype`: static FP8 KV cache / attention (default: `auto`).

All options other than `--tasks` are passed unchanged to both drivers.

Outputs are written under `<model_name>-tp<N>-eval/` (lm_eval) and `<model_name>-tp<N>-evalscope/` (evalscope) in the current directory.
