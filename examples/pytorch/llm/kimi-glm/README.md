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
  --input_model moonshotai/Kimi-K2.6 \
  --output_model /workspace/models/moonshotai/Kimi-K2.6-MXFP4 \
  --model_type kimi \
  --format llm_compressor
```

## Evaluation

`run_benchmark.sh` is aligned with Llama benchmark style:

- Automatically infers `tensor_parallel_size` from `CUDA_VISIBLE_DEVICES`
- Exports `VLLM_QDQ=1`
- Uses vLLM backend through `lm_eval`

### Benchmark Quick Start

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,8 bash run_benchmark.sh \
  --model_path=/workspace/models/moonshotai/Kimi-K2.6-MXFP4/Kimi-K2.6-mxfp-w4g32
```

Equivalent default command:

```bash
VLLM_QDQ=1 lm_eval --model vllm \
  --model_args pretrained=/workspace/models/moonshotai/Kimi-K2.6-MXFP4/Kimi-K2.6-mxfp-w4g32,tensor_parallel_size=8,data_parallel_size=1,max_model_len=8192,trust_remote_code=True \
  --tasks gsm8k,mmlu,piqa,hellaswag \
  --batch_size auto
```

You can override defaults:

- `--tasks=<task1,task2,...>`
- `--batch_size=<auto|int>`
- `--max_model_len=<int>`
