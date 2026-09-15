# Kimi / GLM AutoRound (INC prepare/convert)

This example demonstrates model-free quantization and evaluation for Kimi and GLM models.

## Quantization

The quantization flow is aligned with INC `prepare/convert` model-free API and uses:

- `scheme=MXFP4`
- `format=llm_compressor`
- `ignore_layers=shared_experts,self_attn,mlp.gate_proj,mlp.up_proj,mlp.down_proj`

For GLM MoE mixed setup, you can keep BF16 globally and quantize experts to MXFP4 via `layer_config`.

### Quick Start

```bash
cd examples/pytorch/nlp/huggingface_models/language-modeling/quantization/auto_round/kimi

bash run_quant.sh \
  --dtype=mxfp4 \
  --input_model=/workspace/models/moonshotai/Kimi-K2.6 \
  --output_model=/workspace/models/moonshotai/Kimi-K2.6-MXFP4
```

Equivalent Python command:

```bash
python quantize.py \
  --dtype mxfp4 \
  --input_model /workspace/models/moonshotai/Kimi-K2.6 \
  --output_model /workspace/models/moonshotai/Kimi-K2.6-MXFP4 \
  --ignore_layers shared_experts,self_attn,mlp.gate_proj,mlp.up_proj,mlp.down_proj \
  --format llm_compressor
```

### GLM Example (same folder)

You can run GLM in the same example with `scheme` override + `layer_config`:

```bash
bash run_quant.sh \
  --dtype=mxfp4 \
  --input_model=zai-org/GLM-5.2 \
  --output_model=/workspace/models/zai-org/GLM-5.2-MXFP4 \
  --format=llm_compressor \
  --scheme=BF16 \
  --layer_config='{mlp.experts:{scheme:MXFP4}}'
```

Equivalent Python command:

```bash
python quantize.py \
  --dtype mxfp4 \
  --input_model zai-org/GLM-5.2 \
  --output_model /workspace/models/zai-org/GLM-5.2-MXFP4 \
  --format llm_compressor \
  --scheme BF16 \
  --layer_config '{mlp.experts:{scheme:MXFP4}}'
```

`--layer_config` accepts both JSON and AutoRound shorthand:

- JSON: `'{"mlp.experts": {"scheme": "MXFP4"}}'`
- shorthand: `'{mlp.experts:{scheme:MXFP4}}'`

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
