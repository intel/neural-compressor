# vLLM INC NVFP4 Support

The `nvfp4_hw` package provides out-of-tree vLLM support for AutoRound NVFP4
checkpoints through the INC quantization path, without modifying the vLLM
installation. It supports both vLLM-native NVFP4 metadata and AutoRound
NVFP4_E5M3 (`nvfp4_v2`) checkpoints.

The plugin is loaded through the `vllm.general_plugins` entry-point mechanism.
After installation, users can load a supported NVFP4 checkpoint with vLLM or
with an evaluation tool that uses the vLLM backend.

## Supported Formats

### Native NVFP4

- AutoRound NVFP4 packed E2M1 weights.
- FP8 E4M3 per-group weight scales.
- Weight and input global scales.
- Dense Linear layers.
- MoE expert layers, including fused `w13` and `w2` weights.
- vLLM's native NVFP4 linear and MoE kernels.

### NVFP4_E5M3

- Packed E2M1 weights with unsigned E5M3 per-group scales.
- AutoRound `data_type: nvfp4_v2`, including per-layer `extra_config` overrides.
- Dense linear and fused MoE execution with group size 16.
- CuTe and pure-Torch reference activation QDQ implementations.
- vLLM FP4 Marlin weight repacking, GEMMs, expert routing, and top-k reduction.

## Installation

Install the plugin in editable mode:

```bash
pip install -e /path/to/vllm-qdq-plugin
```

The package declares the following vLLM general plugin entry point:

```text
inc_nvfp4 = nvfp4_hw.patch:register
```

## Usage

After installation, the plugin is loaded automatically by vLLM through the
`vllm.general_plugins` entry point. The following example loads an AutoRound
NVFP4 checkpoint through vLLM:

```python
from vllm import LLM

llm = LLM(
    model="/path/to/autoround-nvfp4-checkpoint",
    dtype="bfloat16",
)
```

No changes to the vLLM source code or model code are required. The plugin
recognizes the NVFP4 metadata, selects the dense or MoE implementation for
each layer, and passes the packed weights and scales to the corresponding
vLLM NVFP4 kernel.

## NVFP4_E5M3

### Format and Runtime Behavior

AutoRound checkpoints can use `data_type: nvfp4_v2` globally or override
selected layers such as `mlp.experts` in `extra_config`. The plugin accepts
both `auto_round:llm_compressor` and
`auto_round:llm_compressor_nvfp4_e5m3` packing formats.

The checkpoint stores raw `uint8 [N, K/2]` E2M1 payloads and
`uint8 [N, K/16]` UE5M3 block scales. Dense and MoE layers decode the scale
bytes during loading, then use vLLM's FP4 Marlin weight repacking, scale
processing, workspace management, GEMMs, expert routing, and top-k reduction.
Activations retain UE5M3 + E2M1 QDQ semantics before dense GEMMs and before
both MoE expert GEMMs. UE5M3 conversion uses bit-level round-to-nearest-even.

### QDQ Backend Selection

CuTe is selected automatically when all applicable requirements are met:

- NVIDIA CUDA input on an SM80 or newer GPU.
- NVIDIA CUTLASS DSL is installed.
- The input is contiguous.
- `K` is divisible by the activation group size.
- The group size is 16 or 32 for the activation QDQ kernel. Checkpoint-backed
  dense and MoE execution requires group size 16.

If a requirement is not met, the plugin emits a warning containing the reason,
uses the pure-Torch reference implementation, and warns that QDQ performance
will be lower. Set `VLLM_QDQ_CUTE=0` to force the reference implementation
without a fallback warning. Set it to `1` to explicitly request CuTe; leaving
it unset enables automatic selection.

### Local Model Commands

```bash
source /path/to/venv/bin/activate
MODEL_PATH=/path/to/nvfp4_e5m3_model
CUDA_VISIBLE_DEVICES=<idle-gpu> vllm serve "$MODEL_PATH" \
  --dtype bfloat16 --trust-remote-code

# Spawn-safe one-prompt verification
CUDA_VISIBLE_DEVICES=<idle-gpu> python scripts/test_nvfp4_ue5m3_model.py \
  "$MODEL_PATH"

# Default vLLM TorchDynamo/AOT and CUDA Graph configuration
CUDA_VISIBLE_DEVICES=<idle-gpu> VLLM_QDQ=1 vllm bench throughput \
  --model "$MODEL_PATH" \
  --dataset-name random --num-prompts 200 \
  --random-input-len 512 --random-output-len 128

# Force the reference activation QDQ implementation
CUDA_VISIBLE_DEVICES=<idle-gpu> VLLM_QDQ=1 VLLM_QDQ_CUTE=0 \
  vllm serve "$MODEL_PATH" --dtype bfloat16 --trust-remote-code
```

### Limitations

- Dense linear and fused MoE require BF16 activations.
- Checkpoint-backed dense and MoE execution requires group size 16.
- MoE does not support `apply_router_weight_on_input` or EPLB.
- The fused path supports vLLM TorchDynamo/AOT compilation and CUDA Graph capture.

### Validation Status

- Configuration routing tests passed for global and mixed `nvfp4_v2` settings.
- CuTe activation QDQ and weight dequantization have pure-Torch parity references
  and fake-aware custom-op boundaries for `torch.compile`.
- The standalone batched Triton MoE implementation matched its per-expert
  reference for group sizes 16 and 32 with cosine similarity above 0.9999 and
  relative L2 error below 1%. The production path now uses vLLM fused Marlin
  MoE instead.
- A real NVFP4_E5M3 model loaded and generated successfully with vLLM
  TorchDynamo/AOT and PIECEWISE/FULL CUDA Graph capture.

## Evaluation Results

### Native NVFP4

The following results were obtained with the vLLM backend. PIQA uses zero-shot
evaluation and GSM8K uses five-shot evaluation.

| Model | Task | Metric | Score |
|---|---|---|---:|
| Qwen3-8B-NVFP4 | PIQA | `acc` | 76.50% |
| Qwen3-8B-NVFP4 | PIQA | `acc_norm` | 77.20% |
| Qwen3-8B-NVFP4 | GSM8K | `exact_match` (strict) | 86.73% |
| Qwen3-8B-NVFP4 | GSM8K | `exact_match` (flexible) | 87.26% |
| Qwen3-30B-A3B-NVFP4 | PIQA | `acc` | 79.16% |
| Qwen3-30B-A3B-NVFP4 | PIQA | `acc_norm` | 80.14% |
| Qwen3-30B-A3B-NVFP4 | GSM8K | `exact_match` (strict) | 88.48% |
| Qwen3-30B-A3B-NVFP4 | GSM8K | `exact_match` (flexible) | 89.23% |

### MXFP4 and NVFP4_E5M3 Comparison

These internal results use Qwen3.6-35B-A3B and GSM8K v3 with 5-shot prompting,
the vLLM backend, automatic batch size, chat template enabled, thinking
disabled, tensor and data parallel size 1, maximum model length 8192, and expert
parallelism enabled. They are not independently verified benchmark claims.

| Format | Flexible exact match | Strict exact match | Prompts/s | Input tokens/s | Output tokens/s | Elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 baseline | 0.8863 +/- 0.0087 | 0.8772 +/- 0.0090 | N/A | N/A | N/A | N/A |
| MXFP4 | 0.8666 +/- 0.0094 | 0.8499 +/- 0.0098 | 9.71 | 10,346.38 | 1,511.55 | 2m 15s |
| NVFP4_E5M3 | 0.8726 +/- 0.0092 | 0.8560 +/- 0.0097 | 9.22 | 9,824.72 | 1,459.98 | 2m 23s |

NVFP4_E5M3 recovered 0.60 percentage points on both accuracy metrics relative
to MXFP4 in this run. MXFP4 was about 5% faster for prompt and input-token
throughput. Hardware, nondeterminism, runtime state, and dependency versions
can affect these measurements.

### Reproduce Quantization and Evaluation

Use the same commands for MXFP4 and NVFP4_E5M3, changing only `SCHEME`:

```bash
MODEL_PATH=/path/to/Qwen3.6-35B-A3B/
SCHEME=nvfp4_e5m3  # Use mxfp4 for MXFP4.
auto-round "$MODEL_PATH" \
  --format auto_round \
  --scheme "$SCHEME" \
  --output_dir "/path/to/output/qwen3.6-moe-${SCHEME}" \
  --ignore_layers shared_expert_gate \
  --model_free
```

```bash
SCHEME=nvfp4_e5m3
MODEL_PATH=/path/to/output/qwen3.6-moe-${SCHEME}
VLLM_QDQ=1 CUDA_VISIBLE_DEVICES=<idle-gpu> \
  lm_eval --model vllm \
  --model_args pretrained="$MODEL_PATH",tensor_parallel_size=1,data_parallel_size=1,max_model_len=8192,enable_expert_parallel=True,trust_remote_code=True,enable_thinking=False \
  --tasks gsm8k \
  --batch_size auto \
  --apply_chat_template
```

For the BF16 baseline, point `MODEL_PATH` to the original checkpoint and omit
`VLLM_QDQ=1`.

## Code Structure

| File | Purpose |
| --- | --- |
| `patch.py` | Register native NVFP4 and NVFP4_E5M3 metadata and scheme routing. |
| `inc_nvfp4_scheme.py` | Select the native NVFP4 dense or MoE implementation. |
| `inc_nvfp4_linear.py` | Load native NVFP4 weights and run dense linear layers. |
| `inc_nvfp4_moe.py` | Load native NVFP4 experts and run fused MoE. |
| `inc_nvfp4_ue5m3_scheme.py` | Select the NVFP4_E5M3 dense or MoE implementation. |
| `inc_nvfp4_ue5m3_linear.py` | Load and prepare NVFP4_E5M3 dense weights for vLLM FP4 Marlin. |
| `inc_nvfp4_ue5m3_moe.py` | Load and prepare NVFP4_E5M3 experts for vLLM fused Marlin MoE. |
| `fused_moe_ue5m3.py` | Provide a standalone batched Triton reference and benchmark implementation. |
| `../vllm_qdq_plugin/qdq/nvfp4_e5m3.py` | Select CuTe or reference UE5M3 + E2M1 activation QDQ. |
| `../vllm_qdq_plugin/qdq/nvfp4_e5m3_cute.py` | Register fake-aware CuTe custom ops for TorchDynamo. |
| `../vllm_qdq_plugin/qdq/cute_kernels.py` | Implement CuTe activation QDQ and packed-weight dequantization. |
| `../../tests/test_nvfp4_ue5m3.py` | Test configuration routing, dense execution, and fullgraph capture. |
| `../../tests/test_fused_moe_ue5m3.py` | Test MoE parity, fullgraph capture, and CUDA Graph replay. |
| `../../scripts/test_nvfp4_ue5m3_model.py` | Load a local model and run one prompt with spawn workers. |
