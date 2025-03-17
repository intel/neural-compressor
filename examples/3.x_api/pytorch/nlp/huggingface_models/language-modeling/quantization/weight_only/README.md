Weight-only quantization
===============

##  Prerequisite
```
# Installation
pip install -r requirements.txt
```

## Support status on HPU

Below is the current support status on Intel Gaudi AI Accelerator with PyTorch.

| woq_algo  | Status   | Validated Models                                                               |
|-----------|----------|--------------------------------------------------------------------------------|
|   GPTQ    | &#10004; | `meta-llama/Llama-2-7b-hf`<br/> `EleutherAI/gpt-j-6B`<br/> `facebook/opt-125m` |
| AutoRound | &#10004; | `meta-llama/Llama-2-7b-chat-hf`                                                |


> Notes:
> 1. `--gptq_actorder` is not supported by HPU.
> 2. Only support inference using uint4.
> 3. Double quantization is not supported on HPU.

## Support status on CPU

Below is the current support status on Intel® Xeon® Scalable Processor with PyTorch.


| woq_algo |   status |
|--------------|----------|
|       RTN      |  &#10004;  |
|       GPTQ     |  &#10004;  |
|       AutoRound|  &#10004;  |
|       AWQ      |  &#10004;  |
|       TEQ      |  &#10004;  |

> We validated the typical LLMs such as: `meta-llama/Llama-2-7b-hf`, `EleutherAI/gpt-j-6B`, `facebook/opt-125m`.


## Run

`run_clm_no_trainer.py` quantizes the large language models using the dataset [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k) calibration and validates datasets accuracy provided by lm_eval, an example command is as follows.

### Quantization (CPU & HPU)

```bash
python run_clm_no_trainer.py \
    --model meta-llama/Llama-2-7b-hf \
    --dataset NeelNanda/pile-10k \
    --quantize \
    --batch_size 8 \
    --woq_algo GPTQ \
    --woq_bits 4 \
    --woq_scheme asym \
    --woq_group_size 128 \
    --gptq_max_seq_length 2048 \
    --gptq_use_max_length \
    --output_dir saved_results
```

#### Quantize model with AutoRound on HPU
```bash
pip install -r requirements-autoround-hpu.txt

PT_ENABLE_INT64_SUPPORT=1 PT_HPU_LAZY_MODE=0 python run_clm_no_trainer.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dataset NeelNanda/pile-10k \
    --quantize \
    --batch_size 8 \
    --woq_algo AutoRound \
    --woq_bits 4 \
    --woq_scheme asym \
    --woq_group_size 128 \
    --autoround_seq_len 2048 \
    --output_dir saved_results
```
> [!TIP]
> We use `torch.compile`  to accelerate the quantization process of AutoRound on HPU.
> Please set the following environment variables before running the command:
> `PT_ENABLE_INT64_SUPPORT=1` and `PT_HPU_LAZY_MODE=0`.


### Evaluation (CPU)

```bash
# original model
python run_clm_no_trainer.py \
    --model meta-llama/Llama-2-7b-hf \
    --accuracy \
    --batch_size 8 \
    --tasks "lambada_openai"

python run_clm_no_trainer.py \
    --model meta-llama/Llama-2-7b-hf \
    --accuracy \
    --batch_size 8 \
    --tasks "lambada_openai" \
    --load \
    --output_dir saved_results
``` 

### Evaluation (HPU)

> Note: The SRAM_SLICER_SHARED_MME_INPUT_EXPANSION_ENABLED=false is an experimental flag which yields better performance for uint4, and it will be removed in a future release.

```bash
# original model
python run_clm_no_trainer.py \
    --model meta-llama/Llama-2-7b-hf \
    --accuracy \
    --batch_size 8 \
    --tasks "lambada_openai"

# quantized model
SRAM_SLICER_SHARED_MME_INPUT_EXPANSION_ENABLED=false ENABLE_EXPERIMENTAL_FLAGS=1 python run_clm_no_trainer.py \
    --model meta-llama/Llama-2-7b-hf \
    --accuracy \
    --batch_size 8 \
    --tasks "lambada_openai" \
    --load \
    --output_dir saved_results
```

For more information about parameter usage, please refer to [PT_WeightOnlyQuant.md](https://github.com/intel/neural-compressor/blob/master/docs/source/3x/PT_WeightOnlyQuant.md)
