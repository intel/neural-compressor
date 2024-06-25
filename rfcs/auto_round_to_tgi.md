

## Add Auto-Round Support

Hi, here is the INC team from Intel. Thank you for developing this amazing project.

### Motivation 

Our team have developed a new weight-only quantization algorithm called Auto-Round. It has achieved superior accuracy compared to [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [OmniQuant](https://arxiv.org/abs/2308.13137), and [HQQ](https://mobiusml.github.io/hqq_blog/) across 11 tasks, particularly excelling in low-bit quantization (e.g., 2-bits and 3-bits). Auto-Round supports quantization from 2 to 8 bits, involves low tuning costs, and imposes no additional overhead during inference. Key results are summarized below, with detailed information available in our [paper](https://arxiv.org/abs/2309.05516), [code repository](https://github.com/intel/auto-round/blob/main/docs/acc.md) and Hugging Face [low-bit quantization leaderboard](https://huggingface.co/spaces/Intel/low_bit_open_llm_leaderboard).

![alt text](Autoround-res.png)


We would like to contribute this quantization algorithm to TGI to enable users to:

1. Quantize a floating model using Auto-Round.
2. Perform inference with an AutoRound-quantized model.

### 1. Quantize Floating Model Using Auto-Round

Extend the current `quantize` API and add `method` as a new argument to select different algorithms. Users can utilize it as follows:

```bash
text-generation-server quantize \
    --MODEL_ID path/to/float/model/\
    --OUTPUT_DIR /path/to/save/quantized/model \
    --method autoround # <--- select the different methods, such as `gptq`, `autoround`
```

<!-- https://github.com/huggingface/text-generation-inference/blob/11ea9ce002e796cc59714950b557b4021cbebc58/server/text_generation_server/cli.py#L300-L319 -->

We propose two options to implement it:

#### Option 1: Adding Auto-Round as a New Python Dependency (Recommended)

Auto-Round is currently released as a pure [Python binary](https://pypi.org/project/auto-round/). The option prefers include `auto-round` in TGI's [`requirements_xx.txt`](https://github.com/huggingface/text-generation-inference/blob/main/server/requirements_cuda.txt) and utilizing Auto-Round's API to obtain the quantized model.

Advantages:

- Minimal maintenance effort for TGI. We already integrated it into our [INC](https://github.com/intel/neural-compressor) project, and enabled the integration tests.
- Easy synchronization with new enhancements. As we continually improve the Auto-Round algorithm, updates can be effortlessly incorporated into TGI by updating the package version.

#### Option 2: Porting All Source Code of Auto-Round into TGI

We are also willing to integrate all source code of Auto-Round directly into TGI.  

Advantages:

- No third-party dependency introduced.
- TGI maintainers have better control.

The overview of the call flow for the two options above is as follows:
```python
# tgi/server/text_generation_server/layers/autoround/quantize.py

def quantize(
    model_id: str,
    bits: int,
    groupsize: int,
    output_dir: str,
    revision: str,
    trust_remote_code: bool,
    ...
):
    # Load model...
    model = ...
    # Quantize model using auto-round
    # Import autoround from auto-round package for Option1
    # Import autoround from the current folder for Option2
    import autoround 
    rounder = autoround.AutoRound(model, ...)
    rounder.save_quantized(output_dir)

```

### 2. Perform inference with an AutoRound-quantized model.

we support exporting it in several formats compatible with high-speed, low-bits kernels such as `Exllama v1`, `Exllama v2`, and Triton kernels from AutoGPTQ for Nvidia GPUs, as well as QBits Kernel, W4A16 kernel from IPEX for Intel CPUs. We propose extending the current `text-generation-launcher` API to include `autoround` as a new option under `--quantize`. We propose supporting both loading an already quantized model and quantizing it on the fly. Users can utilize it as follows:

```bash
text-generation-launcher \
    --model-id INC/Llama-2-7b-Chat-Autoround \ # or float model, like meta-llama/Llama-2-17b-hf
    --trust-remote-code --port 8080 \
    --max-input-length 3072 --max-total-tokens 4096 --max-batch-prefill-tokens 4096 \
    --quantize autoround   # <------ select Auto-Round
```


Your feedback is important. Please feel free to comment on the options mentioned above or suggest additional approaches to ensure the most appropriate contribution method :). Thank you in advance!
