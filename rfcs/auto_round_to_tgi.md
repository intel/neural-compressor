

## Add Auto-Round Support

Hi, here is the INC team from Intel. Thank you for developing this amazing project.

### Motivation 

Our team have developed a new weight-only quantization algorithm called Auto-Round. It has achieved superior accuracy compared to [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [OmniQuant](https://arxiv.org/abs/2308.13137), and [HQQ](https://mobiusml.github.io/hqq_blog/) across 11 tasks, particularly excelling in low-bit quantization (e.g., 2-bits and 3-bits). Auto-Round supports quantization from 2 to 8 bits, involves low tuning costs, and imposes no additional overhead during inference. Key results are summarized below, with detailed information available in our [paper](https://arxiv.org/abs/2309.05516) and [code repository](https://github.com/intel/auto-round/blob/main/docs/acc.md).

| Config | Method | Mistral-7B | V2-7B     | V2-13B    | V2-70B    |
| ------ | ------ | ---------- | --------- | --------- | --------- |
| W4G-1  | FP16   | 63.3       | 57.98     | 61.42     | 66.12     |
|        | RTN    | 58.84      | 55.49     | 60.46     | 65.22     |
|        | HQQ    | 58.4       | 46.05     | 46.82     | 57.47     |
|        | Omni   | 60.52      | 56.62     | 60.31     | 65.8      |
|        | GPTQ   | 61.37      | 56.76     | 59.79     | 65.75     |
|        | AWQ    | 61.36      | 57.25     | 60.58     | **66.28** |
|        | Ours   | **62.33**  | **57.48** | **61.2**  | 66.27     |
|        | Ours\* | **62.64**  | **57.52** | **61.23** | 66.27     |
| W4G128 | FP16   | 63.3       | 57.98     | 61.42     | 66.12     |
|        | RTN    | 62.36      | 56.92     | 60.65     | 65.87     |
|        | HQQ    | **62.75**  | 57.41     | 60.65     | 66.06     |
|        | Omni   | 62.18      | 57.3      | 60.51     | 66.02     |
|        | GPTQ   | 62.32      | 56.85     | 61        | 66.22     |
|        | AWQ    | 62.16      | 57.35     | **60.91** | 66.23     |
|        | Ours   | 62.62      | **57.57** | 60.85     | **66.39** |
|        | Ours\* | **62.87**  | **57.97** | 60.9      | **66.41** |
| W3G128 | FP16   | 63.3       | 57.98     | 61.42     | 66.12     |
|        | RTN    | 58.2       | 53.81     | 58.57     | 64.08     |
|        | HQQ    | 59.33      | 54.31     | 58.1      | 64.8      |
|        | Omni   | 58.53      | 54.72     | 59.18     | 65.12     |
|        | GPTQ   | 59.91      | 54.14     | **59.58** | 65.08     |
|        | AWQ    | 59.96      | 55.21     | 58.86     | 65.12     |
|        | Ours   | 60.4**3**  | **56.68** | 59.44     | **65.3**1 |
|        | Ours\* | **60.96**  | **56.68** | **59.78** | **65.59** |
| W2G128 | FP16   | 63.3       | 57.98     | 61.42     | 66.12     |
|        | RTN    | 30.52      | 29.94     | 33.51     | 38.14     |
|        | HQQ    | 31.41      | 29.87     | 35.28     | 37.42     |
|        | Omni   | 32.17      | 40.74     | 46.55     | 51.31     |
|        | GPTQ   | 39.61      | 35.37     | 42.46     | 28.47     |
|        | AWQ    | 30.06      | 30.1      | 32.16     | 32.23     |
|        | Ours   | **52.71**  | **48.64** | **53.46** | **61.69** |
|        | Ours\* | **53.01**  | **50.34** | **54.16** | **61.77** |


We would like to contribute this quantization algorithm to TGI to enable users to:

1. Quantize a floating model using Auto-Round.
2. Perform inference using the quantized model.

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

Auto-Round is currently released as a pure [Python binary](https://pypi.org/project/auto-round/). The option prefer include `auto-round` in TGI's [`requirements_xx.txt`](https://github.com/huggingface/text-generation-inference/blob/main/server/requirements_cuda.txt) and utilize Auto-Round's API to obtain the quantized model.

Advantages:

- Minimal maintenance effort for TGI. We already integrated it into our [INC](https://github.com/intel/neural-compressor) project, and enabled the integration tests.
- Easy synchronization with new enhancements. As we continually improve the Auto-Round algorithm, updates can be effortlessly incorporated into TGI by updating the package version.

### Option 2: Porting All Source Code of Auto-Round into TGI

We are also willing to integrate all source code of Auto-Round directly into TGI.  

Advantages:

- No third-party dependency introduced.
- TCI maintainers have better control.

### 2. Inference AutoRound-Quantized Model
After obtaining the quantized model using Auto-Round, users can use it like other algorithms by specifying `--quantize` with `autoround`.

```bash
text-generation-launcher \
    --model-id INC/Llama-2-7b-Chat-Autoround \
    --trust-remote-code --port 8080 \
    --max-input-length 3072 --max-total-tokens 4096 --max-batch-prefill-tokens 4096 \
    --quantize autoround   # <------ select Auto-Round
```


Your feedback is important. Please feel free to comment on the options mentioned above or suggest additional approaches to ensure the most appropriate contribution method :). Thank you in advance!
