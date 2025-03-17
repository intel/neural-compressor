Transformers-like API
=====

1. [Introduction](#introduction)

2. [Supported Algorithms](#supported-algorithms)

3. [Usage For Intel CPU](#usage-for-cpu)

4. [Usage For Intel GPU](#usage-for-intel-gpu)

5. [Examples](#examples)

## Introduction

Transformers-like API provides a seamless user experience of model compressions on Transformer-based models by extending Hugging Face transformers APIs, leveraging neural compressor existing weight-only quantization capability and replacing Linear operator with Intel® Extension for PyTorch.

## Supported Algorithms

| Support Device |  RTN  |  AWQ  |  TEQ |  GPTQ  | AutoRound |
|:--------------:|:----------:|:----------:|:----------:|:----:|:----:|
|     Intel CPU        |  &#10004;  |  &#10004;  |  &#10004;  |  &#10004;  |  &#10004;  |
|     Intel GPU        |  &#10004;  |  stay tuned  |  stay tuned  |  &#10004;  |  &#10004;  |

> Please refer to [weight-only quantization document](./PT_WeightOnlyQuant.md) for more details.


## Usage For CPU 

Our motivation is to improve CPU support for weight only quantization. We have extended the `from_pretrained` function so that `quantization_config` can accept [`RtnConfig`](https://github.com/intel/neural-compressor/blob/master/neural_compressor/transformers/utils/quantization_config.py#L243), [`AwqConfig`](https://github.com/intel/neural-compressor/blob/72398b69334d90cdd7664ac12a025cd36695b55c/neural_compressor/transformers/utils/quantization_config.py#L394), [`TeqConfig`](https://github.com/intel/neural-compressor/blob/72398b69334d90cdd7664ac12a025cd36695b55c/neural_compressor/transformers/utils/quantization_config.py#L464), [`GPTQConfig`](https://github.com/intel/neural-compressor/blob/72398b69334d90cdd7664ac12a025cd36695b55c/neural_compressor/transformers/utils/quantization_config.py#L298), [`AutoroundConfig`](https://github.com/intel/neural-compressor/blob/72398b69334d90cdd7664ac12a025cd36695b55c/neural_compressor/transformers/utils/quantization_config.py#L527) to implements conversion on the CPU.

### Usage examples for CPU device
quantization and inference with `RtnConfig`, `AwqConfig`, `TeqConfig`, `GPTQConfig`, `AutoRoundConfig` on CPU device.
```python
# RTN
from neural_compressor.transformers import AutoModelForCausalLM, RtnConfig

model_name_or_path = "MODEL_NAME_OR_PATH"
woq_config = RtnConfig(bits=4)
q_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=woq_config,
)

# AWQ
from neural_compressor.transformers import AutoModelForCausalLM, AwqConfig

model_name_or_path = "MODEL_NAME_OR_PATH"
woq_config = AwqConfig(bits=4)
q_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=woq_config,
)

# TEQ
from transformers import AutoTokenizer
from neural_compressor.transformers import AutoModelForCausalLM, TeqConfig

model_name_or_path = "MODEL_NAME_OR_PATH"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
woq_config = TeqConfig(bits=4, tokenizer=tokenizer)
q_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=woq_config,
)

# GPTQ
from transformers import AutoTokenizer
from neural_compressor.transformers import AutoModelForCausalLM, GPTQConfig

model_name_or_path = "MODEL_NAME_OR_PATH"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
woq_config = GPTQConfig(bits=4, tokenizer=tokenizer)
woq_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=woq_config,
)

# AutoRound
from transformers import AutoTokenizer
from neural_compressor.transformers import AutoModelForCausalLM, AutoRoundConfig

model_name_or_path = "MODEL_NAME_OR_PATH"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
woq_config = AutoRoundConfig(bits=4, tokenizer=tokenizer)
woq_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=woq_config,
)

# inference
from transformers import AutoTokenizer

prompt = "Once upon a time, a little girl"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)
gen_ids = q_model.generate(input_ids, **generate_kwargs)
gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
print(gen_text)
```

You can also save and load your quantized low bit model by the below code.

```python
# quant
from neural_compressor.transformers import AutoModelForCausalLM, RtnConfig

model_name_or_path = "MODEL_NAME_OR_PATH"
woq_config = RtnConfig(bits=4)
q_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=woq_config,
)

# save quant model
saved_dir = "SAVE_DIR"
q_model.save_pretrained(saved_dir)

# load quant model
loaded_model = AutoModelForCausalLM.from_pretrained(saved_dir)
```

## Usage For Intel GPU
Intel® Neural Compressor implement weight-only quantization for Intel GPU,(PVC/ARC/MTL/LNL) with [intel-extension-for-pytorch](https://github.com/intel/intel-extension-for-pytorch).

Now 4-bit/8-bit inference with `RtnConfig`, `GPTQConfig`, `AutoRoundConfig` are support on Intel GPU device.

We support experimental woq inference on Intel GPU,(PVC/ARC/MTL/LNL) with replacing Linear op in PyTorch. Validated models: meta-llama/Meta-Llama-3-8B, meta/llama-Llama-2-7b-hf, Qwen/Qwen-7B-Chat, microsoft/Phi-3-mini-4k-instruct.

Here are the example codes.

#### Prepare Dependency Packages
1. Install Oneapi Package  
The Oneapi DPCPP compiler is required to compile intel-extension-for-pytorch. Please follow [the link](https://www.intel.com/content/www/us/en/developer/articles/guide/installation-guide-for-oneapi-toolkits.html) to install the OneAPI to "/opt/intel folder".

2. Build and Install PyTorch and intel-extension-for-pytorch. Please follow [the link](https://intel.github.io/intel-extension-for-pytorch/index.html#installation).

3. Quantization Model and Inference
```python
import intel_extension_for_pytorch as ipex
from neural_compressor.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch

model_name_or_path = "Qwen/Qwen-7B-Chat"  # MODEL_NAME_OR_PATH
prompt = "Once upon a time, a little girl"
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

q_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="xpu", trust_remote_code=True)

# optimize the model with ipex, it will improve performance.
quantization_config = q_model.quantization_config if hasattr(q_model, "quantization_config") else None
q_model = ipex.optimize_transformers(
    q_model, inplace=True, dtype=torch.float16, quantization_config=quantizaiton_config, device="xpu"
)

output = q_model.generate(input_ids, max_new_tokens=100, do_sample=True)
print(tokenizer.batch_decode(output, skip_special_tokens=True))
```

> Note: If your device memory is not enough, please quantize and save the model first, then rerun the example with loading the model as below, If your device memory is enough, skip below instruction, just quantization and inference.

4. Saving and Loading quantized model
 * First step: Quantize and save model
```python
from neural_compressor.transformers import AutoModelForCausalLM, RtnConfig

model_name_or_path = "MODEL_NAME_OR_PATH"
woq_config = RtnConfig(bits=4)
q_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, quantization_config=woq_config, device_map="xpu", trust_remote_code=True，
)

# Please note, saving model should be executed before ipex.optimize_transformers function is called.
q_model.save_pretrained("saved_dir")
```
 * Second step: Load model and inference(In order to reduce memory usage, you may need to end the quantize process and rerun the script to load the model.)
```python
# Load model
loaded_model = AutoModelForCausalLM.from_pretrained("saved_dir", trust_remote_code=True)

# Before executed the loaded model, you can call ipex.optimize_transformers function.
quantization_config = q_model.quantization_config if hasattr(q_model, "quantization_config") else None
loaded_model = ipex.optimize_transformers(
    loaded_model, inplace=True, dtype=torch.float16, quantization_config=quantization_config, device="xpu"
)

# inference
from transformers import AutoTokenizer

prompt = "Once upon a time, a little girl"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)
gen_ids = loaded_model.generate(input_ids, **generate_kwargs)
gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
print(gen_text)
```

5. You can directly use [example script](https://github.com/intel/neural-compressor/blob/master/examples/3.x_api/pytorch/nlp/huggingface_models/language-modeling/quantization/transformers/weight_only/text-generation/run_generation_gpu_woq.py)
```bash
python run_generation_gpu_woq.py --woq --benchmark --model save_dir
```

>Note:
> * Saving quantized model should be executed before the optimize_transformers function is called.
> * The optimize_transformers function is designed to optimize transformer-based models within frontend Python modules, with a particular focus on Large Language Models (LLMs). It provides optimizations for both model-wise and content-generation-wise. The detail of `optimize_transformers`, please refer to [the link](https://github.com/intel/intel-extension-for-pytorch/blob/xpu-main/docs/tutorials/llm/llm_optimize_transformers.md).
>* The quantization process is performed on the CPU accelerator by default. Users can override this setting by specifying the environment variable `INC_TARGET_DEVICE`. Usage on bash: ```export INC_TARGET_DEVICE=xpu```.
>* For Linux systems, users need to configure the environment variables appropriately to achieve optimal performance. For example, set the OMP_NUM_THREADS explicitly. For processors with hybrid architecture (including both P-cores and E-cores), it is recommended to bind tasks to all P-cores using taskset.

## Examples

Users can also refer to [examples](https://github.com/intel/neural-compressor/blob/master/examples/3.x_api/pytorch/nlp/huggingface_models/language-modeling/quantization/transformers/weight_only/text-generation) on how to quantize a model with transformers-like api.
