# Step-by-Step
We provide a Transformers-like API for model compression using the `WeightOnlyQuant` with `Rtn/Awq/Teq/GPTQ/AutoRound` algorithms, besides we provide use ipex to use intel extension for pytorch to accelerate the model.
We provide the inference benchmarking script `run_generation.py` for large language models, the default search algorithm is beam search with `num_beams = 4`. [Here](./llm_quantization_recipes.md) are some well accuracy and performance optimized models we validated, more models are working in progress.

# Quantization for CPU device

## Prerequisite​
### Create Environment​
python version requests equal or higher than 3.9 due to [text evaluation library](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) limitation, the dependent packages are listed in requirements, we recommend create environment as the following steps.

```bash
pip install -r requirements_cpu_woq.txt
```


### Run
#### Performance
```shell
# fp32
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation_cpu_woq.py  \
    --model <MODEL_NAME_OR_PATH> \
    --batch_size 1 \
    --benchmark

# quant and do benchmark.
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation_cpu_woq.py  \
    --model <MODEL_NAME_OR_PATH> \
    --woq \
    --woq_algo <ALGORITHM_NAME> \  # Default is "Rtn", "Awq", "Teq", "GPTQ", "AutoRound" are provided.
    --output_dir <WOQ_MODEL_SAVE_PATH> \  # Default is "./saved_results"
    --batch_size \
    --benchmark

# load WOQ quantized model and do benchmark.
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation_cpu_woq.py  \
    --model <WOQ_MODEL_SAVE_PATH> \
    --benchmark

# load WOQ model from Huggingface and do benchmark.
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation_cpu_woq.py \
    --model <MODEL_NAME_OR_PATH> \
    --benchmark

```
#### Accuracy
The accuracy validation is based from [lm_evaluation_harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.3/lm_eval/__main__.py).
```shell
# fp32
python run_generation_cpu_woq.py \
    --model <MODEL_NAME_OR_PATH> \
    --accuracy \
    --tasks lambada_openai,piqa,hellaswag \  # notice: no space.
    --device cpu \
    --batch_size 56

# quant and do accuracy.
python run_generation_cpu_woq.py \
    --model <MODEL_NAME_OR_PATH> \
    --woq \
    --woq_algo <ALGORITHM_NAME> \  # Default is "Rtn", "Awq", "Teq", "GPTQ", "AutoRound" are provided.
    --output_dir <WOQ_MODEL_SAVE_PATH> \
    --accuracy \
    --tasks lambada_openai,piqa,hellaswag \  # notice: no space.
    --batch_size 56 

# load WOQ model quantied by itrex and do benchmark.
python run_generation_cpu_woq.py \
    --model <WOQ_MODEL_SAVE_PATH> \
    --accuracy \
    --tasks lambada_openai,piqa,hellaswag \  # notice: no space.
    --batch_size 56 

# load WOQ model quantied by itrex and do benchmark with neuralspeed.
# only support quantized with algorithm "Awq", "GPTQ", "AutoRound"
python run_generation_cpu_woq.py \
    --model <WOQ_MODEL_SAVE_PATH> \
    --accuracy \
    --tasks lambada_openai,piqa,hellaswag \  # notice: no space.
    --device cpu \
    --batch_size 56
    

# load WOQ model from Huggingface and do benchmark.
python run_generation_cpu_woq.py \
    --model <MODEL_NAME_OR_PATH> \
    --accuracy \
    --tasks lambada_openai,piqa,hellaswag \  # notice: no space.
    --device cpu \
    --batch_size 56

# load WOQ model from Huggingface and do benchmark with neuralspeed.
python run_generation_cpu_woq.py \
    --model <MODEL_NAME_OR_PATH> \
    --accuracy \
    --tasks lambada_openai,piqa,hellaswag \  # notice: no space.
    --device cpu \
    --batch_size 56 \
    
```

# Quantization for GPU device
>**Note**: 
> 1.  default search algorithm is beam search with num_beams = 1.
> 2. [ipex.optimize_transformers](https://github.com/intel/intel-extension-for-pytorch/blob/v2.1.10%2Bxpu/docs/tutorials/llm/llm_optimize_transformers.md) Support for the optimized inference of model types "gptj," "mistral," "qwen," and "llama" to achieve high performance and accuracy. Ensure accurate inference for other model types as well.
> 3. We provide compression technologies `WeightOnlyQuant` with `Rtn/GPTQ/AutoRound` algorithms and `load_in_4bit` and `load_in_8bit` work on intel GPU device.
> 4. The quantization process is performed on the CPU accelerator by default. Users can override this setting by specifying the environment variable `INC_TARGET_DEVICE`. Usage on bash: ```export INC_TARGET_DEVICE=xpu```.
> 5. For Linux systems, users need to configure the environment variables appropriately to achieve optimal performance. For example, set the OMP_NUM_THREADS explicitly. For processors with hybrid architecture (including both P-cores and E-cores), it is recommended to bind tasks to all P-cores using taskset.

## Prerequisite​
### Dependencies
Intel-extension-for-pytorch dependencies are in oneapi package, before install intel-extension-for-pytorch, we should install oneapi first. Please refer to [Installation Guide](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.10%2Bxpu) to install the OneAPI to "/opt/intel folder".

### Create Environment​
Pytorch and Intel-extension-for-pytorch version for intel GPU > 2.1 are required, python version requests equal or higher than 3.9 due to [text evaluation library](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) limitation, the dependent packages are listed in requirements_GPU.txt, we recommend create environment as the following steps. For Intel-exension-for-pytorch, we should install from source code now, and Intel-extension-for-pytorch will add weight-only quantization in the next version.

>**Note**: please install transformers==4.38.1.

```bash
pip install -r requirements_GPU.txt
pip install transformers==4.38.1 # llama use 4.38.1
git clone https://github.com/intel/intel-extension-for-pytorch.git ipex-gpu
cd ipex-gpu
git submodule update --init --recursive
export USE_AOT_DEVLIST='pvc,ats-m150'
export BUILD_WITH_CPU=OFF

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/:$LD_LIBRARY_PATH
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
export CCL_ROOT=${CONDA_PREFIX}
source /opt/intel/oneapi/setvars.sh --force
export LLM_ACC_TEST=1

python setup.py install
```

## Run
The following are command to show how to use it.

### 1. Performance
``` bash
# fp16
python run_generation_gpu_woq.py \
    --model EleutherAI/gpt-j-6b \
    --benchmark

# weightonlyquant
python run_generation_gpu_woq.py \
    --model EleutherAI/gpt-j-6b \
    --woq \
    --woq_algo <ALGORITHM_NAME> \  # Default is "Rtn", "GPTQ", "AutoRound" are provided.
    --benchmark
```
> Note: If your device memory is not enough, please quantize and save the model first, then rerun the example with loading the model as below, If your device memory is enough, skip below instruction, just quantization and inference.
```bash
# First step: Quantize and save model
python run_generation_gpu_woq.py \
    --model EleutherAI/gpt-j-6b \
    --woq \ # default quantize method is Rtn
    --woq_algo <ALGORITHM_NAME> \  # Default is "Rtn", "GPTQ", "AutoRound" are provided.
    --output_dir "saved_dir"

# Second step: Load model and inference
python run_generation_gpu_woq.py \
    --model "saved_dir" \
    --benchmark
```

### 2. Accuracy
```bash
# quantized model by following the steps above
python run_generation_gpu_woq.py \
    --model "saved_dir" \
    --accuracy \
    --tasks "lambada_openai"
```
