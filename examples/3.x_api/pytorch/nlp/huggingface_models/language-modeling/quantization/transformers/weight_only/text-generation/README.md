# Step-by-Step
We provide the inference benchmarking script `run_generation.py` for large language models, the default search algorithm is beam search with `num_beams = 4`. [Here](./llm_quantization_recipes.md) are some well accuracy and performance optimized models we validated, more models are working in progress.

# Quantization for CPU device

## Smooth Quantization
## Prerequisite​
### Create Environment​

The recomandation transformers version is 4.35.2, Pytorch and Intel-extension-for-pytorch version 2.2 are required. Model type "gptj", "opt", "llama" and "falcon" default used [ipex.optimize_transformers](https://github.com/intel/intel-extension-for-pytorch/blob/339bd251841e153ad9c34e1033ab8b2d936a1781/docs/tutorials/llm/llm_optimize_transformers.md) to accelerate the inference. Python version requests equal or higher than 3.9 due to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) evaluation library limitation, the dependent packages are listed in requirements, we recommend create environment as the following steps.

```bash
pip install -r requirements_sq.txt
```
> Note: If `ImportError: /lib64/libstdc++.so.6: version ``GLIBCXX_3.4.29`` not found` error raised when import intel-extension-for-pytorch, it is due to the high gcc library request, there is the solution to find the correct version.
> ```bash
> find $CONDA_PREFIX | grep libstdc++.so.6
> export LD_PRELOAD=<the path of libstdc++.so.6>:${LD_PRELOAD}
> ```
### Run
#### Performance
```shell
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
# fp32
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation_sq.py \
    --model <MODEL_NAME_OR_PATH> \
    --batch_size 1 \
    --benchmark

# quant and do benchmark.
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation_sq.py \
    --model <MODEL_NAME_OR_PATH> \
    --sq \
    --output_dir <SQ_MODEL_SAVE_PATH> \ # Default is "./saved_results."
    --benchmark \
    --batch_size 1
# load SQ model quantied by itrex and do benchmark.
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation_sq.py \
    --model <SQ_MODEL_SAVE_PATH> \
    --benchmark \
    --batch_size 1
# load SQ model quantied configure.json and do benchmark.
python run_generation_sq.py \
    --model <MODEL_NAME_OR_PATH> \
    --output_dir <SQ_MODEL_SAVE_PATH> \
    --restore_sq_model_from_json \
    --benchmark \
    --batch_size 1
```
#### Accuracy
```shell
# fp32
python run_generation_sq.py \
    --model <MODEL_NAME_OR_PATH> \
    --tasks lambada_openai,piqa,hellaswag \  # notice: no space.
    --device cpu \
    --batch_size 56

# quant and do accuracy.
python run_generation_sq.py \
    --model <MODEL_NAME_OR_PATH> \
    --sq \
    --output_dir <SQ_MODEL_SAVE_PATH> \ # Default is "./saved_results."
    --accuracy \
    --batch_size 56 

# load SQ model quantied by itrex and do benchmark.
python run_generation_sq.py \
    --model <SQ_MODEL_SAVE_PATH> \
    --accuracy \
    --batch_size 56 

# load SQ model quantied configure.json and do benchmark.
python run_generation_sq.py \
    --model <MODEL_NAME_OR_PATH> \
    --output_dir <SQ_MODEL_SAVE_PATH> \
    --restore_sq_model_from_json \
    --accuracy \
    --batch_size 56 

## Weight Only Quantization
## Prerequisite​
### Create Environment​
python version requests equal or higher than 3.9 due to [text evaluation library](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) limitation, the dependent packages are listed in requirements, we recommend create environment as the following steps.

```bash
pip install -r requirements_cpu_woq.txt
```

> Note: If `ImportError: /lib64/libstdc++.so.6: version ``GLIBCXX_3.4.29`` not found` error raised, it is due to the high gcc library request, there is the solution to find the correct version.
> ```bash
> find $CONDA_PREFIX | grep libstdc++.so.6
> export LD_PRELOAD=<the path of libstdc++.so.6>:${LD_PRELOAD}
> ```

### Run
We provide compression technologies such as `WeightOnlyQuant` with `Rtn/Awq/Teq/GPTQ/AutoRound` algorithms and `BitsandBytes`, `load_in_4bit` and `load_in_8bit` work on CPU device, besides we provide use ipex by `--use_ipex` to use intel extension for pytorch to accelerate the model, also provided use [neural-speed](https://github.com/intel/neural-speed) by `--use_neural_speed` to accelerate the optimized model, [here](https://github.com/intel/neural-speed/blob/main/docs/supported_models.md) is neural-speed supported list.
The followings are command to show how to use it.
#### Performance
```shell
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
# fp32
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generate_cpu_woq.py  \
    --model <MODEL_NAME_OR_PATH> \
    --batch_size 1 \
    --benchmark

# quant and do benchmark.
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generate_cpu_woq.py  \
    --model <MODEL_NAME_OR_PATH> \
    --woq \
    --woq_algo <ALGORITHM_NAME> \  # Default is "Rtn", "Awq", "Teq", "GPTQ", "AutoRound" are provided.
    --output_dir <WOQ_MODEL_SAVE_PATH> \  # Default is "./saved_results"
    --batch_size \
    --benchmark

# load WOQ model quantied by itrex and do benchmark.
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generate_cpu_woq.py  \
    --model <WOQ_MODEL_SAVE_PATH> \
    --benchmark

# load WOQ model quantied by itrex and do benchmark with neuralspeed.
# only support quantized with algorithm "Awq", "GPTQ", "AutoRound"
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> run_generate_cpu_woq.py \
    --model <WOQ_MODEL_SAVE_PATH> \
    --use_neural_speed \
    --benchmark

# load WOQ model from Huggingface and do benchmark.
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generate_cpu_woq.py \
    --model <MODEL_NAME_OR_PATH> \
    --benchmark

# load WOQ model from Huggingface and do benchmark with neuralspeed.
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generate_cpu_woq.py \
    --model <MODEL_NAME_OR_PATH> \
    --use_neural_speed \
    --benchmark
```
#### Accuracy
The accuracy validation is based from [lm_evaluation_harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.2/lm_eval/__main__.py).
```shell
# fp32
python run_generate_cpu_woq.py \
    --model <MODEL_NAME_OR_PATH> \
    --accuracy \
    --tasks lambada_openai,piqa,hellaswag \  # notice: no space.
    --device cpu \
    --batch_size 56

# quant and do accuracy.
python run_generate_cpu_woq.py \
    --model <MODEL_NAME_OR_PATH> \
    --woq \
    --woq_algo <ALGORITHM_NAME> \  # Default is "Rtn", "Awq", "Teq", "GPTQ", "AutoRound" are provided.
    --output_dir <WOQ_MODEL_SAVE_PATH> \
    --accuracy \
    --tasks lambada_openai,piqa,hellaswag \  # notice: no space.
    --batch_size 56 

# load WOQ model quantied by itrex and do benchmark.
python run_generate_cpu_woq.py \
    --model <WOQ_MODEL_SAVE_PATH> \
    --accuracy \
    --tasks lambada_openai,piqa,hellaswag \  # notice: no space.
    --batch_size 56 

# load WOQ model quantied by itrex and do benchmark with neuralspeed.
# only support quantized with algorithm "Awq", "GPTQ", "AutoRound"
python run_generate_cpu_woq.py \
    --model <WOQ_MODEL_SAVE_PATH> \
    --accuracy \
    --tasks lambada_openai,piqa,hellaswag \  # notice: no space.
    --device cpu \
    --batch_size 56
    --use_neural_speed

# load WOQ model from Huggingface and do benchmark.
python run_generate_cpu_woq.py \
    --model <MODEL_NAME_OR_PATH> \
    --accuracy \
    --tasks lambada_openai,piqa,hellaswag \  # notice: no space.
    --device cpu \
    --batch_size 56

# load WOQ model from Huggingface and do benchmark with neuralspeed.
python run_generate_cpu_woq.py \
    --model <MODEL_NAME_OR_PATH> \
    --accuracy \
    --tasks lambada_openai,piqa,hellaswag \  # notice: no space.
    --device cpu \
    --batch_size 56 \
    --use_neural_speed
```

# Quantization for GPU device
## Weight Only Quantization
>**Note**: 
> 1.  default search algorithm is beam search with num_beams = 1.
> 2. [ipex.optimize_transformers](https://github.com/intel/intel-extension-for-pytorch/blob/v2.1.10%2Bxpu/docs/tutorials/llm/llm_optimize_transformers.md) Support for the optimized inference of model types "gptj," "mistral," "qwen," and "llama" to achieve high performance and accuracy. Ensure accurate inference for other model types as well.
> 3. We provide compression technologies `WeightOnlyQuant` with `Rtn/GPTQ/AutoRound` algorithms and `load_in_4bit` and `load_in_8bit` work on intel GPU device.

## Prerequisite​
### Dependencies
Intel-extension-for-pytorch dependencies are in oneapi package, before install intel-extension-for-pytorch, we should install oneapi first. Please refer to [Installation Guide](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.10%2Bxpu) to install the OneAPI to "/opt/intel folder".

### Create Environment​
Pytorch and Intel-extension-for-pytorch version for intel GPU > 2.1 are required, python version requests equal or higher than 3.9 due to [text evaluation library](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) limitation, the dependent packages are listed in requirements_GPU.txt, we recommend create environment as the following steps. For Intel-exension-for-pytorch, we should install from source code now, and Intel-extension-for-pytorch will add weight-only quantization in the next version.

>**Note**: please install transformers==4.35.2.

```bash
pip install -r requirements_GPU.txt
pip install transformers==4.35.2
source /opt/intel/oneapi/setvars.sh
git clone https://github.com/intel/intel-extension-for-pytorch.git ipex-gpu
cd ipex-gpu
git submodule update --init --recursive
export USE_AOT_DEVLIST='pvc,ats-m150'
export BUILD_WITH_CPU=OFF

python setup.py install
```

## Run
The followings are command to show how to use it.

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
