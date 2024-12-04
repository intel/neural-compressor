FP8 Quantization
=======

1. [Introduction](#introduction)
2. [Supported Parameters](#supported-parameters)
3. [Get Start with FP8 Quantization](#get-start-with-fp8-quantization)
4. [Optimum-habana LLM example](#Optimum-habana LLM example) 
5. [VLLM example](#VLLM example) 

## Introduction

Float point 8 (FP8) is a promising data type for low precision quantization which provides a data distribution that is completely different from INT8 and it's shown as below.

<div align="center">
    <img src="./imgs/fp8_dtype.png" height="250"/>
</div>

Intel Gaudi2, also known as HPU, provides this data type capability for low precision quantization, which includes `E4M3` and `E5M2`. For more information about these two data type, please refer to [link](https://arxiv.org/abs/2209.05433).

Intel Neural Compressor provides general quantization APIs to leverage HPU FP8 capability. with simple  with lower memory usage and lower compute cost, 8 bit model

## Supported Parameters

<table class="tg"><thead>
  <tr>
    <th class="tg-fymr">Attribute</th>
    <th class="tg-fymr">Description</th>
    <th class="tg-fymr">Values</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky">fp8_config</td>
    <td class="tg-0pky">The target data type of FP8 quantization.</td>
    <td class="tg-0pky">E4M3 (default) - As Fig. 2<br>E5M2 - As Fig. 1.</td>
  </tr>
  <tr>
    <td class="tg-0pky">hp_dtype</td>
    <td class="tg-0pky">The high precision data type of non-FP8 operators.</td>
    <td class="tg-0pky">bf16 (default) - torch.bfloat16<br>fp16 - torch.float16.<br>fp32 - torch.float32.</td>
  </tr>
  <tr>
    <td class="tg-0pky">observer</td>
    <td class="tg-0pky">The observer to measure the statistics.</td>
    <td class="tg-0pky">maxabs (default), saves all tensors to files.</td>
  </tr>
  <tr>
    <td class="tg-0pky">allowlist</td>
    <td class="tg-0pky">List of nn.Module names or types to quantize. When setting an empty list, all the supported modules will be quantized by default. See Supported Modules. Not setting the list at all is not recommended as it will set the allowlist to these modules only: torch.nn.Linear, torch.nn.Conv2d, and BMM.</td>
    <td class="tg-0pky">Default = {'names': [], 'types': <span title=["Matmul","Linear","FalconLinear","KVCache","Conv2d","LoRACompatibleLinear","LoRACompatibleConv","Softmax","ModuleFusedSDPA","LinearLayer","LinearAllreduce","ScopedLinearAllReduce","LmHeadLinearAllreduce"]>FP8_WHITE_LIST}</span></td>
  </tr>
  <tr>
    <td class="tg-0pky">blocklist</td>
    <td class="tg-0pky">List of nn.Module names or types not to quantize. Defaults to empty list, so you may omit it from the config file.</td>
    <td class="tg-0pky">Default = {'names': [], 'types': ()}</td>
  </tr>
  <tr>
    <td class="tg-0pky">mode</td>
    <td class="tg-0pky">The mode, measure or quantize, to run HQT with.</td>
    <td class="tg-0pky">MEASURE - Measure statistics of all modules and emit the results to dump_stats_path.<br>QUANTIZE - Quantize and run the model according to the provided measurements.<br>AUTO (default) - Select from [MEASURE, QUANTIZE] automatically.</td>
  </tr>
  <tr>
    <td class="tg-0pky">dump_stats_path</td>
    <td class="tg-0pky">The path to save and load the measurements. The path is created up until the level before last "/". The string after the last / will be used as prefix to all the measurement files that will be created.</td>
    <td class="tg-0pky">Default = "./hqt_output/measure"</td>
  </tr>
  <tr>
    <td class="tg-0pky">scale_method</td>
    <td class="tg-0pky">The method for calculating the scale from the measurement.</td>
    <td class="tg-0pky">- unit_scale - Always use scale of 1.<br>- hw_aligned_single_scale - Always use scale that's aligned to the corresponding HW accelerated scale.<br>- maxabs_hw (default) - Scale is calculated to stretch/compress the maxabs measurement to the full-scale of FP8 and then aligned to the corresponding HW accelerated scale.<br>- maxabs_pow2 - Scale is calculated to stretch/compress the maxabs measurement to the full-scale of FP8 and then rounded to the power of 2.<br>- maxabs_hw_opt_weight - Scale of model params (weights) is chosen as the scale that provides minimal mean-square-error between quantized and non-quantized weights, from all possible HW accelerated scales. Scale of activations is calculated the same as maxabs_hw.<br>- act_maxabs_pow2_weights_pcs_opt_pow2 - Scale of model params (weights) is calculated per-channel of the params tensor. The scale per-channel is calculated the same as maxabs_hw_opt_weight. Scale of activations is calculated the same as maxabs_pow2.<br>- act_maxabs_hw_weights_pcs_maxabs_pow2 - Scale of model params (weights) is calculated per-channel of the params tensor. The scale per-channel is calculated the same as maxabs_pow2. Scale of activations is calculated the same as maxabs_hw.</td>
  </tr>
  <tr>
    <td class="tg-0pky">measure_exclude</td>
    <td class="tg-0pky">If this attribute is not defined, the default is OUTPUT. Since most models do not require measuring output tensors, you can exclude it to speed up the measurement process.</td>
    <td class="tg-0pky">NONE - All tensors are measured.<br>OUTPUT (default) - Excludes measurement of output tensors.</td>
  </tr>
</tbody></table>

## Get Start with FP8 Quantization
[Demo Usage](https://github.com/intel/neural-compressor?tab=readme-ov-file#getting-started)    
[Computer vision example](../../../examples/3.x_api/pytorch/cv/fp8_quant)

## Optimum-habana LLM example
### Overview
[Optimum](https://huggingface.co/docs/optimum) is an extension of Transformers that provides a set of performance optimization tools to train and run models on targeted hardware with maximum efficiency.    
[Optimum-habana](https://github.com/huggingface/optimum-habana) is the interface between the Transformers, Diffusers libraries and Intel Gaudi AI Accelerators (HPU). It provides higher performance based on modified modeling files, and utilizes Intel Neural Compressor for FP8 quantization internally,  [running-with-fp8](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation#running-with-fp8)    
![](./optimum-habana.png)
### Installation
Refer to [optimum-habana, install-the-library-and-get-example-scripts](https://github.com/huggingface/optimum-habana?tab=readme-ov-file#install-the-library-and-get-example-scripts)    
Option to install from source,
```
git clone https://github.com/huggingface/optimum-habana
cd optimum-habana && git checkout v1.14.0 (change the version)
pip install -e .
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.18.0
cd examples/text-generation
pip install -r requirements_lm_eval.txt  (Option)
```
### Check neural_compressor code
> optimum-habana/examples/text-generation/utils.py
>> initialize_model() -> setup_model() -> setup_quantization() -> FP8Config/prepare()/convert() 

### FP8 KV cache
Introduction: [kv-cache-quantization in huggingface transformers](https://huggingface.co/blog/kv-cache-quantization)    

BF16 KVCache Code -> [Modeling_all_models.py -> KVCache()](https://github.com/huggingface/optimum-habana/blob/main/optimum/habana/transformers/models/modeling_all_models.py#L40)    

FP8 KVCache code trace, for example Llama models,    
> optimum-habana/optimum/habana/transformers/models/llama/modeling_llama.py     
>> GaudiLlamaForCausalLM()  -> self.model()
>>>    GaudiLlamaModel() -> forward() -> decoder_layer() ->  GaudiLlamaDecoderLayer() forward() -> pre_attn() -> pre_attn_forward() -> self.k_cache.update     

> neural_compressor/torch/algorithms/fp8_quant/_quant_common/helper_modules.py    
>> PatchedKVCache() -> update()    

Models list which support FP8 KV Cache,
```
microsoft/Phi-3-mini-4k-instruct
bigcode/starcoder2-3b
Qwen/Qwen2.5-7B-Instruct|
meta-llama/Llama-3.2-3B-Instruct
tiiuae/falcon-7b-instruct
mistralai/Mixtral-8x7B-Instruct-v0.1
EleutherAI/gpt-j-6b
mistralai/Mistral-Nemo-Instruct-2407
```

### Running with FP8
Refer to [here](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation#running-with-fp8). Change "--model_name_or_path" to be your model like "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "mistralai/Mixtral-8x7B-Instruct-v0.1" and so on.    
"--use_kv_cache" is to enable FP8 KV cache.

### Profiling
Add "--profiling_warmup_steps 5 --profiling_steps 2 --profiling_record_shapes" as args in the end of commandline of run_generation.py. Refer to [code](https://github.com/huggingface/optimum-habana/blob/c9e1c23620618e2f260c92c46dfeb163545ec5ba/optimum/habana/utils.py#L305).    

### FP8 Accuracy 
| Llama-2-7b-hf| fp8 & fp8 KVCache| bf16 w/o fp8 KVCache||
|---------------|---------|--------|--------|
| hellaswag     | 0.5691097390957977   | 0.5704043019318861    ||
| lambada_openai| 0.7360760721909567   | 0.7372404424607025 | |
| piqa          | 0.7850924918389554   | 0.7818280739934712 ||
| winogrande    | 0.6929755327545383   | 0.6929755327545383 ||

| Qwen2.5-7B-Instruct| fp8 & fp8 KVCache| bf16 w/o fp8 KVCache||
|---------------|---------|--------|--------|
| hellaswag     |  0.2539334793865764  |   0.2539334793865764    ||
| lambada_openai| 0.0   | 0.0 | |
| piqa          | 0.5391730141458106   | 0.5391730141458106 ||
| winogrande    | 0.4956590370955012  | 0.4956590370955012 ||

| Llama-3.1-8B-Instruct| fp8 & fp8 KVCache| bf16 w/o fp8 KVCache||
|---------------|---------|--------|--------|
| hellaswag     | 0.5934076877116112   |   0.5975901214897431    ||
| lambada_openai| 0.7230739375121289   | 0.7255967397632447 | |
| piqa          | 0.7932535364526659   | 0.8030467899891186 ||
| winogrande    | 0.7434885556432518  | 0.7371744277821626 ||


| Mixtral-8x7B-Instruct-v0.1| fp8 & fp8 KVCache| bf16 w/o fp8 KVCache||
|---------------|---------|--------|--------|
| hellaswag     | 0.25323640709022105   |   0.25323640709022105    ||
| lambada_openai| 0.0   | 0.0  | |
| piqa          | 0.528835690968444   | 0.528835690968444  ||
| winogrande    | 0.4956590370955012  | 0.4956590370955012 ||

## VLLM example
### Overview
### FP8 KV cache
### llama2/3, Qwen 2/2.5, Mixtral
### accuracy table

## Reference
https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation#running-with-fp8   
https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html     