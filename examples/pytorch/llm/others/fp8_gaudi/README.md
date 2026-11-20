# Step-by-step

Here we demonstrate FP8 quantization with some advanced techniques.
- block-wise calibration: reduce device memory requirement during calibration
- layer-wise quantization (base on memory mapping): reduce host memory requirement during quantization
- lm_eval evaluation for HPU: balance performance and memory usage, `--use_hpu_graph` is required.

Typically, quantization requires calibration with a high-precision model (such as bf16), which occupies a lot of device memory. Block-wise calibration splits the LLM into blocks and performs calibration one by one. Use ` --enable_block_wise_calibration` to enable this feature.

By default, This example loads model into shared memory from disk and loads to physical host memory layer-by-layer during quantization. The occupied physical host memory will be released in time.

In this example, you can measure and quantize`llama3.1/Meta-Llama-3.1-405B-Instruct` in torch.bfloat16 dtype with 8 Gaudi2 cards or even less, and host memory requirement is also low.

## Install deepspeed
Due to a known issue [microsoft/DeepSpeed/issues/3207](https://github.com/microsoft/DeepSpeed/issues/3207), we recommend installing deepspeed as follows.
```shell
git clone https://github.com/HabanaAI/DeepSpeed.git
cd DeepSpeed
git checkout 1.19.0
pip install -e .
cd ..
```

# Run

## meta-llama/Llama-2-70b-hf

```bash
# Measure, quantize and save
deepspeed --num_gpus 2 quantize.py --model_name_or_path meta-llama/Llama-2-70b-hf --quantize --save --save_path llama2_70b_fp8/
# With block-wise calibration, we can quantize 70b with one Gaudi2 cards
python quantize.py --model_name_or_path meta-llama/Llama-2-70b-hf --quantize --enable_block_wise_calibration --save --save_path llama2_70b_fp8/


# Load fp8 model and verify accuracy
python quantize.py --model_name_or_path llama2_70b_fp8/ --load --use_hpu_graph --accuracy
```

> Note: To get the best performance of fp8 model, please go to [optimum-habana](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation#running-with-fp8) to quantize the model. These advanced techniques will be upstreamed to optimum-habana soon.

## meta-llama/Llama-3.1-405B-Instruct

```bash
# Measure
deepspeed --num_gpus 8 quantize.py --model_name_or_path meta-llama/Llama-3.1-405B-Instruct --quantize --enable_block_wise_calibration --save --save_path llama3.1_405b_fp8/ 

# Load fp8 model and verify accuracy
deepspeed --num_gpus 8 quantize.py --model_name_or_path llama3.1_405b_fp8/ --load --use_hpu_graph --accuracy
```
