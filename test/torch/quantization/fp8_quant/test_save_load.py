import shutil

import deepspeed
import pytest
import torch
import transformers

from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import PatchedLinear
from neural_compressor.torch.algorithms.fp8_quant.prepare_quant.prepare_model import get_local_rank, get_world_size
from neural_compressor.torch.quantization import FP8Config, convert, load, prepare, save
from neural_compressor.torch.utils import get_used_hpu_mem_MB


def get_model_param_buffers(model):
    tmp = {}
    for name, param in model.named_parameters():
        tmp[name] = param
    for name, buffer in model.named_buffers():
        tmp[name] = buffer
    return tmp


def compare_parameters_buffers(model1, model2, atol=1e-8):
    import torch

    dict1 = get_model_param_buffers(model1)
    dict2 = get_model_param_buffers(model2)
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    unique_keys_in_dict1 = keys1 - keys2
    unique_keys_in_dict2 = keys2 - keys1
    unique_keys = unique_keys_in_dict1.union(unique_keys_in_dict2)
    assert len(dict1) == len(dict2), (
        f"The number of parameters and buffers are different, {unique_keys}.\n"
        + f"unique_keys_in_model1: {unique_keys_in_dict1}\nunique_keys_in_model2: {unique_keys_in_dict2}\n"
    )
    for k, v in dict1.items():
        assert k in dict2, "k not in dict2"
        assert v.dtype == dict2[k].dtype, f"dtype of {k} is different.\n{v.dtype}\n{dict2[k].dtype}"
        assert torch.allclose(v.float(), dict2[k].float(), atol=atol), (
            f"{k} is different in model1 and model2.\n" + f"{v}\n" + f"{dict2[k]}\n"
        )


@torch.no_grad()
def calib_func(model):
    example_inputs = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long).to("hpu")
    for i in range(2):
        model(example_inputs)


def test_save_vllm_compatible_model():
    name = "Qwen/Qwen2-0.5B-Instruct"
    world_size = get_world_size()
    if world_size > 0:
        # Do not use random weights since multi-processes will get different weights for Embedding
        model = transformers.AutoModelForCausalLM.from_pretrained(name)
        ds_inference_kwargs = {
            "dtype": torch.bfloat16,
            "tensor_parallel": {"tp_size": world_size},
        }
        model = deepspeed.init_inference(model, **ds_inference_kwargs)
        model = model.module
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(name)
    model = model.eval()
    generation_config = transformers.GenerationConfig.from_model_config(model.config)
    qconfig = FP8Config(
        fp8_config="E4M3",
        scale_format="const",
        allowlist={"types": ["Linear", "LinearLayer", "LinearAllreduce", "KVCache", "VLLMKVCache"]},
        blocklist={"names": ["lm_head"]},
    )
    model = prepare(model, qconfig)
    calib_func(model)
    model = convert(model)
    save(model, "saved_results_qwen", format="vllm")
    # save tokenizer and generation_configs.
    generation_config.save_pretrained("saved_results_qwen")
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    tokenizer.save_pretrained("saved_results_qwen")
    shutil.rmtree("saved_results_qwen", ignore_errors=True)
    shutil.rmtree("nc_workspace", ignore_errors=True)


@pytest.mark.skip(reason="[SW-226589] Skip this test since the model was updated")
def test_load_model_provided_by_neuralmagic():
    world_size = get_world_size()
    model_name_or_path = "neuralmagic/Qwen2-0.5B-Instruct-FP8"
    hpu_mem0 = get_used_hpu_mem_MB()
    model = load(model_name_or_path, format="huggingface", device="hpu")
    hpu_mem1 = get_used_hpu_mem_MB()
    if world_size > 0:
        assert (hpu_mem1 - hpu_mem0) < 480, "The memory usage is too high."
    assert isinstance(model, torch.nn.Module)
    assert isinstance(model.model.layers[0].self_attn.q_proj, PatchedLinear)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    prompt = "There existed a little girl, who liked to have adventures."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("hpu")
    generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=1)
    gen_ids = model.generate(
        input_ids,
        max_new_tokens=5,
        **generate_kwargs,
    )
    assert isinstance(gen_ids, torch.Tensor)


def init_model(world_size):
    name = "stas/tiny-random-llama-2"
    dtype = torch.bfloat16
    if world_size > 0:
        # Do not use random weights since multi-processes will get different weights for Embedding
        model = transformers.AutoModelForCausalLM.from_pretrained(name)
        ds_inference_kwargs = {
            "dtype": dtype,
            "tensor_parallel": {"tp_size": world_size},
        }
        model = deepspeed.init_inference(model, **ds_inference_kwargs)
        model = model.module
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(name)
    model = model.to(dtype)
    model = model.eval()
    return model


@torch.no_grad()
@pytest.mark.parametrize("scale_method", ["maxabs_hw", "act_maxabs_hw_weights_pcs_maxabs_pow2"])
def test_default_save_load(scale_method):
    world_size = get_world_size()
    local_rank = get_local_rank()
    example_inputs = torch.tensor([[10, 20]], dtype=torch.long).to("hpu")
    model = init_model(world_size)
    # The default value of model.generation_config.max_length in transformers is 20
    model.generation_config.max_length = 2

    qconfig = FP8Config(fp8_config="E4M3", blocklist={"names": ["q_proj", "lm_head"]}, scale_method=scale_method)
    model = prepare(model, qconfig)
    calib_func(model)
    model = convert(model)
    # save and load on multi cards
    save_folder_name = "saved_results_" + str(scale_method)
    save(model, save_folder_name, format="huggingface")

    # load with original world size
    hpu_mem0 = get_used_hpu_mem_MB()
    new_model = load(save_folder_name, format="huggingface", device="hpu")
    hpu_mem1 = get_used_hpu_mem_MB()
    if world_size > 0:
        assert (hpu_mem1 - hpu_mem0) < 300, "The memory usage is too high."
    assert new_model.generation_config.max_length == 2, "The generation config is not loaded."
    # check result
    compare_parameters_buffers(model, new_model)
    fp8_out = model(example_inputs)[0]
    loaded_fp8_out = new_model(example_inputs)[0]
    assert (loaded_fp8_out == fp8_out).all(), "Loaded FP8 model output is different with raw FP8 output."

    # load with half of existing cards
    half_world_size = max(world_size // 2, 1)
    if world_size > 1 and local_rank < half_world_size:
        # world_size argument is not need if program is starting with correct world size
        # we pass it since we are dynamically changing world size
        hpu_mem0 = get_used_hpu_mem_MB()
        new_model = load(save_folder_name, format="huggingface", device="hpu", world_size=half_world_size)
        hpu_mem1 = get_used_hpu_mem_MB()
        if half_world_size > 1:
            assert (hpu_mem1 - hpu_mem0) < 300, "The memory usage is too high."
        assert new_model.generation_config.max_length == 2, "The generation config is not loaded."

        # quantize model for comparison
        model = init_model(half_world_size)
        qconfig = FP8Config(fp8_config="E4M3", blocklist={"names": ["q_proj", "lm_head"]}, scale_method=scale_method)
        model = prepare(model, qconfig)
        calib_func(model)
        model = convert(model)

        # check result, use higher atol since weights are re-quantized with all_reduced scale.
        compare_parameters_buffers(model, new_model, atol=0.005)
    # wait for all processes to finish
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    print("saving and loading test passed.")
    shutil.rmtree(save_folder_name, ignore_errors=True)


if __name__ == "__main__":
    """This script supports running on multi-cards, command: `deepspeed --num_gpus N test_save_load.py`"""
    test_save_vllm_compatible_model()
    test_load_model_provided_by_neuralmagic()
    # The abnormal half world_size test in test_default_save_load() will cause unexpected behavior.
    # So we test it at the bottom and only one scale_method is allowed at a time.
    for scale_method in ["act_maxabs_hw_weights_pcs_maxabs_pow2"]:
        test_default_save_load(scale_method)
