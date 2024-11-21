import copy
import shutil

import deepspeed
import torch
import transformers

from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import local_rank, world_size
from neural_compressor.torch.quantization import FP8Config, convert, load, prepare, save
from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import PatchedLinear


def get_hpu_used_mem():
    from habana_frameworks.torch.hpu import memory_stats
    import numpy as np
    torch.hpu.synchronize()
    mem_stats = memory_stats()
    return np.round(mem_stats["InUse"] / 1024**3, 3)


@torch.no_grad()
def calib_func(model):
    example_inputs = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long).to("hpu")
    for i in range(2):
        model(example_inputs)


def test_load_model_provided_by_neuralmagic():
    model_name_or_path = "neuralmagic/Qwen2-0.5B-Instruct-FP8"
    model = load(model_name_or_path, format="huggingface", device="hpu")
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


def test_multi_cards_save_load():
    name = "facebook/opt-350m"
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
    example_inputs = torch.tensor([[10, 20]], dtype=torch.long).to("hpu")

    # TODO: [SW-205970] update state_dict to save scalar scale format
    qconfig = FP8Config(fp8_config="E4M3", scale_format="const")
    model = prepare(model, qconfig)
    calib_func(model)
    model = convert(model)
    # save and load on multi cards
    save(model, "saved_results", format="huggingface")
    new_model = load("saved_results", format="huggingface", device="hpu")
    # check result
    fp8_out = model(example_inputs)[0]
    loaded_fp8_out = new_model(example_inputs)[0]
    assert (loaded_fp8_out == fp8_out).all(), "Loaded FP8 model output is different with raw FP8 output."
    print("saving and loading test passed.")
    shutil.rmtree("saved_results", ignore_errors=True)


if __name__ == "__main__":
    test_multi_cards_save_load()
    test_load_model_provided_by_neuralmagic()
