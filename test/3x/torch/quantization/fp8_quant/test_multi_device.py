import copy
import shutil

import deepspeed
import torch
import transformers

from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import local_rank, world_size
from neural_compressor.torch.quantization import FP8Config, convert, load, prepare, save


@torch.no_grad()
def calib_func(model):
    example_inputs = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long).to("hpu")
    for i in range(2):
        model(example_inputs)


def test_multi_cards_save_load():
    config = transformers.AutoConfig.from_pretrained("./model_configs/tiny_gptj.json")
    if world_size > 0:
        model = transformers.AutoModelForCausalLM.from_config(config)
        ds_inference_kwargs = {
            "dtype": torch.bfloat16,
            "tensor_parallel": {"tp_size": world_size},
        }
        model = deepspeed.init_inference(model, **ds_inference_kwargs)
        model = model.module
    else:
        model = transformers.AutoModelForCausalLM.from_config(config)
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
