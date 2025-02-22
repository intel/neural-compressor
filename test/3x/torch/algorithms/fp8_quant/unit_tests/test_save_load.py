import torch
import habana_frameworks.torch.core as htcore
import pytest
import shutil

htcore.hpu_set_env()

from transformers import LlamaConfig, LlamaForCausalLM
from neural_compressor.torch.quantization import FP8Config, convert, prepare, save, load
from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import Matmul

torch.manual_seed(1)
torch.set_grad_enabled(False)


def get_model_param_buffers(model):
    tmp = {}
    for name, param in model.named_parameters():
        tmp[name] = param
    for name, buffer in model.named_buffers():
        tmp[name] = buffer
    return tmp


def compare_parameters_buffers(model1, model2):
    import torch
    dict1 = get_model_param_buffers(model1)
    dict2 = get_model_param_buffers(model2)
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    unique_keys_in_dict1 = keys1 - keys2
    unique_keys_in_dict2 = keys2 - keys1
    unique_keys = unique_keys_in_dict1.union(unique_keys_in_dict2)
    assert len(dict1) == len(dict2), f"The number of parameters and buffers are different, {unique_keys}.\n" + \
            f"unique_keys_in_model1: {unique_keys_in_dict1}\nunique_keys_in_model2: {unique_keys_in_dict2}\n"
    for k, v in dict1.items():
        assert k in dict2, "k not in dict2"
        assert v.dtype == dict2[k].dtype, f"dtype of {k} is differnt.\n{v.dtype}\n{dict2[k].dtype}"
        assert torch.allclose(v, dict2[k]), f"{k} is differnt in model1 and model2.\n" + f"{v}\n" + f"{dict2[k]}\n"


@pytest.mark.parametrize("scale_method", [
    "unit_scale", "hw_aligned_single_scale", "maxabs_hw", "maxabs_pow2",
    "maxabs_arbitrary", "maxabs_hw_opt_weight", "maxabs_pow2_opt_weight",
    # per-channel
    "act_maxabs_hw_weights_pcs_maxabs_pow2", "act_maxabs_hw_weights_pcs_opt_pow2",
    "act_maxabs_pow2_weights_pcs_maxabs_pow2", "act_maxabs_pow2_weights_pcs_opt_pow2",
])
@pytest.mark.parametrize("scale_format", ["const", "scalar"])
def test_save_load(scale_method, scale_format):
    config = LlamaConfig(hidden_size=128, num_attention_heads=2, num_hidden_layers=2, vocab_size=512)
    model = LlamaForCausalLM(config)
    model = model.to(torch.bfloat16)  # ensure all buffers are bf16 to avoid [SW-214576]
    model = model.eval()
    qconfig = FP8Config(
        scale_method=scale_method,
        scale_format=scale_format,
    )
    if scale_method in ["unit_scale", "hw_aligned_single_scale"]:
        model = convert(model, qconfig)
    else:
        model = prepare(model, qconfig)
        model(torch.tensor([[3, 4]]).to("hpu"))
        model = convert(model)
    # save and load on multi cards
    save(model, "saved_results", format="huggingface")
    new_model = load("saved_results", format="huggingface", device="hpu")
    compare_parameters_buffers(model, new_model)
    shutil.rmtree("saved_results", ignore_errors=True)
    # inference and compare
    htcore.hpu_inference_initialize(model, mark_only_scales_as_const=True)
    htcore.hpu_inference_initialize(new_model, mark_only_scales_as_const=True)
    example_input = torch.tensor([[5, 6]]).to("hpu")
    with torch.no_grad():
        out1 = model(example_input)[0].cpu()
        out2 = new_model(example_input)[0].cpu()
    assert (out1==out2).all(), \
            f"The output of the model is different after save and load with scale_method: {scale_method}"
