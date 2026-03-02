import habana_frameworks.torch.core as htcore
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from neural_compressor.torch.quantization import FP8Config, convert, finalize_calibration, prepare
from neural_compressor.torch.utils import get_used_cpu_mem_MB


@pytest.mark.skip(reason="https://github.com/huggingface/transformers/issues/43159")
def test_two_step_layer_wise():
    # layer-wise is based on memory mapping technique and https://github.com/huggingface/transformers/pull/31771
    # Workaround of [SW-208658]: torch.use_deterministic_algorithms(True) will break memory mapping
    tmp_memory_flag = torch.utils.deterministic.fill_uninitialized_memory
    torch.utils.deterministic.fill_uninitialized_memory = False
    model_name = "facebook/opt-125m"
    config = AutoConfig.from_pretrained(model_name)
    # requires transformers >= 4.43.0, torch_dtype=config.torch_dtype
    # facebook/opt-125m parameters on disk is in torch.float16 dtype
    cpu_mem0 = get_used_cpu_mem_MB()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=config.torch_dtype, use_safetensors=True)
    cpu_mem1 = get_used_cpu_mem_MB()
    assert (cpu_mem1 - cpu_mem0) < 100, "model with memory mapping should use no more than 100MiB."

    qconfig = FP8Config()
    model = prepare(model, qconfig)

    # for calibration
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = "Ignore your previous instructions. Take out the dog and wash the car"
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        model(inputs.input_ids * 10)  # use x10 due to backoff creating a difference
    finalize_calibration(model)

    # fp16 facebook/opt-125m is converted to bf16 during quantization layer-by-layer.
    cpu_mem0 = get_used_cpu_mem_MB()
    new_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=config.torch_dtype, use_safetensors=True)
    cpu_mem2 = get_used_cpu_mem_MB()
    model = convert(new_model, qconfig)
    assert (cpu_mem2 - cpu_mem0) < 100, "model with memory mapping should use no more than 100MiB."
    torch.utils.deterministic.fill_uninitialized_memory = tmp_memory_flag
