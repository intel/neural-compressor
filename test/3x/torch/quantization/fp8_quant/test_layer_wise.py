import torch
import habana_frameworks.torch.core as htcore

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from neural_compressor.torch.quantization import FP8Config, convert, prepare, finalize_calibration
from neural_compressor.torch.utils import get_used_cpu_mem_MB


htcore.hpu_set_env()


def test_two_step_layer_wise():
    # layer-wise is based on memory mapping technique and https://github.com/huggingface/transformers/pull/31771
    # Workaround of [SW-208658]: Memory mapping is blocked unreasonably
    tmp_deterministic_algorithms_flag = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    model_name = "facebook/opt-350m"
    config = AutoConfig.from_pretrained(model_name)
    # requires transformers >= 4.43.0, torch_dtype=config.torch_dtype
    # facebook/opt-350m parameters on disk is in torch.float16 dtype
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

    # fp16 llama2-7b is converted to bf16 during quantization layer-by-layer.
    cpu_mem0 = get_used_cpu_mem_MB()
    new_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=config.torch_dtype, use_safetensors=True)
    cpu_mem2 = get_used_cpu_mem_MB()
    model = convert(new_model, qconfig)
    assert (cpu_mem2 - cpu_mem0) < 100, "model with memory mapping should use no more than 100MiB."
    torch.use_deterministic_algorithms(tmp_deterministic_algorithms_flag)
