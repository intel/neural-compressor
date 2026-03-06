import shutil

import pytest
import torch
import transformers


def is_xpu_available():
    return torch.xpu.is_available()


from neural_compressor.torch.quantization import (
    AutoRoundConfig,
    convert,
    prepare,
)
from neural_compressor.torch.utils import logger

torch.backends.__allow_nonbracketed_mutation_flag = True

try:
    import auto_round

    auto_round_installed = True
except ImportError:
    auto_round_installed = False


tagert_modules = ["QuantLinear", "QuantLinearGPTQ", "QuantLinearAWQ"]


@torch.no_grad()
def run_fn(model, dataloader):
    for data in dataloader:
        if isinstance(data, tuple) or isinstance(data, list):
            model(*data)
        elif isinstance(data, dict):
            model(**data)
        else:
            model(data)


@pytest.mark.skipif(not is_xpu_available(), reason="XPU is not available")
@pytest.mark.skipif(not auto_round_installed, reason="auto_round module is not installed")
class TestAutoRoundGPU:
    @pytest.mark.parametrize(
        "scheme", ["W4A16", "W2A16", "W3A16", "W8A16", "MXFP4", "MXFP8", "NVFP4", "FPW8A16", "FP8_STATIC"]
    )
    def test_scheme(self, scheme):
        # INC API
        from transformers import AutoModelForCausalLM, AutoTokenizer

        fp32_model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
        )
        inp = torch.ones([1, 10], dtype=torch.long)
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", trust_remote_code=True)

        output_dir = "./saved_inc"
        quant_config = AutoRoundConfig(
            tokenizer=tokenizer,
            nsamples=32,
            seqlen=10,
            iters=1,
            device_map="xpu",
            scheme=scheme,
            export_format="auto_round",
            output_dir=output_dir,  # default is "temp_auto_round"
        )

        # quantizer execute
        model = prepare(model=fp32_model, quant_config=quant_config)
        convert(model)
        if scheme in ["FPW8A16"]:  # FPW8A16 loading not supported yet
            return
        inc_model = AutoModelForCausalLM.from_pretrained(
            output_dir,
        )
        out = inc_model(inp)[0]

        # AutoRound API
        from transformers import AutoModelForCausalLM, AutoTokenizer

        fp32_model = transformers.AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
        )
        inp = torch.ones([1, 10], dtype=torch.long)
        tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/opt-125m", trust_remote_code=True)
        from auto_round import AutoRound

        ar = AutoRound(
            model=fp32_model,
            tokenizer=tokenizer,
            nsamples=32,
            seqlen=10,
            iters=1,
            device_map="xpu",
            scheme=scheme,
        )
        quantized_model_path = "./saved_ar"
        ar.quantize_and_save(output_dir=quantized_model_path, inplace=True, format="auto_round")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path,
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        out_ar = model(inp)[0]
        assert torch.all(out_ar.eq(out))
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    @pytest.mark.parametrize("format", ["auto_awq", "auto_gptq", "llm_compressor"])
    def test_format(self, format):
        # INC API
        scheme = "W4A16" if format != "llm_compressor" else "MXFP4"
        from transformers import AutoModelForCausalLM, AutoTokenizer

        fp32_model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
        )
        inp = torch.ones([1, 10], dtype=torch.long)
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", trust_remote_code=True)

        output_dir = "./saved_inc"
        quant_config = AutoRoundConfig(
            tokenizer=tokenizer,
            nsamples=32,
            seqlen=10,
            iters=1,
            device_map="xpu",
            scheme=scheme,
            export_format=format,
            output_dir=output_dir,  # default is "temp_auto_round"
        )

        # quantizer execute
        model = prepare(model=fp32_model, quant_config=quant_config)
        inc_model = convert(model)
        assert inc_model is not None
        shutil.rmtree(output_dir, ignore_errors=True)

    def test_vlm_model(self):
        # INC API
        scheme = "W4A16"
        model_name = "Qwen/Qwen2-VL-2B-Instruct"
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

        fp32_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
        from neural_compressor.torch.algorithms.autoround import get_mllm_dataloader

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        output_dir = "./saved_inc"
        quant_config = AutoRoundConfig(
            tokenizer=tokenizer,
            nsamples=1,
            iters=1,
            seqlen=10,
            # quant_nontext_module=True,
            processor=processor,
            device_map="xpu:0",
            scheme=scheme,
            export_format="auto_round",
            output_dir=output_dir,  # default is "temp_auto_round"
        )

        # quantizer execute
        model = prepare(model=fp32_model, quant_config=quant_config)
        convert(model)
        inc_model = Qwen2VLForConditionalGeneration.from_pretrained(
            output_dir,
        )
        assert inc_model is not None
        shutil.rmtree(output_dir, ignore_errors=True)

    def test_quant_lm_head(self):
        # INC API
        scheme = "W4A16"
        model_name = "Qwen/Qwen3-8B"
        from transformers import AutoModelForCausalLM, AutoTokenizer

        fp32_model = AutoModelForCausalLM.from_pretrained(
            model_name,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        output_dir = "./saved_inc"
        quant_config = AutoRoundConfig(
            tokenizer=tokenizer,
            nsamples=1,
            seqlen=10,
            iters=0,  # rtn
            device_map="xpu",
            scheme=scheme,
            export_format="auto_round",
            output_dir=output_dir,  # default is "temp_auto_round"
            quant_lm_head=True,
        )

        # quantizer execute
        model = prepare(model=fp32_model, quant_config=quant_config)
        inc_model = convert(model)
        assert inc_model is not None
        shutil.rmtree(output_dir, ignore_errors=True)
