import copy
import os
import shutil
from functools import lru_cache

import pytest
import torch
import transformers
from packaging.version import Version, parse
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig


@lru_cache(None)
def is_habana_framework_installed():
    """Check if Habana framework is installed.

    Only check for the habana_frameworks package without importing it to avoid
    initializing lazy-mode-related components.
    """
    from importlib.util import find_spec

    package_spec = find_spec("habana_frameworks")
    return package_spec is not None


def set_hpu_torch_compile_envs():
    if not is_habana_framework_installed():
        return None
    import torch._dynamo.config as dynamo_config
    import torch._inductor.config as inductor_config

    os.environ["PT_HPU_LAZY_MODE"] = "0"
    os.environ["PT_ENABLE_INT64_SUPPORT"] = "1"
    inductor_config.force_disable_caches = True
    dynamo_config.inline_inbuilt_nn_modules = True


# The `TestAutoRoundHPU` is expected to be run with `compile` mode,
# so set the HPU environment variables before importing INC.
if is_habana_framework_installed():
    set_hpu_torch_compile_envs()


from neural_compressor.torch.quantization import (
    AutoRoundConfig,
    convert,
    get_default_AutoRound_config,
    prepare,
    quantize,
)
from neural_compressor.torch.utils import logger

torch.backends.__allow_nonbracketed_mutation_flag = True

try:
    import auto_round

    auto_round_installed = True
except ImportError:
    auto_round_installed = False


tagert_modules = ["QuantLinear", "QuantLinearGPTQ", "QuantLinearAWQ"]


@pytest.mark.skipif(not is_habana_framework_installed(), reason="Habana framework is not installed")
@pytest.mark.skipif(os.getenv("PT_HPU_LAZY_MODE", "0") == "1", reason="Lazy mode is enabled")
@pytest.mark.skipif(not auto_round_installed, reason="auto_round module is not installed")
class TestAutoRoundHPU:
    @classmethod
    def setup_class(self):

        model_name = "TheBloke/Llama-2-7B-Chat-GPTQ"
        from neural_compressor.torch.algorithms.autoround import get_dataloader

        config = LlamaConfig(num_hidden_layers=2)
        with transformers.modeling_utils.no_init_weights():
            self.tiny_llama_model = AutoModelForCausalLM.from_config(config=config)

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.dataloader = get_dataloader(tokenizer, 32, dataset_name="NeelNanda/pile-10k", seed=42, bs=8, nsamples=10)
        self.inp = torch.ones([1, 10], dtype=torch.long)
        self.label = self.tiny_llama_model(self.inp)[0]

    @classmethod
    def teardown_class(self):
        shutil.rmtree("saved_results", ignore_errors=True)

    def setup_method(self, method):
        torch.compiler.reset()
        logger.info(f"Running TestAutoRound test: {method.__name__}")

    @pytest.mark.skip(reason="Disabled, see JIRA: https://jira.habana-labs.com/browse/SW-227554")
    def test_autoround_w4a8(self):
        fp32_model = copy.deepcopy(self.tiny_llama_model)
        quant_config = AutoRoundConfig(
            nsamples=32,
            seqlen=10,
            iters=2,
            scale_dtype="bf16",
            dtype="fp8_to_int_sym",
            act_bits=8,
            act_group_size=-1,
            act_dtype="fp8_sym",
            act_dynamic=False,
        )

        quant_config.set_local("lm_head", AutoRoundConfig(dtype="fp32"))
        logger.info(f"Test AutoRound with config {quant_config}")

        # prepare + convert API
        model = prepare(model=fp32_model, quant_config=quant_config)

        run_fn(model, self.dataloader)
        q_model = convert(model)
        assert q_model is not None, "Quantization failed!"
        # We quantize the model with compile mode, if we want to run the model directly,
        # we need use the compile mode as well.
        # We can use the lazy mode but need to restart the python process.
        from neural_compressor.torch.algorithms.weight_only.save_load import load

        model = load(
            model_name_or_path="temp_auto_round",
            original_model=copy.deepcopy(self.tiny_llama_model),
            device="hpu",
            format="huggingface",
        )
        print(f"loaded model {model}")
        from neural_compressor.torch.algorithms.mixed_low_precision.modules import HPUMixedPrecisionLinear

        has_hpu_mixed_precision_module = False
        for name, module in model.named_modules():
            if isinstance(module, HPUMixedPrecisionLinear):
                has_hpu_mixed_precision_module = True
                break
        assert has_hpu_mixed_precision_module, "loading compressed model failed."
        model.eval()
        model = model.to(torch.bfloat16)
        model = torch.compile(model, backend="hpu_backend")
        out = model(self.inp.to("hpu"))[0]
        print(f"out: {out}")
        assert out is not None, "Loading compressed model failed."

    def test_quant_lm_head(self):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM", trust_remote_code=True
        )

        quant_config = AutoRoundConfig(
            tokenizer=tokenizer,
            nsamples=32,
            seqlen=10,
            iters=1,
            amp=False,
            scale_dtype="fp32",
            quant_lm_head=True,
            group_size=32,
        )
        logger.info(f"Test AutoRound with config {quant_config}")
        text = "Replace me by any text you'd like."
        encoded_input = tokenizer(text, return_tensors="pt")
        model = prepare(model=model, quant_config=quant_config)
        q_model = convert(model)
        output = tokenizer.decode(q_model.generate(**encoded_input, max_new_tokens=10)[0])
        print(output)
        assert output is not None
        assert q_model.lm_head.__class__.__name__ in tagert_modules, "packing model failed."

    def test_int4_dtype(self):
        fp32_model = copy.deepcopy(self.tiny_llama_model)
        quant_config = AutoRoundConfig(dtype="int4", nsamples=32, seqlen=10, iters=1, amp=False, scale_dtype="fp32")
        logger.info(f"Test AutoRound with config {quant_config}")

        # prepare + convert API
        model = prepare(model=fp32_model, quant_config=quant_config)

        run_fn(model, self.dataloader)
        q_model = convert(model)
        _ = q_model(self.inp)  # inference
        assert q_model.model.layers[0].self_attn.k_proj.__class__.__name__ in tagert_modules, "packing model failed."

    def test_autoround_with_quantize_API(self):
        fp32_model = copy.deepcopy(self.tiny_llama_model)

        quant_config = AutoRoundConfig(scheme="W4A16", seqlen=10, iters=1, use_sym=False, amp=False, scale_dtype="fp32")
        logger.info(f"Test AutoRound with config {quant_config}")

        # quantize API
        q_model = quantize(
            model=fp32_model,
            quant_config=quant_config,
            run_fn=run_fn,
            run_args=(self.dataloader,),
        )
        _ = q_model(self.inp)  # inference
        tagert_modules = ["WQLinear_GEMM"]
        assert q_model.model.layers[0].self_attn.k_proj.__class__.__name__ in tagert_modules, "packing model failed."

