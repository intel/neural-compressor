import copy
import shutil

import pytest
import torch
import transformers
from packaging.version import Version, parse
import os
from functools import lru_cache
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

def is_xpu_available():
    return torch.xpu.is_available()

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
    
try:
    import compressed_tensors

    ct_installed = True
except ImportError:
    ct_installed = False


@torch.no_grad()
def run_fn(model, dataloader):
    for data in dataloader:
        if isinstance(data, tuple) or isinstance(data, list):
            model(*data)
        elif isinstance(data, dict):
            model(**data)
        else:
            model(data)

@pytest.mark.skipif(is_habana_framework_installed(), reason="These tests are not supported on HPU for now.")
@pytest.mark.skipif(not auto_round_installed, reason="auto_round module is not installed")
class TestAutoRoundCPU:
    @classmethod
    def setup_class(self):
        self.opt_model = transformers.AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torchscript=True,
        ).to("cpu")
        self.inp = torch.ones([1, 10], dtype=torch.long, device="cpu")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "facebook/opt-125m", trust_remote_code=True
        )
        from neural_compressor.torch.algorithms.autoround import get_dataloader
        self.dataloader = get_dataloader(self.tokenizer, 32, dataset_name="NeelNanda/pile-10k", seed=42, bs=8, nsamples=10)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("saved_results", ignore_errors=True)
        shutil.rmtree("temp_auto_round", ignore_errors=True)

    def setup_method(self, method):
        logger.info(f"Running TestAutoRound test: {method.__name__}")

    def test_quant_lm_head(self):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            # "trl-internal-testing/tiny-Phi3ForCausalLM",
            "optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM"
            )
        tokenizer =  AutoTokenizer.from_pretrained("optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM", trust_remote_code=True)
        
        quant_config = AutoRoundConfig(tokenizer=tokenizer, nsamples=32, seqlen=10, iters=1, amp=False ,scale_dtype="fp32", 
                                           quant_lm_head=True, group_size=32)
        logger.info(f"Test AutoRound with config {quant_config}")
        text = "Replace me by any text you'd like."
        encoded_input = tokenizer(text, return_tensors="pt")
        model = prepare(model=model, quant_config=quant_config)
        q_model = convert(model)
        output = tokenizer.decode(q_model.generate(**encoded_input, max_new_tokens=10)[0])
        print(output)
        assert output is not None
        tagert_modules = ["QuantLinear"]
        assert  q_model.lm_head.__class__.__name__ in tagert_modules, "packing model failed."

    def test_int4_dtype(self):
        fp32_model = copy.deepcopy(self.opt_model)
        quant_config = AutoRoundConfig(dtype="int4", nsamples=32, seqlen=10, iters=1, amp=False ,scale_dtype="fp32")
        logger.info(f"Test AutoRound with config {quant_config}")

        # prepare + convert API
        model = prepare(model=fp32_model, quant_config=quant_config)

        run_fn(model, self.dataloader)
        q_model = convert(model)
        _ = q_model(self.inp) # inference
        tagert_modules = ["QuantLinear"]
        assert  q_model.model.decoder.layers[0].self_attn.k_proj.__class__.__name__ in tagert_modules, "packing model failed."


    def test_autoround_with_quantize_API(self):
        fp32_model = copy.deepcopy(self.opt_model)

        quant_config = AutoRoundConfig(scheme="W4A16", seqlen=10, iters=1, use_sym=False, amp=False ,scale_dtype="fp32")
        logger.info(f"Test AutoRound with config {quant_config}")

        # quantize API
        q_model = quantize(
            model=fp32_model,
            quant_config=quant_config,
            run_fn=run_fn,
            run_args=(self.dataloader,),
        )
        _ = q_model(self.inp) # inference
        tagert_modules = ["QuantLinear"]
        assert  q_model.model.decoder.layers[0].self_attn.k_proj.__class__.__name__ in tagert_modules, "packing model failed."

    def test_conv1d(self):
        model = AutoModelForCausalLM.from_pretrained("MBZUAI/LaMini-GPT-124M", device_map="cpu", trust_remote_code=True)
        tokenizer =  AutoTokenizer.from_pretrained("MBZUAI/LaMini-GPT-124M", trust_remote_code=True)
        text = "Replace me by any text you'd like."
        encoded_input = tokenizer(text, return_tensors="pt")
        quant_config = AutoRoundConfig(nsamples=32, seqlen=10, iters=0, tokenizer=tokenizer, export_format="auto_round", device_map="cpu")
        model = prepare(model=model, quant_config=quant_config)
        q_model = convert(model)
        output = tokenizer.decode(q_model.generate(**encoded_input, max_new_tokens=10)[0])
        print(output)
        assert output is not None
        assert not isinstance(q_model.transformer.h[0].attn.c_attn, transformers.pytorch_utils.Conv1D), "loading compressed model failed."

    def test_utils(self):
        from neural_compressor.torch.utils.utility import (
            detect_device,
            get_layer_names_in_block,
            get_multimodal_block_names,
        )

        fp32_model = copy.deepcopy(self.opt_model)
        to_quant_block_names = get_multimodal_block_names(fp32_model, quant_vision=True)
        quant_config = AutoRoundConfig(
            nsamples=32, seqlen=10, iters=10, amp=False ,scale_dtype="fp16", to_quant_block_names=to_quant_block_names, device_map="cpu",
        )
        logger.info(f"Test AutoRound with config {quant_config}")
        device = "cpu"
        layers_list = get_layer_names_in_block(fp32_model, to_quant_block_names=to_quant_block_names)
        layers_list = get_layer_names_in_block(fp32_model)
        fp32_model.to(device)
        # quantizer execute
        model = prepare(model=fp32_model, quant_config=quant_config)
        run_fn(model, self.dataloader)
        q_model = convert(model)
        _ = q_model(self.inp) # inference
        tagert_modules = ["QuantLinear"]
        assert  q_model.model.decoder.layers[0].self_attn.k_proj.__class__.__name__ in tagert_modules, "packing model failed."


    @pytest.mark.skipif(Version(auto_round.__version__) <= Version("0.5.1"), reason="visual layer_name not processed.")
    def test_mllm(self):
        input = torch.randn(1, 32)
        from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

        from neural_compressor.torch.algorithms.autoround import get_mllm_dataloader

        model_name = "Qwen/Qwen2-VL-2B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True, device_map="cpu")
        dataloader, template, truncation, batch_size, gradient_accumulate_steps, seqlen, nsamples = get_mllm_dataloader(
            template=None,
            model=model,
            tokenizer=tokenizer,
            image_processor=None,
            dataset="NeelNanda/pile-10k",
            extra_data_dir=None,
            seqlen=32,
            batch_size=1,
            split=None,
            apply_template=None,
            truncation=False,
            seed=42,
            nsamples=1,
            gradient_accumulate_steps=1,
            quant_nontext_module=True,
            processor=processor,
        )
        quant_config = AutoRoundConfig(
            bits=4,
            group_size=128,
            nsamples=1,
            batch_size=batch_size,
            iters=1,
            seqlen=seqlen,
            quant_nontext_module=True,
            truncation=truncation,
            gradient_accumulate_steps=gradient_accumulate_steps,
            device_map="cpu",
        )

        model = prepare(model=model, quant_config=quant_config)
        run_fn(model, dataloader)
        q_model = convert(model)
        tagert_modules = ["QuantLinear"]
        assert q_model.model.language_model.layers[0].mlp.up_proj.__class__.__name__ in tagert_modules, "model quantization failed."

    def test_set_local(self):
        fp32_model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torchscript=True,
            device_map="cpu",
        )
        inp = torch.ones([1, 10], dtype=torch.long, device='cpu')
        output_dir = "./saved_inc"
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/opt-125m", trust_remote_code=True)
        quant_config = AutoRoundConfig(
            tokenizer=tokenizer, output_dir=output_dir,
            dtype="int4", nsamples=32, seqlen=10, iters=0, amp=False ,scale_dtype="fp32", export_format="auto_round", device_map="cpu")
        logger.info(f"Test AutoRound with config {quant_config}")
        quant_config.set_local("self.attn", AutoRoundConfig(dtype="fp16"))
        # {"self_attn": {"bits": 4, "data_type": "nv_fp", "act_bits": 16, "group_size": 16}}

        # prepare + convert API
        model = prepare(model=fp32_model, quant_config=quant_config)
        q_model = convert(model)
        model = AutoModelForCausalLM.from_pretrained(
            output_dir,
            torch_dtype="auto",
            device_map="cpu",
        )
        out = model(self.inp)[0]
        assert isinstance(q_model.model.decoder.layers[0].self_attn.v_proj, torch.nn.Linear), "set_local failed."
        
        # AutoRound API
        fp32_model = transformers.AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torchscript=True,
            device_map="cpu",
        )
        inp = torch.ones([1, 10], dtype=torch.long, device='cpu')
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "facebook/opt-125m", trust_remote_code=True)
        from auto_round import AutoRound
        layer_config = {"self.attn":{"data_type":"fp16"}}
        ar = AutoRound(
            tokenizer=tokenizer, model=fp32_model, layer_config=layer_config,
            data_type="int4", nsamples=32, seqlen=10, iters=0, amp=False ,scale_dtype="fp32", export_format="auto_round", device_map="cpu")
        quantized_model_path = "./saved_ar"
        ar.quantize_and_save(output_dir=quantized_model_path, inplace=True, format="auto_round")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path,
            torch_dtype="auto",
            device_map="cpu",
        )
        out_ar = model(inp)[0]
        assert torch.all(out_ar.eq(out))
        shutil.rmtree("./saved_inc", ignore_errors=True)
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    @pytest.mark.skipif(not ct_installed, reason="The compressed-tensors module is not installed.")
    @pytest.mark.parametrize("scheme", ["W4A16","W2A16","W3A16","W8A16","MXFP4","MXFP8", "NVFP4","FPW8A16","FP8_STATIC"])
    def test_scheme(self, scheme):
        # INC API
        fp32_model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torchscript=True,
            device_map="cpu",
        )
        inp = torch.ones([1, 10], dtype=torch.long, device='cpu')
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/opt-125m", trust_remote_code=True)

        output_dir = "./saved_inc"
        quant_config = AutoRoundConfig(
            tokenizer=tokenizer,
            nsamples=32,
            seqlen=10,
            iters=1,
            amp=False,
            scale_dtype="fp16",
            scheme=scheme,
            export_format="auto_round",
            output_dir=output_dir, # default is "temp_auto_round"
            device_map="cpu",
        )

        # quantizer execute
        model = prepare(model=fp32_model, quant_config=quant_config)
        inc_model = convert(model)
        if scheme in ["FPW8A16"]: # FP8_STATIC loading not supported yet
            return
        inc_model = AutoModelForCausalLM.from_pretrained(
            output_dir,
            torch_dtype="auto",
            device_map="cpu",
        )
        out = inc_model(inp)[0]
        
        # AutoRound API
        fp32_model = transformers.AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torchscript=True,
            device_map="cpu",
        )
        inp = torch.ones([1, 10], dtype=torch.long, device='cpu')
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "facebook/opt-125m", trust_remote_code=True)
        from auto_round import AutoRound
        ar = AutoRound(
            model=fp32_model,
            tokenizer=tokenizer,
            nsamples=32,
            seqlen=10,
            iters=1,
            amp=False,
            scale_dtype="fp16",
            scheme=scheme,
            device_map="cpu",
        )
        quantized_model_path = "./saved_ar"
        ar.quantize_and_save(output_dir=quantized_model_path, inplace=True, format="auto_round")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path,
            torch_dtype="auto",
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        out_ar = model(inp)[0]
        assert torch.all(out_ar.eq(out))
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.rmtree(quantized_model_path, ignore_errors=True)


    @pytest.mark.skipif(not ct_installed, reason="The compressed-tensors module is not installed.")
    @pytest.mark.skipif(Version(auto_round.__version__) < Version("0.9.0"), reason="target bits is not supported.")
    def test_target_bits(self):
        fp32_model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torchscript=True,
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/opt-125m", trust_remote_code=True)

        output_dir = "./saved_inc"
        quant_config = AutoRoundConfig(
            tokenizer=tokenizer,
            nsamples=32,
            seqlen=10,
            iters=1,
            target_bits=5,
            options=("MXFP4", "MXFP8"),
            enable_torch_compile=True,
            low_gpu_mem_usage=True,
            export_format="auto_round",
            device_map="cpu",
        )
        # quantizer execute
        model = prepare(model=fp32_model, quant_config=quant_config)
        model = convert(model)
        # mxfp4/8 model inference relys on autoround extension for vLLM.
        target_modules = ["MXFP4QuantLinear", "MXFP8QuantLinear"]
        assert (model.model.decoder.layers[0].self_attn.k_proj.__class__.__name__ in target_modules and \
                model.model.decoder.layers[1].fc1.__class__.__name__ in target_modules) ,\
            "model is not quantized correctly, please check."


    @pytest.mark.skipif(not ct_installed, reason="The compressed-tensors module is not installed.")
    @pytest.mark.skipif(Version(auto_round.__version__) < Version("0.9.0"), reason="target bits is not supported.")
    def test_target_bits_autotune(self):
        from neural_compressor.torch.quantization import TuningConfig, autotune
        baseline = 1
        eval_result = [0.9, 0.8, 0.99]
        acc_list = [baseline] + eval_result

        def eval_acc_fn(model) -> float:
            acc = acc_list.pop(0)
            return acc

        fp32_model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torchscript=True,
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/opt-125m", trust_remote_code=True)
        # AutoRound API
        custom_tune_config = TuningConfig(
            config_set=[
                AutoRoundConfig(
                    tokenizer=tokenizer,
                    target_bits=[5, 6, 7],
                    options=("MXFP4", "MXFP8"),
                    enable_torch_compile=True,
                    low_gpu_mem_usage=True,
                    export_format="auto_round",
                    iters=0,
                    device_map="cpu",
                )
            ]
        )
        best_model = autotune(model=fp32_model, tune_config=custom_tune_config, eval_fn=eval_acc_fn)
        # mxfp4/8 model inference relys on autoround extension for vLLM.
        target_modules = ["MXFP4QuantLinear", "MXFP8QuantLinear"]
        assert (best_model.model.decoder.layers[0].self_attn.k_proj.__class__.__name__ in target_modules and \
                best_model.model.decoder.layers[1].fc1.__class__.__name__ in target_modules) ,\
            "model is not quantized correctly, please check."

    def test_static_attention_dtype(self):
        fp32_model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torchscript=True,
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/opt-125m", trust_remote_code=True)

        output_dir = "./saved_inc"
        quant_config = AutoRoundConfig(
            tokenizer=tokenizer,
            iters=0,
            nsamples=2,
            seqlen=2,
            scheme="FP8_STATIC",
            static_attention_dtype="fp8",
            output_dir=output_dir,
            export_format="auto_round",
            device_map="cpu",
        )
        # quantizer execute
        model = prepare(model=fp32_model, quant_config=quant_config)
        model = convert(model)
        
        from safetensors import safe_open
        f = safe_open(os.path.join(output_dir, "model.safetensors"), framework="pt")
        assert "model.decoder.layers.8.self_attn.k_proj.input_scale" in f.keys()
        assert "model.decoder.layers.8.self_attn.k_proj.weight_scale" in f.keys()
        assert f.get_tensor("model.decoder.layers.5.self_attn.v_proj.input_scale").shape == torch.Size([1])
        assert f.get_tensor("model.decoder.layers.5.self_attn.v_proj.weight").dtype == torch.float8_e4m3fn
        check_attrs = ["k_scale", "v_scale", "q_scale"]

        for attr in check_attrs:
            weight_name = f"model.decoder.layers.8.self_attn.{attr}"
            assert weight_name in f.keys()
            assert f.get_tensor(weight_name).shape == torch.Size([1])
            assert f.get_tensor(weight_name).dtype == torch.float32
        shutil.rmtree(output_dir, ignore_errors=True)

    @pytest.mark.parametrize("static_kv_dtype", [None, "fp8", "float16"])
    def test_static_afp8_export(self, static_kv_dtype):
        fp32_model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torchscript=True,
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/opt-125m", trust_remote_code=True)

        output_dir = "./saved_inc"
        quant_config = AutoRoundConfig(
            tokenizer=tokenizer,
            bits=8,
            group_size=-1,
            iters=0,
            act_bits=8,
            nsamples=2,
            seqlen=2,
            data_type="fp8",
            act_data_type="fp8",
            act_dynamic=False,
            act_group_size=0,
            static_kv_dtype=static_kv_dtype,
            export_format="auto_round",
            output_dir=output_dir,
            device_map="cpu",
        )
        
        # quantizer execute
        model = prepare(model=fp32_model, quant_config=quant_config)
        model = convert(model)
        
        from safetensors import safe_open
        f = safe_open(os.path.join(output_dir, "model.safetensors"), framework="pt")
        assert "model.decoder.layers.8.self_attn.k_proj.input_scale" in f.keys()
        assert "model.decoder.layers.8.self_attn.k_proj.weight_scale" in f.keys()
        assert f.get_tensor("model.decoder.layers.5.self_attn.v_proj.input_scale").shape == torch.Size([1])
        assert f.get_tensor("model.decoder.layers.5.self_attn.v_proj.weight").dtype == torch.float8_e4m3fn
        if static_kv_dtype is None:
            with torch.no_grad():
                import transformers

                model = transformers.AutoModelForCausalLM.from_pretrained(
                    output_dir,
                    torch_dtype="auto",
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                model.eval()
                assert (
                    model.model.decoder.layers[0].self_attn.k_proj.__class__.__name__
                    == "WeightFP8ActFP8StaticQuantLinear"
                ), f"Expected WeightFP8ActFP8StaticQuantLinear, got {model.model.decoder.layers[0].self_attn.k_proj.__class__.__name__}"
                tokenizer = transformers.AutoTokenizer.from_pretrained(output_dir)
                prompt = "AI is "
                encode = tokenizer.encode(prompt, return_tensors="pt")
                with torch.no_grad():
                    output_tokens = model.generate(
                        encode,
                        max_length=10,
                    )
                    output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
                    print(f"Prompt: {prompt}")
                    print(f"Output: {output}")
                    assert output is not None, "Output should not be None"
        shutil.rmtree(output_dir, ignore_errors=True)
        
    @pytest.mark.parametrize(
        "scheme,  static_kv_dtype, static_attention_dtype",
        [
            ("MXFP4", None, "fp8"),
            ("MXFP4", "fp8", None),
            ("MXFP8", None, "fp8"),
            ("MXFP8", "fp8", None),
            ("NVFP4", None, "fp8"),
            ("NVFP4", "fp8", None),
        ]
    )
    def test_fp8_kv_attn(self, scheme, static_kv_dtype, static_attention_dtype):

        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from transformers.models.opt.modeling_opt import OPTForCausalLM

        model_name = "facebook/opt-125m"
        config = AutoConfig.from_pretrained(model_name)
        config.num_hidden_layers = 1
        model = OPTForCausalLM(config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        output_dir = "./saved_inc"
        quant_config = AutoRoundConfig(
            tokenizer=tokenizer,
            scheme=scheme,
            iters=0,
            seqlen=2,
            static_kv_dtype=static_kv_dtype,
            static_attention_dtype=static_attention_dtype,
            export_format="auto_round",
            output_dir=output_dir,
            reloading=False,
            device_map="cpu",
        )
        
        # quantizer execute
        model = prepare(model=model, quant_config=quant_config)
        compressed_model = convert(model)
        
        attn = compressed_model.model.decoder.layers[0].self_attn
        q_proj = attn.q_proj

        # weight_scale should exist for all quantized schemes
        assert hasattr(q_proj, "weight_scale"), f"Missing weight_scale in q_proj for scheme={scheme}"
        if static_kv_dtype == "fp8":
            assert (
                compressed_model.config.quantization_config["static_kv_dtype"] == "fp8"
            ), f"Invalid static_kv_dtype in config for scheme={scheme}, static_kv_dtype={static_kv_dtype}"

        # Only when static_kv_dtype / static_attention_dtype are fp8 do we expect FP8 KV scales
        if static_kv_dtype == "fp8" or static_attention_dtype == "fp8":
            assert attn.k_scale is not None and attn.v_scale is not None, (
                f"Missing k_scale/v_scale in attention for scheme={scheme}, "
                f"static_kv_dtype={static_kv_dtype}, static_attention_dtype={static_attention_dtype}"
            )

        if static_attention_dtype == "fp8":
            assert (
                compressed_model.config.quantization_config["static_attention_dtype"] == "fp8"
            ), f"Invalid static_attention_dtype in config for scheme={scheme}, static_attention_dtype={static_attention_dtype}"
            assert (
                getattr(attn, "q_scale", None) is not None
            ), f"Missing q_scale in attention for scheme={scheme}, static_attention_dtype={static_attention_dtype}"
        shutil.rmtree(output_dir, ignore_errors=True)

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


    @pytest.mark.parametrize("quant_lm_head", [True, False])
    def test_autoround(self, quant_lm_head):
        fp32_model = copy.deepcopy(self.tiny_llama_model)
        quant_config = AutoRoundConfig(nsamples=32, seqlen=10, iters=10, act_dtype="fp32", amp=False ,scale_dtype="fp32", quant_lm_head=quant_lm_head)
        logger.info(f"Test AutoRound with config {quant_config}")

        # prepare + convert API
        model = prepare(model=fp32_model, quant_config=quant_config)

        run_fn(model, self.dataloader)
        q_model = convert(model)
        assert "model.layers.0.self_attn.k_proj" in q_model.autoround_config.keys()
        assert "scale_dtype" in q_model.autoround_config["model.layers.0.self_attn.k_proj"].keys()
        assert torch.float32 == q_model.autoround_config["model.layers.0.self_attn.k_proj"]["scale_dtype"]
        assert isinstance(q_model.model.layers[0].self_attn.k_proj, WeightOnlyLinear), "packing model failed."
        if quant_lm_head is True:
            assert isinstance(q_model.lm_head, WeightOnlyLinear), "quantization for lm_head failed."

    def test_int4_dtype(self):
        fp32_model = copy.deepcopy(self.tiny_llama_model)
        quant_config = AutoRoundConfig(
            dtype="int4", nsamples=32, seqlen=10, iters=10, act_dtype="fp32", amp=False ,scale_dtype="fp32"
        )
        logger.info(f"Test AutoRound with config {quant_config}")

        # prepare + convert API
        model = prepare(model=fp32_model, quant_config=quant_config)
        run_fn(model, self.dataloader)
        q_model = convert(model)
        assert "model.layers.0.self_attn.k_proj" in q_model.autoround_config.keys()
        assert "scale_dtype" in q_model.autoround_config["model.layers.0.self_attn.k_proj"].keys()
        assert torch.float32 == q_model.autoround_config["model.layers.0.self_attn.k_proj"]["scale_dtype"]
        assert isinstance(q_model.model.layers[0].self_attn.k_proj, WeightOnlyLinear), "packing model failed."

    def test_autoround_with_quantize_API(self):
        model = copy.deepcopy(self.tiny_llama_model)

        quant_config = AutoRoundConfig(nsamples=32, seqlen=10, iters=10, act_dtype="fp32", amp=False ,scale_dtype="fp32")

        logger.info(f"Test AutoRound with config {quant_config}")

        # quantize API
        q_model = quantize(
            model=model,
            quant_config=quant_config,
            run_fn=run_fn,
            run_args=(self.dataloader,),
        )
        assert isinstance(q_model.model.layers[0].self_attn.k_proj, WeightOnlyLinear), "packing model failed."

@pytest.mark.skipif(not is_xpu_available(), reason="These tests are not supported on XPU for now.")
@pytest.mark.skipif(not auto_round_installed, reason="auto_round module is not installed")
class TestAutoRoundGPU:
    @pytest.mark.parametrize("scheme", ["W4A16","W2A16","W3A16","W8A16","MXFP4","MXFP8", "NVFP4","FPW8A16","FP8_STATIC"])
    def test_scheme(self, scheme):
        # INC API
        from transformers import AutoModelForCausalLM, AutoTokenizer
        fp32_model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
        )
        inp = torch.ones([1, 10], dtype=torch.long)
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/opt-125m", trust_remote_code=True)

        output_dir = "./saved_inc"
        quant_config = AutoRoundConfig(
            tokenizer=tokenizer,
            nsamples=32,
            seqlen=10,
            iters=1,
            device_map="xpu",
            scheme=scheme,
            export_format="auto_round",
            output_dir=output_dir, # default is "temp_auto_round"
        )

        # quantizer execute
        model = prepare(model=fp32_model, quant_config=quant_config)
        inc_model = convert(model)
        if scheme in ["FPW8A16"]: # FP8_STATIC loading not supported yet
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
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "facebook/opt-125m", trust_remote_code=True)
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

    @pytest.mark.parametrize("format", ["auto_awq","auto_gptq", "llm_compressor"])
    def test_format(self, format):
        # INC API
        scheme = "W4A16" if format != "llm_compressor" else "MXFP4"
        from transformers import AutoModelForCausalLM, AutoTokenizer
        fp32_model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
        )
        inp = torch.ones([1, 10], dtype=torch.long)
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/opt-125m", trust_remote_code=True)

        output_dir = "./saved_inc"
        quant_config = AutoRoundConfig(
            tokenizer=tokenizer,
            nsamples=32,
            seqlen=10,
            iters=1,
            device_map="xpu",
            scheme=scheme,
            export_format=format,
            output_dir=output_dir, # default is "temp_auto_round"
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
        from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
        fp32_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
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
            output_dir=output_dir, # default is "temp_auto_round"
        )

        # quantizer execute
        model = prepare(model=fp32_model, quant_config=quant_config)
        inc_model = convert(model)
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
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)

        output_dir = "./saved_inc"
        quant_config = AutoRoundConfig(
            tokenizer=tokenizer,
            nsamples=1,
            seqlen=10,
            iters=0, #rtn
            device_map="xpu",
            scheme=scheme,
            export_format="auto_round",
            output_dir=output_dir, # default is "temp_auto_round"
            quant_lm_head=True,
        )

        # quantizer execute
        model = prepare(model=fp32_model, quant_config=quant_config)
        inc_model = convert(model)
        assert inc_model is not None
        shutil.rmtree(output_dir, ignore_errors=True)
