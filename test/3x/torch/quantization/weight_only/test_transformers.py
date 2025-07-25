import shutil
from math import isclose

import pytest
import torch
import transformers
from packaging.version import Version
from transformers import AutoTokenizer

from neural_compressor.torch.utils import get_ipex_version
from neural_compressor.common.utils.utility import CpuInfo
from neural_compressor.transformers import (
    AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration,
    AutoRoundConfig,
    AwqConfig,
    GPTQConfig,
    RtnConfig,
    TeqConfig,
)

torch.manual_seed(42)

ipex_version = get_ipex_version()

try:
    import auto_round

    auto_round_installed = True
except ImportError:
    auto_round_installed = False

class TestTansformersLikeAPI:
    def setup_class(self):
        self.model_name_or_path = "hf-tiny-model-private/tiny-random-GPTJForCausalLM"
        self.autoawq_model = "casperhansen/opt-125m-awq"
        self.prompt = "One day, the little girl"
        self.generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)

    def teardown_class(self):
        shutil.rmtree("nc_workspace", ignore_errors=True)
        shutil.rmtree("transformers_tmp", ignore_errors=True)
        shutil.rmtree("transformers_vlm_tmp", ignore_errors=True)

    def test_quantization_for_llm(self):
        model_name_or_path = self.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        fp32_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        dummy_input = fp32_model.dummy_inputs["input_ids"]
        label = fp32_model(dummy_input)[0]

        # weight-only
        # RTN
        woq_config = RtnConfig(bits=4, group_size=16)
        woq_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=woq_config,
        )
        woq_model.eval()

        output = woq_model(dummy_input)[0]
        assert torch.allclose(output, label, atol=0.12), "Accuracy gap atol > 0.1 is unexpected."
        # label[0][0][0] = -0.0910
        assert isclose(float(output[0][0][0]), -0.1006, abs_tol=1e-04)

        # AWQ
        woq_config = AwqConfig(
            bits=4, zero_point=False, n_samples=5, batch_size=1, seq_len=512, group_size=16, tokenizer=tokenizer
        )

        woq_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=woq_config)
        woq_model.eval()
        output = woq_model(dummy_input)
        assert isclose(float(output[0][0][0][0]), -0.1045, abs_tol=1e-04)

        # TEQ
        woq_config = TeqConfig(bits=4, n_samples=5, batch_size=1, seq_len=512, group_size=16, tokenizer=tokenizer)
        woq_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=woq_config)
        woq_model.eval()
        output = woq_model(dummy_input)
        assert isclose(float(output[0][0][0][0]), -0.1006, abs_tol=1e-04)

        # GPTQ
        woq_config = GPTQConfig(
            bits=4,
            weight_dtype="int4_clip",
            sym=True,
            desc_act=False,
            damp_percent=0.01,
            blocksize=32,
            n_samples=3,
            seq_len=256,
            tokenizer=tokenizer,
            batch_size=1,
            group_size=16,
        )
        woq_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=woq_config)
        woq_model.eval()
        output = woq_model(dummy_input)
        # The output of torch.cholesky() changes on different torch version
        if ipex_version < Version("2.5.0"):
            assert isclose(float(output[0][0][0][0]), -0.08614, abs_tol=1e-04)
        else:
            assert isclose(float(output[0][0][0][0]), -0.0874, abs_tol=1e-04)

        # AUTOROUND
        woq_config = AutoRoundConfig(
            bits=4, weight_dtype="int4_clip", n_samples=128, seq_len=32, iters=5, group_size=16, tokenizer=tokenizer
        )
        woq_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=woq_config)
        woq_model.eval()
        output = woq_model(dummy_input)
        # The output might change when device supports bf16
        if CpuInfo().bf16:
            assert isclose(float(output[0][0][0][0]),  -0.07275, abs_tol=1e-04)
        else:
            assert isclose(float(output[0][0][0][0]), -0.0786, abs_tol=1e-04)

    def test_save_load(self):
        model_name_or_path = self.model_name_or_path

        fp32_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        dummy_input = fp32_model.dummy_inputs["input_ids"]

        # RTN
        woq_config = RtnConfig(bits=4, group_size=16)
        woq_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=woq_config,
        )
        woq_output = woq_model(dummy_input)[0]

        # save
        output_dir = "./transformers_tmp"
        woq_model.save_pretrained(output_dir)

        # load
        loaded_model = AutoModelForCausalLM.from_pretrained(output_dir)
        loaded_output = loaded_model(dummy_input)[0]
        assert torch.equal(woq_output, loaded_output), "loaded output should be same. Please double check."

    def test_use_layer_wise(self):
        model_name_or_path = self.model_name_or_path

        fp32_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        dummy_input = fp32_model.dummy_inputs["input_ids"]

        # RTN
        # Case1: use_layer_wise=True
        woq_config = RtnConfig(bits=4, group_size=16, use_layer_wise=True)
        woq_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=woq_config,
        )
        woq_output = woq_model(dummy_input)[0]

        # save
        output_dir = "./transformers_tmp"
        woq_model.save_pretrained(output_dir)

        # load
        loaded_model = AutoModelForCausalLM.from_pretrained(output_dir)
        loaded_output = loaded_model(dummy_input)[0]
        assert torch.equal(woq_output, loaded_output), "loaded output should be same. Please double check."

        # Case2: use_layer_wise=False
        woq_config = RtnConfig(bits=4, group_size=16, use_layer_wise=False)
        woq_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=woq_config,
        )
        woq_output2 = woq_model(dummy_input)[0]
        assert torch.equal(woq_output, woq_output2), "use_layer_wise output should be same. Please double check."

        # Case3: test safetensors model file
        from neural_compressor.torch.algorithms.layer_wise.utils import get_path

        model_path = get_path(model_name_or_path)
        from transformers import AutoModelForCausalLM as RawAutoModelForCausalLM

        ori_model = RawAutoModelForCausalLM.from_pretrained(model_name_or_path)
        # test 1 safetensors file
        ori_model.save_pretrained(model_path, safe_serialization=True)
        woq_config = RtnConfig(bits=4, group_size=16, use_layer_wise=True)

        woq_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=woq_config,
        )
        woq_output_1_safetensors = woq_model(dummy_input)[0]
        assert torch.equal(woq_output, woq_output_1_safetensors)

        # test 3 safetensors files
        ori_model.save_pretrained(model_path, safe_serialization=True, max_shard_size="250KB")
        woq_config = RtnConfig(bits=4, group_size=16, use_layer_wise=True)
        woq_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=woq_config,
        )
        woq_output_3_safetensors = woq_model(dummy_input)[0]
        assert torch.equal(woq_output, woq_output_3_safetensors)

        # case4: test dowload_hf_model
        shutil.rmtree(model_path, ignore_errors=True)
        woq_config = RtnConfig(bits=4, group_size=16, use_layer_wise=True)

        woq_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=woq_config,
        )
        woq_output_download = woq_model(dummy_input)[0]
        assert torch.equal(woq_output_download, woq_output)

    @pytest.mark.skipif(Version(transformers.__version__) > Version("4.52.0"), reason="modeling_opt.py changed.")
    def test_loading_autoawq_model(self):
        user_model = AutoModelForCausalLM.from_pretrained(self.autoawq_model)
        tokenizer = AutoTokenizer.from_pretrained(self.autoawq_model)
        input_ids = tokenizer(self.prompt, return_tensors="pt")["input_ids"]
        self.generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)
        gen_ids = user_model.generate(input_ids, **self.generate_kwargs)
        gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        if Version(transformers.__version__) < Version("4.47.0"):
            target_text = ["One day, the little girl in the back of my mind will ask me if I'm a"]
        else:
            target_text = ["One day, the little girl in the back of my mind will say, “I’m so glad you’"]
        assert gen_text == target_text, "loading autoawq quantized model failed."
    
    @pytest.mark.skipif(not auto_round_installed, reason="auto_round module is not installed")
    def test_vlm(self):
        model_name = "Qwen/Qwen2-VL-2B-Instruct"
        from neural_compressor.transformers import Qwen2VLForConditionalGeneration
        from neural_compressor.transformers import AutoModelForCausalLM
        woq_config = AutoRoundConfig(
            bits=4, 
            group_size=128,
            is_vlm=True,
            dataset="NeelNanda/pile-10k",  
            iters=1,
            n_samples=1,
            seq_len=32,
            batch_size=1,
        )
        
        woq_model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True)
       
        if  hasattr(torch, "xpu") and torch.xpu.is_available():
            from intel_extension_for_pytorch.nn.utils._quantize_convert import WeightOnlyQuantizedLinear
        else:    
            from intel_extension_for_pytorch.nn.modules import WeightOnlyQuantizedLinear

        if Version(transformers.__version__) >= Version("4.52"):
            assert isinstance(woq_model.model.language_model.layers[0].self_attn.k_proj, WeightOnlyQuantizedLinear), "replacing model failed."
        else:
            assert isinstance(woq_model.model.layers[0].self_attn.k_proj, WeightOnlyQuantizedLinear), "replacing model failed."
        
        #save
        woq_model.save_pretrained("transformers_vlm_tmp")
        
        #load
        loaded_model = Qwen2VLForConditionalGeneration.from_pretrained("transformers_vlm_tmp")
        if Version(transformers.__version__) >= Version("4.52"):
            assert isinstance(loaded_model.model.language_model.layers[0].self_attn.k_proj, WeightOnlyQuantizedLinear), "loaing model failed."
        else:
            assert isinstance(loaded_model.model.layers[0].self_attn.k_proj, WeightOnlyQuantizedLinear), "loaing model failed."
        # phi-3-vision-128k-instruct, disable as CI consumes too much time
        # woq_config = AutoRoundConfig(
        #     bits=4, 
        #     group_size=128,
        #     is_vlm=True,
        #     dataset="liuhaotian/llava_conv_58k",
        #     iters=2,
        #     n_samples=5,
        #     seq_len=64,
        #     batch_size=1,
        # )
        # model_name = "microsoft/Phi-3-vision-128k-instruct"
        # woq_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True, attn_implementation='eager')
        # assert isinstance(woq_model.model.layers[0].self_attn.o_proj, WeightOnlyQuantizedLinear), "quantizaion failed."

    def test_save_load_for_inc_model(self):
        model_name_or_path = self.model_name_or_path

        fp32_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        dummy_input = fp32_model.dummy_inputs["input_ids"]

        # RTN
        woq_config = RtnConfig(bits=4, group_size=16)
        woq_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=woq_config,
        )
        woq_output = woq_model(dummy_input)[0]
        
        # RTN
        woq_config = RtnConfig(bits=4, group_size=16)
        woq_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=woq_config,
            for_inference=False,
        )

        # save
        output_dir = "./transformers_tmp"
        woq_model.save_pretrained(output_dir)

        # load
        loaded_model = AutoModelForCausalLM.from_pretrained(output_dir)
        loaded_output = loaded_model(dummy_input)[0]
        assert torch.equal(woq_output, loaded_output), "loaded output should be same. Please double check."
