import copy
import os
import shutil

import pytest

os.environ["PT_HPU_LAZY_MODE"] = "1"  # FIXME should be removed when GPTQ kernels are fixed
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer

from neural_compressor.torch.quantization import (
    FP8Config,
    HybridGPTQConfig,
    convert,
    finalize_calibration,
    get_default_gptq_config,
    load,
    prepare,
)
from neural_compressor.torch.quantization.save_load_entry import load
from neural_compressor.torch.utils import accelerator, get_accelerator, is_hpex_available


def change_to_cur_file_dir():
    import os

    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    os.chdir(current_directory)


@torch.no_grad()
def calib_func(model, dataset, tokenizer=None):
    cur_accelerator = get_accelerator()
    logits = []
    num_samples = 20
    num_samples_to_iterate = 100

    j = 0
    for i in range(num_samples_to_iterate):
        if j > num_samples:
            return torch.cat(logits)
        sample_text = dataset["test"][i]["text"]
        if len(sample_text) < num_samples_to_iterate:  # skip empty or very short examples
            continue
        j += 1
        inputs = tokenizer(sample_text, return_tensors="pt", max_length=128, truncation=True, padding="max_length").to(
            "hpu"
        )
        logs = model(**inputs).logits
        cur_accelerator.synchronize()
        logits.append(logs.detach().cpu())


def run_fn(model):
    model(torch.tensor([[10, 20, 30]], dtype=torch.long).to(device))


device = accelerator.name()


class TestGPTQHybrid:
    def setup_class(self):
        self.tiny_opt = transformers.AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            device_map=device,
        )
        self.tiny_opt.model.decoder.layers = torch.nn.ModuleList(
            [self.tiny_opt.model.decoder.layers[0]]
        )  # Keep only the first decoder layer to speed up testing
        self.example_inputs = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long).to(device)
        self.label = self.tiny_opt(self.example_inputs)[0]
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.my_dir = os.getcwd()
        self.qmodel_weight_file_path = os.path.join(self.current_dir, "tested_saved_model")
        self.measure_json_path = os.path.join(self.current_dir, "test_fp8_jsons", "test_measure.json")
        self.quant_json_path = os.path.join(self.current_dir, "test_fp8_jsons", "test_pow2_w4a8_quant.json")
        self.output_path = os.path.join(self.my_dir, "test_outputs")

    def teardown_class(self):
        shutil.rmtree(self.output_path, ignore_errors=True)
        shutil.rmtree(self.qmodel_weight_file_path, ignore_errors=True)

    def get_fresh_model(self):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            device_map=device,
        )
        model.model.decoder.layers = torch.nn.ModuleList([model.model.decoder.layers[0]])
        return model

    @torch.no_grad()
    def test_fp8_aware_gptq(self):
        """This test checks W4A8 DPQ algorirthm."""
        from neural_compressor.common import set_random_seed

        set_random_seed(12345)

        # step1: load bf16 model and convert to int4 using fp8 aware quantization
        model = self.get_fresh_model()
        ref_logits = model(self.example_inputs)[0]  # Inference on bf16 model for reference results
        quant_config = get_default_gptq_config()
        quant_config.fp8_aware = True
        quant_config.global_config.fp8_aware = True  # enabling fp8 aware quantization in gptq
        model = prepare(model, quant_config=quant_config)
        run_fn(model)
        model_4_bits = convert(model)
        model_4_bits.save(self.qmodel_weight_file_path)  # we test also the save and load

        # step2: Load 4-bit model and get measurements for FP8 activation quantization
        model = load(
            model_name_or_path=self.qmodel_weight_file_path,
            format="default",
            device="hpu",
            original_model=self.tiny_opt,
        )
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")  # for fp8 measurements
        config = HybridGPTQConfig.from_json_file(self.measure_json_path)
        model = prepare(model, config)
        calib_func(model, dataset, self.tokenizer)
        finalize_calibration(model)
        del (
            model.quantizer,
            model.quant_config,
            model.qconfig,
        )  # Remove previous quantization settings to enable hybrid quantization
        config = HybridGPTQConfig.from_json_file(self.quant_json_path)
        model = convert(model, config)
        w4a8_logits = model(self.example_inputs)[0]
        var_diff = torch.mean((ref_logits - w4a8_logits) ** 2)
        var_ref = torch.mean(ref_logits**2)
        rel_err = torch.sqrt(var_diff / var_ref)
        # assert rel_err < 0.07, "should have low rel_err."
        # TODO fix the test - temporarily disabled

    def test_hybrid_ordering_improvement(self):
        """This test GPTQ with hybrid re-ordering (GAR algorithm) compared to vanilla GPTQ."""
        from neural_compressor.common import set_random_seed

        set_random_seed(12345)
        model = self.get_fresh_model()  # loading bf16 model
        ref_label = model(self.example_inputs)[0]  # bf16 reference output
        quant_config = get_default_gptq_config()
        model = prepare(model, quant_config=quant_config)
        run_fn(model)
        model = convert(model)  # quantizing to int4 using gptq
        gptq_label = model(self.example_inputs)[0]  # reference without hybrid activation reordering
        gptq_atol = (gptq_label - ref_label).amax()
        model = self.get_fresh_model()
        quant_config = get_default_gptq_config()
        quant_config.hybrid_order = True
        quant_config.global_config.hybrid_order = True
        model = prepare(model, quant_config=quant_config)
        run_fn(model)
        model = convert(model)  # quantizing to int4 using gptq and hybrid activation reordering
        gptq_hybrid_label = model(self.example_inputs)[0]
        hybrid_atol = (gptq_hybrid_label - ref_label).amax()
        print(gptq_atol - hybrid_atol)
        assert hybrid_atol < gptq_atol, "Hybrid ordering should yield lower ATOL than vanilla GPTQ."
