import copy
import shutil

import pytest
import os
os.environ["PT_HPU_LAZY_MODE"] = "2"  # FIXME should be removed when GPTQ kernels are fixed
import torch
import transformers

from transformers import AutoTokenizer
from datasets import load_dataset
from neural_compressor.torch.quantization import FP8Config, convert, finalize_calibration, prepare, HybridGPTQConfig
from neural_compressor.torch.utils import is_hpex_available, get_accelerator
from neural_compressor.torch.quantization.save_load_entry import load


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
        if len(sample_text) < num_samples_to_iterate: # skip empty or very short examples
            continue
        j += 1
        inputs = tokenizer(sample_text, return_tensors="pt", max_length=128, truncation=True, padding='max_length').to("hpu")
        logs = model(**inputs).logits
        cur_accelerator.synchronize()
        logits.append(logs.detach().cpu())

@pytest.mark.skipif(not is_hpex_available(), reason="HPU environment is required!")
class TestGPTQwithFP8Quant:
    def setup_class(self):
        change_to_cur_file_dir()
        model_name="ybelkada/opt-125m-gptq-4bit"
        self.gptq_model = load(model_name, format="huggingface", device="hpu").to(torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # use a single layer to be able to compare
        self.gptq_model.model.decoder.layers = torch.nn.ModuleList([self.gptq_model.model.decoder.layers[0]])


    def teardown_class(self):
        shutil.rmtree("test_ouputs", ignore_errors=True)
        shutil.rmtree("saved_results", ignore_errors=True)


    """ 
    This test checks both vanilla gptq (w4a16) vs naive mixed precision (Hybrid) GPTQ - w4a8.
    Calibrates the w4a16 model on N num of samples, and compares the same samples with the hybrid model.
    """
    @torch.no_grad()
    @pytest.mark.skip(reason="SW-223106 load model error")
    def test_mixed_precision_gptq_fp8_quant_only_nlp(self):
        from neural_compressor.common import set_random_seed
        set_random_seed(12345)
        
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

        # step 1: measurement of a gptq checkpoint in bf16 (w4a16)
        model = copy.deepcopy(self.gptq_model)  # load a gptq checkpoint
        config = FP8Config.from_json_file("./test_fp8_jsons/test_measure.json")
        model = prepare(model, config)

        ref_logits = calib_func(model, dataset, self.tokenizer)
        finalize_calibration(model)
        
        # step 2: quantize matmuls to fp8, based on measurement
        model = copy.deepcopy(self.gptq_model)  # override measured model
        config = HybridGPTQConfig.from_json_file("../jsons/test_hw_quant.json")
        model = convert(model, config)

        quant_logits = calib_func(model, dataset, self.tokenizer)

        var_diff = torch.mean((ref_logits - quant_logits) ** 2)
        var_ref = torch.mean(ref_logits ** 2)
        rel_err = torch.sqrt(var_diff / var_ref)
        print(rel_err)
        assert rel_err < 0.08, "should have low rel_err."  # currently a high threshold has been chosed, due to a small amount of calibration examples
