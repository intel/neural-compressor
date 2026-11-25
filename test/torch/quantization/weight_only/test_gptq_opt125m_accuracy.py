import os
import random
import shutil

import pytest
import torch
import transformers
from datasets import load_dataset
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

from neural_compressor.torch.quantization import GPTQConfig, convert, load, prepare
from neural_compressor.torch.utils import accelerator

device = accelerator.name()

MODEL_NAME = "facebook/opt-125m"
CALIB_DATASET = "NeelNanda/pile-10k"
CALIB_NSAMPLES = 128
CALIB_SEQ_LEN = 2048
EVAL_TASK = "lambada_openai"
EVAL_LIMIT = 800
EVAL_BATCH_SIZE = 8
REFERENCE_ACC = 0.3325  # lambada_openai acc with limit=800 (full set: 0.3528)
RELATIVE_TOLERANCE = 0.03


def get_calib_data(tokenizer):
    # Follow the official GPTQ calibration: take sequences longer than CALIB_SEQ_LEN and crop a random window.
    dataset = load_dataset(CALIB_DATASET, split="train").shuffle(seed=42)
    random.seed(0)
    calib_data = []
    for text in dataset["text"]:
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        length = input_ids.shape[-1]
        if length < CALIB_SEQ_LEN:
            continue
        if length > CALIB_SEQ_LEN:
            start = random.randint(0, length - CALIB_SEQ_LEN - 1)
            input_ids = input_ids[:, start : start + CALIB_SEQ_LEN]
        calib_data.append(input_ids)
        if len(calib_data) == CALIB_NSAMPLES:
            break
    return calib_data


@pytest.mark.skipif(device != "cpu", reason="accuracy reference is measured on CPU")
class TestGPTQOpt125mAccuracy:
    def setup_class(self):
        self.output_dir = "opt_125m_woq_gptq_int4"

    def teardown_class(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_opt_125m_woq_gptq_int4_accuracy(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
        model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
        calib_data = get_calib_data(tokenizer)

        quant_config = GPTQConfig(
            bits=4,
            use_sym=False,
            group_size=128,
            use_mse_search=True,
            percdamp=0.01,
            act_order=False,
            block_size=128,
        )
        model = prepare(model, quant_config)
        with torch.no_grad():
            for input_ids in calib_data:
                model(input_ids)
        model = convert(model)
        model.save(self.output_dir)

        fp32_model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
        q_model = load(os.path.abspath(self.output_dir), fp32_model, device="cpu")
        q_model.eval()

        lm = HFLM(pretrained=q_model, tokenizer=tokenizer, batch_size=EVAL_BATCH_SIZE, device="cpu")
        results = simple_evaluate(model=lm, tasks=[EVAL_TASK], limit=EVAL_LIMIT)
        acc = results["results"][EVAL_TASK]["acc,none"]
        print(f"{EVAL_TASK} accuracy: {acc:.4f}, reference: {REFERENCE_ACC:.4f}")
        # One-sided: GPTQ rounding is sensitive to CPU/BLAS numerics, so higher accuracy is not a regression.
        assert acc >= (1 - RELATIVE_TOLERANCE) * REFERENCE_ACC, (
            f"Accuracy {acc:.4f} is more than {RELATIVE_TOLERANCE:.0%} below reference {REFERENCE_ACC:.4f}"
        )
