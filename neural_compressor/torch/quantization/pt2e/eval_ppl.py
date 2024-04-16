# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from x86_quantizer_hf import quant
import gc
import time

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm


def cleanup():
    torch.cuda.empty_cache()
    # auto_detect_accelerator().empty_cache()
    gc.collect()


def post_process_raw_out(logits, labels, config):
    if not isinstance(logits, torch.Tensor):
        logits = logits.logits
    from torch.nn import CrossEntropyLoss
    logits = logits.float()

    loss = None

    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss

# Adapted from https://huggingface.co/transformers/v4.2.2/perplexity.html
# @dump_elapsed_time("Evaluate wikitext2 time .. ")


def eval_wikitext2(model, tokenizer, max_length=1024, stride=512, verbose=True, msg="", raw_out=False, config = None):
    print("eval model for ", msg)
    # model.to("cuda")
    # model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_eos_token = False

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")

    encodings["input_ids"] = encodings["input_ids"]

    lls, t = [], []
    cur_iter = 0
    for i in range(0, encodings["input_ids"].size(1), stride): #, disable=not verbose:
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings["input_ids"].size(1))
        trg_len = end_loc - i
        print(f"begin_loc: {begin_loc}, end_loc: {end_loc}, trg_len: {trg_len}")
        input_ids = encodings["input_ids"][:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # ignore context

        t1 = time.time()
        with torch.no_grad():
            if raw_out:
                print(f"input_ids shape: {input_ids.shape}")
                out = model(input_ids)
                loss = post_process_raw_out(logits=out, labels=target_ids, config=config)
            else:
                out = model(input_ids, labels=target_ids)
                loss = out.loss
            log_likelihood = loss * trg_len
        # if "cuda" in str(model.device):
        #     torch.cuda.synchronize()
            # auto_detect_accelerator().synchronize()
        t2 = time.time()
        t.append((t2 - t1))
        lls.append(log_likelihood)

        cur_iter += 1
        if cur_iter % 10 == 0:
            tmp_ppl = np.round(float(torch.exp(torch.stack(lls).sum() / end_loc)), 4)
            print(f"Perplexity after {cur_iter} iterations: {tmp_ppl}")

        del input_ids, target_ids

    ppl = np.round(float(torch.exp(torch.stack(lls).sum() / end_loc)), 4)
    pred_time = np.round(np.mean(t), 3)
    if verbose:
        print("perplexity", ppl)
        print("time", str(pred_time) + "  sec")

    del encodings
    cleanup()

    return {"perplexity": ppl, "prediction_time": pred_time}


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "/models/opt-125m"
    model_name = "/mnt/disk4/modelHub/opt-125m/"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    config = model.config
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # eval_wikitext2(model, tokenizer, stride=9, raw_out=True, config = config)

    quantized_model = quant(model_name, fold_quantize=False, eval=True, use_dynamic_shape=True)
    
    eval_wikitext2(quantized_model, tokenizer, raw_out=True, config = config)

    #
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # eval_wikitext2(model, tokenizer, raw_out=True)
