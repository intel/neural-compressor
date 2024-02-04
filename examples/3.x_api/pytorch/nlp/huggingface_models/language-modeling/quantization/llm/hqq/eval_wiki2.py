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

import gc
import time

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from neural_compressor.torch.algorithms.weight_only.hqq.auto_accelerator import auto_detect_accelerator
from neural_compressor.torch.algorithms.weight_only.hqq.utility import dump_elapsed_time


def cleanup():
    # torch.cuda.empty_cache()
    auto_detect_accelerator().empty_cache()
    gc.collect()


# Adapted from https://huggingface.co/transformers/v4.2.2/perplexity.html
@dump_elapsed_time("Evaluate wikitext2 time .. ")
def eval_wikitext2(model, tokenizer, max_length=1024, stride=512, verbose=True):
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_eos_token = False

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")

    encodings["input_ids"] = encodings["input_ids"].to(auto_detect_accelerator().current_device())

    lls, t = [], []
    for i in tqdm(range(0, encodings["input_ids"].size(1), stride), disable=not verbose):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings["input_ids"].size(1))
        trg_len = end_loc - i
        input_ids = encodings["input_ids"][:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # ignore context

        t1 = time.time()
        with torch.no_grad():
            log_likelihood = model(input_ids, labels=target_ids).loss * trg_len
        # torch.cuda.synchronize()
        auto_detect_accelerator().synchronize()
        t2 = time.time()
        t.append((t2 - t1))
        lls.append(log_likelihood)

        del input_ids, target_ids

    ppl = np.round(float(torch.exp(torch.stack(lls).sum() / end_loc)), 4)
    pred_time = np.round(np.mean(t), 3)
    if verbose:
        print("perplexity", ppl)
        print("time", str(pred_time) + "  sec")

    del encodings
    cleanup()

    return {"perplexity": ppl, "prediction_time": pred_time}
