#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import random
import time

import evaluate  # pylint: disable=E0401
import numpy as np
import torch
from tqdm import tqdm

from .hf_datasets.cnn_dailymail import CNNDAILYMAIL, postprocess_text

seed = 9973
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def summarization_evaluate(model, tokenizer_name=None, task="cnn_dailymail", batch_size=1, limit=None):
    generate_kwargs = {
        "max_new_tokens": 128,
        "min_new_tokens": 30,
        "num_beams": 4,
        "early_stopping": True,
    }
    metric = evaluate.load("rouge")
    val_dataset = CNNDAILYMAIL(tokenizer_name, device="cpu", calib=False, num_samples=limit)
    sources = val_dataset.sources
    targets = val_dataset.targets
    tokenizer = val_dataset.tokenizer
    preds = []
    predictions = []
    ground_truths = []
    max_len = 1919
    with torch.inference_mode(), torch.no_grad():
        for i in tqdm(range(0, len(sources), batch_size)):
            input_batch = tokenizer.batch_encode_plus(
                sources[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            )
            input_ids = input_batch.input_ids
            input_lens = input_ids.shape[-1]
            t0 = time.time()
            out_tokens = model.generate(**input_batch, **generate_kwargs, pad_token_id=tokenizer.pad_token_id)
            t1 = time.time()
            print("Inference time: {}".format(round(t1 - t0, 3)))
            print("Seq len: {}".format(input_lens))
            if "t5" in tokenizer_name:
                pred = out_tokens
                print("Out len: {}".format(out_tokens.shape[-1]))
            else:
                pred = out_tokens[:, input_lens:]
                print("Out len: {}".format(out_tokens.shape[-1] - input_lens))
            pred_batch = tokenizer.batch_decode(pred, skip_special_tokens=True)
            targ_batch = targets[i : i + batch_size]
            preds, targs = postprocess_text(pred_batch, targ_batch)
            predictions.extend(preds)
            ground_truths.extend(targs)

            if limit is not None and i == int(limit / batch_size):
                break

    result = metric.compute(
        predictions=predictions, references=ground_truths, use_stemmer=False
    )  # , tokenizer=tokenizer)
    result = {k: round(v * 100, 4) for k, v in result.items()}

    print(result)
    return result
