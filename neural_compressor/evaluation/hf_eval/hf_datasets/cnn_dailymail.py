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
import argparse
import copy
import fnmatch
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import nltk
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

nltk.download("punkt", quiet=False)
random.seed(9973)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n Input:\n {article}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

current_path = os.path.abspath(__file__)
json_path = os.path.join(
    os.path.abspath(os.path.dirname(current_path) + os.path.sep + "."),
    "cnn_validation.json",
)


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return (input_ids, labels)


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


class CNNDAILYMAIL(object):
    def __init__(
        self,
        model_path,
        data_path=json_path,
        device="cpu",
        calib=False,
        num_samples=None,
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.device = device
        self.num_samples = num_samples
        self.load_tokenizer()
        self.load_dataset()
        self.data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)
        self.calib = calib

    def wrap(self, example):
        example["instruction"] = "Summarize the following news article:"
        return example

    def load_dataset(self):
        """Loads dataset."""
        list_data_dict = load_dataset("cnn_dailymail", "3.0.0")["validation"]
        if self.num_samples is not None:
            self.num_samples = min(self.num_samples, len(list_data_dict))
            list_data_dict = random.choices(list_data_dict, k=self.num_samples)
        prompt_input, prompt_no_input = (
            PROMPT_DICT["prompt_input"],
            PROMPT_DICT["prompt_no_input"],
        )
        sources = [prompt_input.format_map(self.wrap(example)) for example in list_data_dict]
        targets = [f"{example['highlights']}" for example in list_data_dict]
        data_dict = preprocess(sources, targets, self.tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.sources = sources
        self.targets = targets

    def load_tokenizer(self):
        """Returns the tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=2048,
            padding_side="left",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


def postprocess_text(preds, targets):
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]

    # rougeLSum expects newline after each sentence
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

    return preds, targets
