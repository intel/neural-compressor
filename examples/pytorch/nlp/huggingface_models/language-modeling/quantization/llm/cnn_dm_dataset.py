import sys
import argparse
import os
import time
import json
import fnmatch

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence


import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from torch.nn.functional import pad
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import random
random.seed(9973)

# Bucketize sequence lengths
MaxLens = range(0,64,1919)
Buckets = dict()
cutoff_step = 64
min_cutoff = 64
min_len = 1
for cutoff in range(min_cutoff, 1921, cutoff_step): # All input sequences
    Buckets[cutoff] = list(range(min_len, cutoff, 1))
    min_len = cutoff

#Buckets[1920] = list(range(min_len, 1921, 1))

input_buckets = dict()
for cutoff, seq_lens in Buckets.items():
    for seq_len in seq_lens:
        input_buckets[seq_len] = cutoff

#print("Buckets: {}".format(input_buckets))

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


class CNNDAILYMAIL(object):
    def __init__(self, model_path, data_path, device="cpu",is_calib=False, num_samples=20, max_len=1920):
        self.model_path = model_path
        self.data_path = data_path
        self.device = device
        self.num_samples = num_samples
        self.is_calib = is_calib

        self.padding = "max_length" if self.is_calib else False
        self.max_len = 2048 if self.is_calib else max_len

        self.calib_collator = self.collate_batch
        self.pad_max = max_len
        self.load_tokenizer()
        self.load_dataset()
    def load_dataset(self):
        """ Loads dataset"""
        with open(self.data_path, "r") as fid:
            list_data_dict = json.load(fid)
            self.list_data_dict = copy.deepcopy(list_data_dict)

        if self.num_samples is not None:
            self.num_samples = min(self.num_samples, len(list_data_dict))
            
            if self.is_calib:
                list_data_dict = list_data_dict[:self.num_samples]
            else:
                list_data_dict = random.choices(list_data_dict, k=self.num_samples)

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [prompt_input.format_map(example) for example in list_data_dict]
        targets = [f"{example['output']}" for example in list_data_dict]

        self.input_ids = []
        self.input_lens = []
        for i in range(len(sources)):
            tok_input = self.tokenize_function(sources[i])
            self.input_ids.append(tok_input.input_ids)
        

        #if self.num_samples is not None:
        #    self.num_samples = min(self.num_samples, len(list_data_dict))
        #    self.input_ids = random.choices(self.input_ids, k=self.num_samples)
        #    print("Sources: {}".format(len(sources)))
        #    print("Targets: {}".format(len(targets)))
        #    sources = random.choices(sources, k=self.num_samples)
        #    targets = random.choices(targets, k=self.num_samples)


        self.sources = sources
        self.targets = targets

    def load_tokenizer(self):
        """ Returns the tokenizer """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=2048,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def tokenize_function(self, text):
        example = self.tokenizer(text, truncation=True, max_length=self.max_len, return_tensors="pt", padding=self.padding)
        return example

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        input_ids = self.input_ids[i]
        input_len = input_ids.shape[-1]
        #pad_size = input_buckets[input_len] - input_len
        #input_ids = F.pad(input_ids, pad=(0, pad_size))
        return (input_ids, input_len)

    @torch.no_grad()
    def collate_batch(self, batch):
        input_ids_padded = []

        for input_ids, input_lens in batch: # input_ids are returned by this dataset (see __getitem__)
            pad_len = self.pad_max - input_ids.shape[0]
            #input_ids = F.pad(input_ids, pad=(0, pad_size), value=self.tokenizer.pad_token_id)
            input_ids_padded.append(input_ids)

        input_ids_padded = torch.vstack(input_ids_padded)
        return (input_ids_padded, input_ids_padded)

    def get_warmup_samples(self):
        cutoff_set = set(range(128, 1920, 64))
        warmup_samples = []
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [prompt_input.format_map(example) for example in self.list_data_dict]
        for source in sources: #self.input_ids:
            tok_input = self.tokenize_function(source)
            input_ids = tok_input.input_ids
            input_len = input_ids.shape[-1]
            bucket = input_buckets[input_len]
            if bucket in cutoff_set:
                #print("inputlen: {}; Bucket: {}".format(input_len, bucket))
                pad_size = bucket - input_len
                input_ids = F.pad(input_ids, pad=(0, pad_size), value=0)
                warmup_samples.append(input_ids)
                cutoff_set.remove(bucket)
                if len(cutoff_set)==0:
                    break

        return warmup_samples
