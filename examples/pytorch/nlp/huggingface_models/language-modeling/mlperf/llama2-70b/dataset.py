import os
import time
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from typing import Optional, Dict, Sequence
import io
#import utils
import copy

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-70B-Dataset")

import random

class Dataset():
    def __init__(self, model_name=None, total_sample_count=24576, perf_count_override=None, dataset_path=None, device="cpu"):
        self.model_name = model_name or "meta-llama/Llama-2-70b-chat-hf"
        self.dataset_path = dataset_path
        self.max_length = 1024
        self.device = device

        #self.total_sample_count = total_sample_count

        self.load_tokenizer()
        self.load_processed_dataset()

        self.total_sample_count = min(len(self.input_ids), total_sample_count)
        self.perf_count = perf_count_override or self.total_sample_count

    def load_tokenizer(self):
        """ Returns tokenizer """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name)

        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_processed_dataset(self):
        if not os.path.isfile(self.dataset_path):
            log.warn("Processed pickle file {} not found. Please check that the path is correct".format(self.dataset_path))

        if "Llama-3" in self.model_name and "orca" not in self.dataset_path:
            import pandas as pd

            self.processed_data = pd.read_json(self.dataset_path)

            self.input = self.processed_data.input.tolist()
            self.input_ids = self.processed_data.tok_input.tolist()
            self.input_lens = [len(x) for x in self.input_ids]
            self.targets = self.processed_data.output.tolist()

            del self.processed_data
            return

        import pandas as pd
        # Note: Using pickle with trusted dataset files only
        # In production, consider using safer serialization formats like JSON or HDF5
        processed_data = pd.read_pickle(self.dataset_path)  # nosec B301
        # input_tokens = processed_data['tok_input']
        
        if ("405b" in self.dataset_path) and ("llama" not in self.model_name):
            # Running 405b dataset with a different model
            # Tokenize the dataset instead of using the tokenized dataset
            input_strs = processed_data['input']
            encode_bs = 128
            self.input_ids = []
            self.input_lens = []
            self.attention_masks = []
            input_strs_batch = []
            for i,strs in enumerate(input_strs):
                # if i%10==9:
                #     print(i)
                input_strs_batch.append(strs)
                if len(input_strs_batch)>=encode_bs:
                    input_ids_batch = self.tokenizer.batch_encode_plus(input_strs_batch)['input_ids']
                    for input_ids in input_ids_batch:
                        input_ids = torch.tensor([input_ids])
                        attn_mask = torch.ones_like(input_ids)
                        self.input_ids.append(input_ids)
                        self.attention_masks.append(attn_mask)
                        self.input_lens.append(input_ids.shape[-1])
                    input_strs_batch = []
                    # exit()
            if len(input_strs_batch)>0:
                input_ids_batch = self.tokenizer.batch_encode_plus(input_strs_batch)['input_ids']
                for input_ids in input_ids_batch:
                    input_ids = torch.tensor([input_ids])
                    attn_mask = torch.ones_like(input_ids)
                    self.input_ids.append(input_ids)
                    self.attention_masks.append(attn_mask)
                    self.input_lens.append(input_ids.shape[-1])
                input_strs_batch = []
        elif ("orca" in self.dataset_path) and ("llama2" not in self.model_name.lower()):
            user_prompts = processed_data['question']
            system_prompts = processed_data['system_prompt']

            self.input_ids = []
            self.input_lens = []
            self.attention_masks = []

            for i in range(len(processed_data)):
                user_prompt = user_prompts.iloc[i]
                system_prompt = system_prompts.iloc[i]
                message = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                input_ids = self.tokenizer.apply_chat_template(message, add_generation_prompt=True)
                input_ids = torch.tensor(input_ids, dtype=torch.int32).view(1,-1).to(self.device)
                attn_mask = torch.ones_like(input_ids)
                self.input_ids.append(input_ids)
                self.attention_masks.append(attn_mask)
                self.input_lens.append(input_ids.shape[-1])
        else:
            input_tokens = processed_data['tok_input']
            self.input_ids = []
            self.input_lens = []
            self.attention_masks = []

            for ids in input_tokens:
                input_ids = torch.tensor(ids, dtype=torch.int32).view(1,-1).to(self.device)
                attn_mask = torch.ones_like(input_ids)
                self.input_ids.append(input_ids)
                self.attention_masks.append(attn_mask)
                self.input_lens.append(input_ids.shape[-1])

    def postProcess(self, out_tokens, input_seq_lens=None, query_id_list=None, sample_index_list=None):
        """ Postprocesses output prediction """

        output_seqs = []
        for i,out_token in enumerate(out_tokens):
            output_seq = np.array(out_token).reshape(-1)
            output_seqs.append(output_seq)

        return output_seqs

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesFromRam(self, sample_list):
        pass

    def __del__(self):
        pass
