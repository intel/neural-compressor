# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np


class SST2DataSet():
    
    def __init__(self, data_dir, tokenizer_dir):
        dataset = load_dataset('glue', 'sst2', cache_dir=data_dir, split='validation')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.dataset = dataset.map(lambda e: tokenizer(e['sentence'], 
                            truncation=True, padding='max_length', max_length=128), batched=True)    
    def __getitem__(self, idx):
        input_ids_data = self.dataset[idx]['input_ids']
        segment_ids_data = self.dataset[idx]['token_type_ids']
        input_mask_data = self.dataset[idx]['attention_mask']
        label_data = self.dataset[idx]['label']
        
        return (np.array(input_ids_data).astype('int32'), 
                np.array(segment_ids_data).astype('int32'),
                np.array(input_mask_data).astype('int32')), label_data

    def __len__(self):
        return len(self.dataset)
