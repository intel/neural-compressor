#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
#

try:
    from neural_compressor.utils.utility import LazyImport

    torch = LazyImport("torch")
    from ...utils import logger
except:  # pragma: no cover
    import logging

    import torch

    logger = logging.getLogger()
from typing import Union


def get_module(model, key):
    """Get module from model by key name.

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
    """
    module = model
    name_list = key.split(".")
    for name in name_list:
        if hasattr(module, name):
            module = getattr(module, name)
            module = module
    return module


def set_module(model, key, new_module):
    """Set new module into model by key name.

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
        new_module (torch.nn.Module): new module to be inserted
    """
    module = model
    name_list = key.split(".")
    for name in name_list[:-1]:
        if hasattr(module, name):
            module = getattr(module, name)
        else:
            module = module
    setattr(module, name_list[-1], new_module)


class OPTRoundQuantizer(object):
    def __init__(self,
                 model,
                 tokenizer=None,
                 bits=4,
                 group_size=128,
                 scheme="asym",
                 # weight_config={},##TODO support later
                 enable_full_range=False,  ##for symmetric, TODO support later
                 optimizer=None,
                 lr_scheduler=None,
                 dataloader=None,
                 default_dataset_name="NeelNanda/pile-10k",
                 dataset_split="train",
                 bs=8,
                 amp=True,
                 amp_dtype="float16",##could we get dtype from model?
                 device="",
                 use_quant_input=True,
                 enable_minmax_tuning=True,
                 seqlen=2048,
                 nsamples=512,
                 seed=42,
                 data_type="int",
                 **kwargs
                 ):
        """
        Args:
            model:
            data_type:
            bits:
            group_size:
            scheme:
            weight_config:
             weight_config={
                   'layer1':##layer_name
                   {
                       'data_type': 'int',
                       'bits': 4,
                       'group_size': 32,
                       'scheme': "sym", ## or asym
                   }
                   ...
               }

            optimizer:
            lr_scheduler:
            enable_full_range:
            **kwargs:

        Returns:
        """
        self.model = model
        self.bits = bits
        self.group_size = group_size
        self.scheme = scheme
        self.data_type = data_type
        self.supported_types = [torch.nn.Linear]  ## TODO support conv1d
        self.weight_config = {}
        assert (dataloader != None or tokenizer != None)  ##TODO datatype
        if dataloader is None:
            self.dataloader = self.get_default_dataloader(data_name=default_dataset_name)
        else:
            self.dataloader = dataloader

        self.dataset_split = dataset_split
        self.seed = seed
        self.tokenizer = tokenizer
        self.seqlen = self.seqlen
        self.bs = bs

    def get_default_dataloader(self, data_name="NeelNanda/pile-10k"):
        from datasets import load_dataset
        from torch.utils.data import DataLoader
        seqlen = self.seqlen

        @torch.no_grad()
        def collate_batch(batch):
            input_ids_new = []
            for text in batch:
                input_ids = text["input_ids"]
                if input_ids.shape[0] < seqlen:
                    continue
                input_ids = input_ids[:seqlen]
                input_ids_list = input_ids.tolist()
                if input_ids_list.count(input_ids_list[-1]) > seqlen // 2:
                    continue
                input_ids_new.append(input_ids)
            if len(input_ids_new) == 0:
                return None
            tmp = torch.vstack(input_ids_new)
            res = {}
            res["input_ids"] = tmp
            return res

        def default_tokenize_function(self, examples):
            example = self.tokenizer(examples["text"], truncation=True, max_length=self.seqlen)
            return example

        calib_dataset = load_dataset(data_name, split=self.dataset_split)
        calib_dataset = calib_dataset.shuffle(seed=self.seed)
        calib_dataset = calib_dataset.map(default_tokenize_function, batched=True)
        calib_dataset.set_format(type='torch', columns=['input_ids'])
        calib_dataloader = DataLoader(
            calib_dataset,
            batch_size=self.bs,
            shuffle=False,
            collate_fn=collate_batch
        )
        return calib_dataloader

    #
    # def load_model(self):
    #     from transformers import AutoModel,AutoTokenizer
    #
    #     model = AutoModel.from_pretrained(self.model)
    #     tokenizer = AutoTokenizer.from_pretrained(self.model)
    #     return

    def export(self):
        pass

    def check_weight_config(self):
        for n, m in self.model.named_modules():
            if not type(m) in self.supported_types:
                continue
            if n in self.weight_config.keys():
                pass
            else:
                pass
