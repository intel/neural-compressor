
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

import math
import time
import re
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
from functools import partial
from ...utils import logger
import random
from collections import UserDict, defaultdict

DEBUG = False 

# ================ device related ===================
def move_input_to_device(input, device=torch.device('cpu')):
    if isinstance(input, dict) or isinstance(input, UserDict):
        for inp in input.keys():
            input[inp] = input[inp].to(device) \
                if isinstance(input[inp], torch.Tensor) else input[inp]
    elif isinstance(input, list) or isinstance(input, tuple):
        input_res, prev_size = [], None
        for inp in input:
            if prev_size:
                if isinstance(inp, torch.Tensor):
                    if inp.size() == prev_size:
                        input_res.append(inp.to(device))
                else:
                    if torch.tensor(inp).size == prev_size:
                        input_res.append(inp)
            else:
                input_res.append(inp.to(device) \
                    if isinstance(inp, torch.Tensor) else inp)
            prev_size = torch.tensor(inp).size()
        input = input_res
    else:
        input = input.to(device)  # pylint: disable=no-member
    return input

# ==============model structure related==============
def is_leaf(module):
    """Judge whether a module has no child-modules.

    Args:
        module: torch.nn.Module

    Returns:
        a bool: whether a module has no child-modules.
    """
    children_cnt = 0
    for n in module.children():
        children_cnt += 1
    return True if children_cnt == 0 else False

def trace_gptq_target_blocks(module, module_types = [torch.nn.ModuleList]):
    """Search transformer stacked structures, which is critical in LLMs and GPTQ execution.

    Args:
        module: torch.nn.Module
        module_types: List of torch.nn.Module.

    Returns:
        gptq_related_blocks = {
            "embeddings": {}, # Dict embedding layers before transfromer stack module, 
            "transformers_pre": {}, # TODO
            "transformers_name": string. LLMs' transformer stack module name ,
            "transformers": torch.nn.ModuleList. LLMs' transformer stack module,
            "transformers": {}, Dict# TODO
        }
    """
    gptq_related_blocks = {
        "embeddings": {},
        "transformers_pre": {}, # todo
        "transformers_name": "", # None
        "transformers": [], # None
        "transformers_post": {}, # todo
    }
    for n, m in module.named_modules():
        if type(m) in module_types:
            gptq_related_blocks["transformers_name"] = n
            gptq_related_blocks["transformers"] = m
            return gptq_related_blocks
        else:
            if is_leaf(m):
                gptq_related_blocks["embeddings"][n] = m
    return gptq_related_blocks

def find_layers(module, layers=[nn.Conv2d, nn.Conv1d, nn.Linear, transformers.Conv1D], name=''):
    """Get all layers with target types."""
    if type(module) in layers:
        return {name: module}
    else:
        # use string type to find name:
        if type(module).__name__ in ['Linear']:
            return {name: module}
        else:
            pass
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def find_layers_name(module, layers=[nn.Conv2d, nn.Conv1d, nn.Linear, transformers.Conv1D], name=''):
    """Get all layers with target types."""
    if type(module) in layers:
        return [name]
    res = []
    for name1, child in module.named_children():
        res += find_layers_name(child, layers=layers, name = name + '.' + name1 if name != '' else name1)
    return res

def log_quantizable_layers_per_transformer(
        transformer_blocks, 
        layers=[nn.Conv2d, nn.Conv1d, nn.Linear, transformers.Conv1D]
    ):
    """Print all layers which will be quantized in GPTQ algorithm."""
    logger.info("* * Layer to be quantized * *")

    for block_id in range(len(transformer_blocks['transformers'])):
        transformer_block = transformer_blocks['transformers'][block_id]
        layers_for_this_tblock = find_layers_name(transformer_block)
        layer_names = [
            (transformer_blocks['transformers_name'] + "." + str(block_id) + '.' + layer_name) \
            for layer_name in layers_for_this_tblock
        ]
        for name in layer_names:
            logger.info(name)

#===============quantization related============================
def quantize(x, scale, zero, maxq):
    """Do quantization."""
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class GPTQuantizer(object):
    """Main API for GPTQ algorithm.
    Please refer to: 
    GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers
    url: https://arxiv.org/abs/2210.17323
    """
    
    def __init__(
        self, 
        model, 
        weight_config={}, 
        dataloader=None, 
        nsamples = 128, 
        use_max_length = True,
        device=None
    ):
        """
        Args:
            model: the fp32 model to quantize
            weight_config (dict, optional): contains all info required by GPTQ. Defaults to {}. For example, 
            weight_config={
                'layer1':
                {
                    'bits': 4, 
                    'group_size': 32, 
                    'sym': False,
                    'percdamp': .01,
                    'act_order': False
                }
                ...
            }
            dataloader: an iterable containing calibration datasets, contains (inputs, targets)
            device: cpu or cuda
        """
        # model
        self.model = model
        self.use_cache = self.model.config.use_cache
        self.gptq_related_blocks = trace_gptq_target_blocks(self.model) # get the transformer block list above
        self.dtype = next(iter(self.model.parameters())).dtype
        log_quantizable_layers_per_transformer(self.gptq_related_blocks)

        # weight config
        self.weight_config = weight_config
        # default settings, check configs
        self.wbits_default = 4
        self.group_size_default = 128
        self.block_size_default = 128
        self.percdamp_default = 0.01
        self.sym_default = False
        self.act_order_default = False
        self.perchannel_default = True
        self.mse_default = False
        self.check_layer_config()

        # device
        self.device = model.device
        self.is_ready = False

        # dataloader
        self.use_max_length = use_max_length
        self.dataloader_original = dataloader
        self.dataloader = []
        self.nsamples = nsamples
        self.prepare_dataloader()

    def prepare_dataloader(self):
        if self.use_max_length:
            # (Recommend) only take sequence whose length exceeds model.seqlen, 
            # which perserves calibration's tokens are all valid
            # This is GPTQ official dataloader implementation
            self.obtain_first_n_samples_fulllength()
            # initialize buffers which are essential for gptq computation.
            self.model_hidden_size = 2048
            self.initialize_inp_buffersize()
            try:
                # Since length is unified, we can allocate a continous space to store inputs
                self.inp = torch.zeros(
                    (len(self.dataloader), self.model.seqlen, self.model_hidden_size), 
                    dtype=self.dtype, 
                    device=self.device
                )
                self.cache = {'i': 0}
                self.out = torch.zeros_like(self.inp)
                self.is_ready = True
            except:
                logger.warning("GPTQ Quantizer initialization failed!")
                pass
        else:
            # general selection, no padding, not GPTQ original implementation.
            self.obtain_first_n_samples()
            try:
                self.inp = [torch.zeros(1) for _ in range(len(self.dataloader))]
                self.cache = {'i': 0}
                self.out = [torch.zeros(1) for _ in range(len(self.dataloader))]
                self.is_ready = True
            except:
                logger.warning("GPTQ Quantizer initialization failed!")
                pass

    def obtain_first_n_samples(self, seed=0):
        """Get first nsample data as the real calibration dataset."""
        self.dataloader.clear()
        random.seed(seed)
        for batch in self.dataloader_original:
            # process data, depends on its data type.
            if len(self.dataloader) == self.nsamples:
                break
            # list, tuple
            if isinstance(batch, list) or isinstance(batch, tuple):
                if batch[0].shape[-1] > self.model.seqlen:
                    i = random.randint(0, batch[0].shape[-1] - self.model.seqlen - 1)
                    j = i + self.model.seqlen
                    batch_final = batch[0][:, i:j]
                else:
                    batch_final = batch[0]
            # dict
            elif isinstance(batch, dict): # pragma: no cover
                try:
                    length = batch['input_ids'].shape[-1]
                except:
                    logger.warning("Please make sure your dict'like data contains key of 'input_ids'.")
                    continue
                batch_final = {}
                if length > self.model.seqlen:
                    i = random.randint(0, length - self.model.seqlen - 1)
                    j = i + self.model.seqlen
                    # may have to slice every sequence related data
                    for key in batch.keys():
                        if isinstance(batch[key], torch.Tensor):
                            batch_final[key] = batch[key][:, i:j] # slice on sequence length dim
                        else:
                            batch_final[key] = batch[key]
                else:
                    batch_final = batch
            # tensor
            else:
                if batch.shape[-1] > self.model.seqlen:
                    i = random.randint(0, batch.shape[-1] - self.model.seqlen - 1)
                    j = i + self.model.seqlen
                    batch_final = batch[:, i:j]
                else:
                    batch_final = batch
            self.dataloader.append(batch_final)

        if len(self.dataloader) < self.nsamples:
            logger.warning(f"Try to use {self.nsamples} data, but entire dataset size is {len(self.dataloader)}.")
    
    def obtain_first_n_samples_fulllength(self, seed=0):
        self.dataloader.clear()
        random.seed(seed)
        unified_length = self.model.seqlen
        for batch in self.dataloader_original:
            if len(self.dataloader) == self.nsamples:
                break
            # list & tuple
            if isinstance(batch, list) or isinstance(batch, tuple):
                if batch[0].shape[-1] == unified_length:
                    batch_final = batch[0]
                elif batch[0].shape[-1] > unified_length:
                    i = random.randint(0, batch[0].shape[-1] - unified_length - 1)
                    j = i + unified_length
                    batch_final = batch[0][:, i:j]
                else:
                    # not match max length, not include in target dataset
                    continue
                self.dataloader.append(batch_final)
            # dict
            elif isinstance(batch, dict):
                try:
                    length = batch['input_ids'].shape[-1]
                except:
                    logger.warning("Please make sure your dict'like data contains key of 'input_ids'.")
                    continue
                batch_final = {}
                if length == self.model.seqlen:
                    batch_final = batch
                elif length > self.model.seqlen:
                    i = random.randint(0, length - self.model.seqlen - 1)
                    j = i + self.model.seqlen
                    # may have to slice every sequence related data
                    for key in batch.keys():
                        if isinstance(batch[key], torch.Tensor):
                            batch_final[key] = batch[key][:, i:j] # slice on sequence length dim with same position
                        else:
                            batch_final[key] = batch[key]
                else:
                    # not match max length, not include in target dataset
                    continue
            # tensor
            else:
                if batch.shape[-1] == unified_length:
                    batch_final = batch
                elif batch.shape[-1] > unified_length:
                    i = random.randint(0, batch.shape[-1] - unified_length - 1)
                    j = i + unified_length
                    batch_final = batch[:, i:j]
                else:
                    # not match max length, not include in target dataset
                    continue
            self.dataloader.append(batch_final)
        if len(self.dataloader) < self.nsamples: # pragma: no cover
            logger.warning(f"Trying to allocate {self.nsamples} data with fixed length {unified_length}, \
            but only {len(self.dataloader)} samples satisfy your setting. You may choose smaller 'model.seqlen' value.")

    @torch.no_grad()
    def initialize_inp_buffersize(self):
        # Run a forward and generate proper buffer tensor
        # Thus, no need to pass hidden_states dimension parameters of model.config
        # e.g. OPT's hidden_states dimension can be called by model.config.hidden_size
        # but mpt's hidden_states dimension can be called by model.config.d_model
        def forward(layer, hidden_states, **kwargs):
            # inputs[inputs_info['idx']] = input_ids # TODO solve the problem of batchsize!=1
            logger.info(f"The hidden_states shape along transformers blocks is {hidden_states.shape}.")
            self.model_hidden_size = hidden_states.shape[-1]
            raise ValueError

        # Step1: fetch the embeddings and other layers before the transformer stack.
        for embedding_name, embedding_layer in self.gptq_related_blocks["embeddings"].items():
            embedding_layer = embedding_layer.to(self.device)

        # Step2: modify the first transformer block's forward function to obtain inputs for calibration
        self.gptq_related_blocks['transformers'][0] = self.gptq_related_blocks['transformers'][0].to(self.device)
        forward_cache = self.gptq_related_blocks['transformers'][0].forward
        self.gptq_related_blocks['transformers'][0].forward = \
            partial(forward, self.gptq_related_blocks['transformers'][0])

        # Step3: run forward to obtain calibration datasets
        logger.info("Collecting calibration inputs...")
        for batch in self.dataloader:
            batch = move_input_to_device(batch, self.device)
            try:
                if isinstance(batch, tuple) or isinstance(batch, list):
                    self.model(batch[0])
                elif isinstance(batch, dict):
                    self.model(**batch)
                else:
                    self.model(batch.to(self.device))
            except ValueError:
                break

        # Step 4: restore original forward function, relocate layers back to cpu.
        self.gptq_related_blocks['transformers'][0].forward = forward_cache
        self.gptq_related_blocks['transformers'][0] = self.gptq_related_blocks['transformers'][0].cpu()
        for embedding_name, embedding_layer in self.gptq_related_blocks["embeddings"].items():
            embedding_layer.to(self.device)
        torch.cuda.empty_cache()
        
    def get_full_layer_name(self, sub_layer_name, block_idx):
        transformer_name = self.gptq_related_blocks["transformers_name"]
        return ".".join([transformer_name, str(block_idx), sub_layer_name])

    def check_layer_config(self):
        """Copy arguments from weight_config to build-in attributes."""
        if 'wbits' in self.weight_config:
            tmp_weight_config = {}
            for name, module in self.model.named_modules():
                tmp_weight_config[name] = {}
                tmp_weight_config[name]['wbits'] = self.weight_config.get('wbits', self.wbits_default)
                tmp_weight_config[name]['group_size'] = self.weight_config.get('group_size', self.group_size_default)
                tmp_weight_config[name]['block_size'] = self.weight_config.get('block_size', self.group_size_default)
                tmp_weight_config[name]['percdamp'] = self.weight_config.get('pecdamp', self.percdamp_default)
                tmp_weight_config[name]['sym'] = self.weight_config.get('sym', self.sym_default)
                tmp_weight_config[name]['act_order'] = self.weight_config.get('act_order', self.act_order_default)
                tmp_weight_config[name]['perchannel'] = self.weight_config.get('perchannel', self.perchannel_default)
                tmp_weight_config[name]['mse'] = self.weight_config.get('mse', self.mse_default)
            self.weight_config = tmp_weight_config
        else:
            for layer_name, config in self.weight_config.items():
                self.weight_config[layer_name]['wbits'] = config.get('wbits', self.wbits_default)
                self.weight_config[layer_name]['group_size'] = config.get('group_size', self.group_size_default)
                self.weight_config[layer_name]['block_size'] = config.get('block_size', self.group_size_default)
                self.weight_config[layer_name]['percdamp'] = config.get('pecdamp', self.percdamp_default)
                self.weight_config[layer_name]['sym'] = config.get('sym', self.sym_default)
                self.weight_config[layer_name]['act_order'] = config.get('act_order', self.act_order_default)
                self.weight_config[layer_name]['perchannel'] = config.get('perchannel', self.perchannel_default)
                self.weight_config[layer_name]['mse'] = config.get('mse', self.mse_default)

    def get_layer_config(self, layer_name):
        """Obtain config for one layer, since GPTQ supports layer-wise config."""
        # First try the exact name matching, if cannot find, use re to search. For example, can support ".*" in op_name
        config = None
        config = self.weight_config.get(layer_name, None)
        if config != None:
            return config
        else:
            for k, v in self.weight_config.items():
                regex = re.compile(k)
                if len(regex.findall(layer_name)) != None:
                    config = v
                    return config
                else:
                    pass
        return config

    @torch.no_grad()
    def pre_quantization(self):
        """Prepare input calibration data and other attributes which are critical for gptq execution."""
        # critical: hooker function which collects inputs
        def forward(layer, hidden_states, **kwargs):
            # inputs[inputs_info['idx']] = input_ids # TODO solve the problem of batchsize!=1
            self.inp[self.cache['i']] = hidden_states
            self.cache['i'] += 1
            for arg in kwargs:
                # TODO: investigate include parameters
                if self.use_max_length:
                    if isinstance(kwargs[arg], torch.Tensor) or arg == "alibi":
                        self.cache[arg] = kwargs[arg]
                    else:
                        continue
                else:
                    # each outputs can be different shape, hence also use list to store
                    if isinstance(kwargs[arg], torch.Tensor) or arg == "alibi":
                        if self.cache.get(arg, None) == None:
                            self.cache[arg] = []
                        self.cache[arg].append(kwargs[arg])
                    continue
            raise ValueError

        # Step1: fetch the embeddings and other layers before the transformer stack.
        for embedding_name, embedding_layer in self.gptq_related_blocks["embeddings"].items():
            embedding_layer = embedding_layer.to(self.device)

        # Step2: modify the first transformer block's forward function to obtain inputs for calibration
        self.gptq_related_blocks['transformers'][0] = self.gptq_related_blocks['transformers'][0].to(self.device)
        forward_cache = self.gptq_related_blocks['transformers'][0].forward
        self.gptq_related_blocks['transformers'][0].forward = \
            partial(forward, self.gptq_related_blocks['transformers'][0])

        # Step3: run forward to obtain calibration datasets
        logger.info("Collecting calibration inputs...")
        for batch in tqdm(self.dataloader):
            batch = move_input_to_device(batch, self.device)
            try:
                if isinstance(batch, tuple) or isinstance(batch, list):
                    self.model(batch[0])
                elif isinstance(batch, dict):
                    self.model(**batch)
                else:
                    self.model(batch)
            except ValueError:
                pass
        # output inp data shape
        logger.info("All calibration data's shape =>")
        for idx in range(len(self.dataloader)):
            logger.info(self.inp[idx].shape)
        logger.info("Done.")

        # Step 4: restore original forward function, relocate layers back to cpu.
        self.gptq_related_blocks['transformers'][0].forward = forward_cache
        self.gptq_related_blocks['transformers'][0] = self.gptq_related_blocks['transformers'][0].cpu()
        for embedding_name, embedding_layer in self.gptq_related_blocks["embeddings"].items():
            embedding_layer.to(self.device)
        torch.cuda.empty_cache()
        # end
        logger.info('GPTQ quantization prepared.')

    def gather_single_batch_from_dict(self, data_dict, idx):
        single_batch = {}
        for k, v in data_dict.items():
            single_batch[k] = data_dict[k][idx]
        return single_batch

    @torch.no_grad()
    def execute_quantization(self, means=None, stds=None):
        """Run quantization."""
        # Step1: prepare quantization (calibration datasets)
        logger.info("Begin ====>")
        self.pre_quantization()
        # Step2: run gptq quantization in a transformer block-wise manner.
        gptq_config = {}
        tblock_length = len(self.gptq_related_blocks['transformers'])
        for block_idx in range(tblock_length):
            logger.info(f"Quantizing layer {block_idx + 1} / {tblock_length}..")
            transformer_block = self.gptq_related_blocks['transformers'][block_idx].to(self.device)
            # Step2.1: obtain all layers (Linear, Conv2d, etc) in the block which can be quantized.
            sub_layers = find_layers(transformer_block)
            sub_layers_to_quant = {}
            for layer_name, layer_obj in sub_layers.items():
                # filter sub_layers with included layer_names in self.weight_config
                full_layer_name = self.get_full_layer_name(layer_name, block_idx)
                # if self.weight_config.get(full_layer_name, None) == None:
                if self.get_layer_config(full_layer_name) == None:
                    logger.warning(f"{full_layer_name} can be quantized " + "but excluded from quantization configs.")
                else:
                    sub_layers_to_quant[layer_name] = layer_obj
            del sub_layers
            sub_layers = sub_layers_to_quant
            # Step 2.2: Initailize GPTQ quantizers for collected layers.
            gptq_for_this_block = {}
            # initialize gptq quantizer for every layer in a transformer block
            for layer_name in sub_layers:
                # weight_config_this_layer = self.weight_config.get(
                #     self.get_full_layer_name(layer_name, block_idx), None
                # )
                weight_config_this_layer = self.get_layer_config(
                    self.get_full_layer_name(layer_name, block_idx)
                )
                gptq_for_this_block[layer_name] = GPTQ(sub_layers[layer_name])
                #gptq_for_this_block[layer_name].quantizer = Quantizer()
                gptq_for_this_block[layer_name].quantizer.configure(
                    weight_config_this_layer['wbits'],
                    weight_config_this_layer['perchannel'],
                    weight_config_this_layer['sym'],
                    weight_config_this_layer['mse'],
                )
            # Step 2.3: modify forward functions to hook inputs data (used in gptq execution)
            def add_batch(_name):
                def tmp(_, inp, out):
                    gptq_for_this_block[_name].add_batch(inp[0].data, out.data)
                return tmp
            handles = [] # register handles which add inputs and outputs to gptq object
            for layer_name in sub_layers:
                handles.append(sub_layers[layer_name].register_forward_hook(add_batch(layer_name)))
            idx = self.cache.pop('i')
            for j in range(len(self.dataloader)):
                if self.use_max_length:
                    # self.inp[j] shape: [seq_len, hidden_size]
                    self.out[j] = transformer_block(self.inp[j].unsqueeze(0), **self.cache)[0]
                else:
                    # self.inp[j] shape: [1, seq_len, hidden_size] (batchsize is 1 by default)
                    cache_batch = self.gather_single_batch_from_dict(self.cache, j)
                    self.out[j] = transformer_block(self.inp[j], **cache_batch)[0]
            self.cache['i'] = idx
            for h in handles:
                h.remove()
            # Step 2.4: everything is prepared, so start quantization!
            for layer_name in sub_layers:
                # weight_config_this_layer = self.weight_config.get(
                #     self.get_full_layer_name(layer_name, block_idx), None
                # )
                weight_config_this_layer = self.get_layer_config(
                    self.get_full_layer_name(layer_name, block_idx)
                )
                logger.info(f"Quantizing layer {layer_name}")
                scale, zp = gptq_for_this_block[layer_name].fasterquant(
                    blocksize = weight_config_this_layer['block_size'],
                    percdamp = weight_config_this_layer['percdamp'], 
                    groupsize = weight_config_this_layer['group_size'], 
                    act_order = weight_config_this_layer['act_order'],
                )
                gptq_config[self.get_full_layer_name(layer_name, block_idx)] = {'scale': scale}
                if not weight_config_this_layer['sym']:
                    gptq_config[self.get_full_layer_name(layer_name, block_idx)]['zero'] = zp
                if weight_config_this_layer['act_order']: # save perm for restoring the weights
                    gptq_config[self.get_full_layer_name(layer_name, block_idx)]['perm'] = \
                                                                gptq_for_this_block[layer_name].perm
                gptq_for_this_block[layer_name].free()
            
            # Step 2.5: replace output data with quantized weights
            idx = self.cache.pop('i')
            for j in range(len(self.dataloader)):
                if self.use_max_length:
                    # self.inp[j] shape: [seq_len, hidden_size]
                    self.out[j] = transformer_block(self.inp[j].unsqueeze(0), **self.cache)[0]
                else:
                    # self.inp[j] shape: [1, seq_len, hidden_size] (batchsize is 1 by default)
                    cache_batch = self.gather_single_batch_from_dict(self.cache, j)
                    self.out[j] = transformer_block(self.inp[j], **cache_batch)[0]
            self.cache['i'] = idx
            self.gptq_related_blocks['transformers'][block_idx] = transformer_block.cpu()
            del gptq_for_this_block
            torch.cuda.empty_cache()
            # iteratively replace the input with output, thus layerwise quantization can continue.
            self.inp, self.out = self.out, self.inp
            logger.info('------------------------------')
        
        logger.info("Quantization done")
        self.model.config.use_cache = self.use_cache

        # obtain model (all weight only quantization API function should return)
        for k, v in gptq_config.items():
            for m, n in v.items():
                gptq_config[k][m] = n.tolist()
        return self.model, gptq_config

class GPTQ:
    """
    Please refer to: 
    GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers (https://arxiv.org/abs/2210.17323)
    """
    def __init__(self, layer):
        self.layer = layer
        self.device = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d) or isinstance(self.layer, nn.Conv1d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0] # output channels
        self.columns = W.shape[1] # input channels
        self.H = torch.zeros((self.columns, self.columns), device=self.device)
        self.nsamples = 0
        self.quantizer = Quantizer()
        self.perm = None # act_order choice

    def add_batch(self, inp, out):
        # if DEBUG:
        #     self.inp1 = inp
        #     self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        # TODO: llm's transformer sequential with nn.conv2d is currently not under test
        # if isinstance(self.layer, nn.Conv2d):
        #     unfold = nn.Unfold(
        #         self.layer.kernel_size,
        #         dilation=self.layer.dilation,
        #         padding=self.layer.padding,
        #         stride=self.layer.stride
        #     )
        #     inp = unfold(inp)
        #     inp = inp.permute([1, 0, 2])
        #     inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t()) # H = X*X, which should be a sysm matrix

    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1, act_order=False):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0 # such channel makes no contribution to quantization computation

        # rearrange considering the diag's value
        if act_order:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            self.perm = perm.clone()

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.device)
        H[diag, diag] += damp # add a average value of 
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        scale = []
        zero = []

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count): # within a block, channel wise
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                        scale.append(self.quantizer.scale)
                        zero.append(self.quantizer.zero)

                q = quantize(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            # if DEBUG:
            #     self.layer.weight.data[:, :i2] = Q[:, :i2]
            #     self.layer.weight.data[:, i2:] = W[:, i2:]
            #     logger.info(f"{torch.sum((self.layer(self.inp1) - self.out1) ** 2)}")
            #     logger.info(f"{torch.sum(Losses)}")

        if self.device != torch.device('cpu'):
            torch.cuda.synchronize()
        logger.info(f'time {(time.time() - tick)}')
        logger.info(f'error {torch.sum(Losses).item()}')

        if act_order:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            logger.info(f"{torch.sum((self.layer(self.inp1) - self.out1) ** 2)}")

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)
        return scale, zero

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()

class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True, 
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False
    ):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        if trits:
            self.maxq = torch.tensor(-1) 

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
          self.scale = xmax
          self.zero = xmin
        else:
          self.scale = (xmax - xmin) / self.maxq
          if self.sym:
              self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
          else:
              self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    # def quantize(self, x):
    #     if self.ready():
    #         return quantize(x, self.scale, self.zero, self.maxq)
    #     return x

    # def enabled(self):
    #     return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)
