
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
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
from functools import partial
from ...utils import logger

DEBUG = False 

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
        "transformers_name": None,
        "transformers": None,
        "transformers": {}, # todo
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

def log_quantizable_layers_per_transformer(transformer_blocks, layers=[nn.Conv2d, nn.Conv1d, nn.Linear, transformers.Conv1D]):
    """Print all layers which will be quantized in GPTQ algorithm."""
    logger.info("* * Layer to be quantized * *")

    for block_id in range(len(transformer_blocks['transformers'])):
        transformer_block = transformer_blocks['transformers'][block_id]
        layers_for_this_tblock = find_layers_name(transformer_block)
        layer_names = [(transformer_blocks['transformers_name'] + "." + str(block_id) + '.' + layer_name) for layer_name in layers_for_this_tblock]
        for name in layer_names:
            logger.info(name)

#===========================================

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
        device=None
    ):
        """
        Args:
            model: the fp32 model to quantize
            weight_config (dict, optional): contains all info required by GPTQ. Defaults to {}.
                For example, 
                    weight_config={
                        'bits': 4, 
                        'group_size': 32, 
                        'sym': True,
                        'actorder': False
                        'percdamp': .01
                    }
            dataloader: an iterable containing calibration datasets, contains (inputs, targets)
            device: cpu or cuda
        """
        # model
        self.model = model
        # weight config related
        self.weight_config = weight_config
        self.wbits = 4
        self.percdamp = 0.01
        self.sym = True
        self.actorder = False
        self.group_size = 128
        self.process_config()
        # data & device
        self.dataloader = dataloader
        self.nsamples = len(dataloader)
        self.device = device
        self.is_ready = False

        self.use_cache = model.config.use_cache
        self.gptq_related_blocks = trace_gptq_target_blocks(model) # get the transformer block list above
        log_quantizable_layers_per_transformer(self.gptq_related_blocks)
        #self.pre_transformer_layers = trace_embeddings_layers(model) # get the embeddings above

        # import pdb;pdb.set_trace()
        # initialize buffers which are essential for gptq computation. 
        try:
            self.dtype = next(iter(self.model.parameters())).dtype
            self.inp = torch.zeros(
                (self.nsamples, model.seqlen, model.config.hidden_size), 
                dtype=self.dtype, 
                device=self.device
            )
            self.cache = {'i': 0}
            # for opt, bloom, llama, etc, their inputs are different thus their cache structures vary
            # initialization
            # for special_input_terms in arch_inputs[self.args.arch]:
            #     self.cache[special_input_terms] = None
            self.out = torch.zeros_like(self.inp)
            self.is_ready = True
        except:
            pass

    def process_config(self):
        """Copy arguments from weight_config to build-in attributes."""
        self.wbits = self.weight_config.get('wbits', self.wbits)
        self.percdamp = self.weight_config.get('perdamo', self.percdamp)
        self.sym = self.weight_config.get('sym', self.sym)
        self.group_size = self.weight_config.get('group_size', self.sym)
        self.actorder = self.weight_config.get('actorder', self.sym)
    
    @torch.no_grad()
    def pre_quantization(self):
        """Prepare input calibration data and other attributes which are critical for gptq execution."""
        # critical: hooker function which collects inputs
        def forward(layer, hidden_states, **kwargs):
            # inputs[inputs_info['idx']] = input_ids # TODO solve the problem of batchsize!=1
            self.inp[self.cache['i']] = hidden_states
            self.cache['i'] += 1
            for arg in kwargs:
                if isinstance(kwargs[arg], torch.Tensor):
                    self.cache[arg] = kwargs[arg]
                else:
                    continue
            raise ValueError

        # Step1: fetch the embeddings and other layers before the transformer stack.
        for embedding_name, embedding_layer in self.gptq_related_blocks["embeddings"].items():
            embedding_layer = embedding_layer.to(self.device)

        # obtain the first layer inputs and registered to inputs
        self.gptq_related_blocks['transformers'][0] = self.gptq_related_blocks['transformers'][0].to(self.device)

        # Step 2: use partial to modify original forward function
        forward_cache = self.gptq_related_blocks['transformers'][0].forward
        self.gptq_related_blocks['transformers'][0].forward = \
            partial(forward, self.gptq_related_blocks['transformers'][0])

        logger.info("Collecting calibration inputs...")
        # import pdb;pdb.set_trace()
        for batch in tqdm(self.dataloader):
            try:
                self.model(batch[0].to(self.device))
            except ValueError:
                pass
        logger.info("Done.")
        # import pdb;pdb.set_trace()
        # restore original forward function
        self.gptq_related_blocks['transformers'][0].forward = forward_cache

        self.gptq_related_blocks['transformers'][0] = self.gptq_related_blocks['transformers'][0].cpu()
        # after store inputs, locate embedding layers and transformer[0] back to cpu
        for embedding_name, embedding_layer in self.gptq_related_blocks["embeddings"].items():
            embedding_layer.to(self.device)
        torch.cuda.empty_cache()
        logger.info('GPTQ quantization prepared.')

    @torch.no_grad()
    def execute_quantization(self, means=None, stds=None):
        """Run quantization."""
        logger.info("Begin ====>")
        # import pdb;pdb.set_trace()
        self.pre_quantization()

        quantizers = {}

        # import pdb;pdb.set_trace()
        tblock_length = len(self.gptq_related_blocks['transformers'])
        # Triggle GPTQ algorithm block by block.
        for block_idx in range(tblock_length):
            logger.info(f"Quantizing layer {block_idx + 1} / {tblock_length}..")
            transformer_block = self.gptq_related_blocks['transformers'][block_idx].to(self.device)
            # trace all layers which can be quantized (Linear, Conv2d, etc.)
            sub_layers = find_layers(transformer_block)
            gptq_for_this_block = {}
            for layer_name in sub_layers:
                gptq_for_this_block[layer_name] = GPTQ(sub_layers[layer_name])
                gptq_for_this_block[layer_name].quantizer = Quantizer()
                gptq_for_this_block[layer_name].quantizer.configure(self.wbits,perchannel=True,sym=self.sym,mse=False)

            def add_batch(_name):
                def tmp(_, inp, out):
                    gptq_for_this_block[_name].add_batch(inp[0].data, out.data)
                return tmp
            
            # register handles which add inputs and outputs to gptq object
            handles = []
            
            for layer_name in sub_layers:
                handles.append(sub_layers[layer_name].register_forward_hook(add_batch(layer_name)))

            idx = self.cache.pop('i')
            for j in range(self.nsamples):
                # during the forward process, the batch data has been registered into gptq object.
                # use dict passing
                self.out[j] = transformer_block(self.inp[j].unsqueeze(0), **self.cache)[0]
            self.cache['i'] = idx
            for h in handles:
                h.remove()
            
            for layer_name in sub_layers:
                logger.info(f"Quantizing layer {layer_name}")
                gptq_for_this_block[layer_name].fasterquant(percdamp=self.percdamp, groupsize=self.group_size, actorder=self.actorder)
                quantizers['%d.%s' % (block_idx, layer_name)] = gptq_for_this_block[layer_name].quantizer
                gptq_for_this_block[layer_name].free()

            idx = self.cache.pop('i')
            for j in range(self.nsamples):
                self.out[j] = transformer_block(self.inp[j].unsqueeze(0), **self.cache)[0]
                # self.out[j] = self.perform_transformer_forward(transformer_block, self.inp[j].unsqueeze(0))
            self.cache['i'] = idx
            self.gptq_related_blocks['transformers'][block_idx] = transformer_block.cpu()
            del gptq_for_this_block
            torch.cuda.empty_cache()
            # iteratively replace the input with output (next block)
            self.inp, self.out = self.out, self.inp
            print('+------------------+--------------+------------+-----------+-------+')
            print('\n')
        
        # import pdb;pdb.set_trace()
        logger.info("Quantization done")
        self.model.config.use_cache = self.use_cache

        return quantizers
    
    @torch.no_grad()
    def post_quantization(self, test_dataloader):
        pass # gptq model can be evaluate using itrex optimized lm_eval

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

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t()) # H = X*X, which should be a sysm matrix

    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False):
        # import pdb;pdb.set_trace()
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
        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.device)
        H[diag, diag] += damp # add a average value of 
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

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

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                logger.info(f"{torch.sum((self.layer(self.inp1) - self.out1) ** 2)}")
                logger.info(f"{torch.sum(Losses)}")

        if self.device != torch.device('cpu'):
            torch.cuda.synchronize()
        logger.info(f'time {(time.time() - tick)}')
        logger.info(f'error {torch.sum(Losses).item()}')

        if actorder:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            logger.info(f"{torch.sum((self.layer(self.inp1) - self.out1) ** 2)}")

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

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)
#======================================================

#==================dataloader related==========================
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

class GPTQLoader(object):
    """Generate a dataloader which fits gptq inputs from user's own datasets and models (tokenizers)."""

    def __init__(self, dataset, tokenizer, nsamples = 128, seqlen = 2048):
        self.dataset = dataset # dataloader should be a iterable of text (to support more items)
        self.tokenizer = tokenizer # transformer_tokenizers
        self.nsamples = nsamples
        self.seqlen = seqlen
    
    def get_gptq_dataloader(self, seed = 0):
        """Generate the datasets."""
        import random
        random.seed(seed)
        gptq_loader = []
        tokenizer_config = {
            "truncation": True,
            "max_length": self.seqlen,
            "return_tensors": "pt",
            "padding": "max_length",
        }
        for _ in range(self.nsamples):
            i = random.randint(0, len(self.dataset) - 1)
            try:
                data = self.tokenizer(self.dataset[i], **tokenizer_config).input_ids
            except:
                raise NotImplementedError
            if data.shape[-1] > self.seqlen:
                j = random.randint(0, data.shape[-1] - self.seqlen - 1)
                k = j + self.seqlen
                data = data[:, j:k]
            tar = data.clone()
            tar[:, :-1] = -100
            gptq_loader.append((data, tar))
        return gptq_loader 
