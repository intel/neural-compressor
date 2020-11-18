# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
#
# Copyright (c) 2020 Intel Corporation
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

""" Transformer Model Classes & Config Class """

import math
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import split_last, merge_last

from torch.quantization import QuantWrapper, QuantStub, DeQuantStub

class Config(NamedTuple):
    """ Configuration for BERT model """
    vocab_size: int = None # Size of Vocabulary
    dim: int = 768 # Dimension of Hidden Layer in Transformer Encoder
    n_layers: int = 12 # Numher of Hidden Layers
    n_heads: int = 12 # Numher of Heads in Multi-Headed Attention Layers
    dim_ff: int = 768*4 # Dimension of Intermediate Layers in Positionwise Feedforward Net
    #activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    p_drop_hidden: float = 0.1 # Probability of Dropout of various Hidden Layers
    p_drop_attn: float = 0.1 # Probability of Dropout of Attention Layers
    max_len: int = 512 # Maximum Length for Positional Embeddings
    n_segments: int = 2 # Number of Sentence Segments
    comments: list = [] # for comments in json file


def gelu(x):
    """ Implementation of the gelu activation function by Hugging Face """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    """ A layernorm module in the TF style (epsilon inside the square root). """
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.dim))
        self.beta  = nn.Parameter(torch.zeros(cfg.dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """ The embedding module from word, position and token_type embeddings. """
    def __init__(self, cfg):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim) # token embedding
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.dim) # position embedding
        self.seg_embed = nn.Embedding(cfg.n_segments, cfg.dim) # segment(token type) embedding

        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x) # (S,) -> (B, S)

        e = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.drop(self.norm(e))


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.dim, cfg.dim)
        self.proj_k = nn.Linear(cfg.dim, cfg.dim)
        self.proj_v = nn.Linear(cfg.dim, cfg.dim)
        self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, cfg.dim_ff)
        self.fc2 = nn.Linear(cfg.dim_ff, cfg.dim)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class Block(nn.Module):
    """ Transformer Block """
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, mask):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        self.blocks = nn.ModuleList([
            Block(cfg) for _ in range(cfg.n_layers)])

    def forward(self, x, seg, mask):
        h = self.embed(x, seg)
        for block in self.blocks:
            h = block(h, mask)
        return h




### Models for classification ###


class Classifier4Transformer(nn.Module):
    """ Classifier with Transformer """
    def __init__(self, cfg, n_labels):
        super().__init__()
        self.transformer = Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, n_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.transformer(input_ids, segment_ids, input_mask)
        # only use the first h in the sequence
        pooled_h = self.activ(self.fc(h[:, 0])) 
        logits = self.classifier(self.drop(pooled_h))
        return logits

"""
class kimCNN(nn.Module):
    def __init__(self, cfg, n_labels, channels=[768,1000,1000], kernel_sizes=[3,3]):
        super().__init__()
        self.embed = Embeddings(cfg)
        self.convs = nn.ModuleList([
            nn.Conv1d(n_in, n_out, k, padding=k)
            for n_in, n_out, k in zip(channels[:-1], channels[1:], kernel_sizes)])
        
    def forward(self, input_ids, segment_ids, input_mask):
        h = self.embed(input_ids, segment_ids)
        for conv in self.convs:
            h = conv(h)
"""

class BlendCNN(nn.Module):
    "Blend CNN in https://openreview.net/pdf?id=HJxM3hftiX"
    def __init__(self, cfg, 
                 n_labels,
                 channels=[768, 100, 100, 100, 100, 100, 100, 100, 100],
                 kernel_sizes=[5, 5, 5, 5, 5, 5, 5, 5],
                 n_hidden_dense=100):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.conv_quant = QuantStub()
        self.conv_dequant = DeQuantStub()

        self.embed = Embeddings(cfg)
        self.convs = nn.ModuleList([
            nn.Conv1d(n_in, n_out, k, padding=int(k/2))
            for n_in, n_out, k in zip(channels[:-1], channels[1:], kernel_sizes)
        ])
        self.dense = nn.Linear(channels[-1]*len(kernel_sizes), n_hidden_dense)
        self.classifier = nn.Linear(n_hidden_dense, n_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.embed(input_ids, segment_ids).transpose(1, 2)
        output = []
        h = self.conv_quant(h)
        for conv in self.convs:
            h = F.relu(conv(h))
            output.append(h.mean(2))
        
        output = [self.conv_dequant(item) for item in output]
        output = torch.cat(output, 1)
        output = self.quant(output)
        logits = self.classifier(F.relu(self.dense(output)))
        logist = self.dequant(logits)
        return logits


#class GatedCNN(nn.Module):
#    def __init__(self, cfg,

