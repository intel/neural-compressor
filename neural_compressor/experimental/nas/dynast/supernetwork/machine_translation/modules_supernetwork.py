#noqa: D100
# https://github.com/mit-han-lab/hardware-aware-transformers/blob/master/LICENSE
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

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.modules.module import _addindent

from neural_compressor.utils.utility import LazyImport

fairseq = LazyImport("fairseq")

INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__

    # assign a unique ID to each module instance, so that incremental state is
    # not shared across module instances
    if not hasattr(module_instance, '_fairseq_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._fairseq_instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]

    return '{}.{}.{}'.format(module_name, module_instance._fairseq_instance_id, key)


def get_incremental_state(module, incremental_state, key):  #noqa: D102
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):  #noqa: D102
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


class EmbeddingSuper(nn.Embedding):  #noqa: D101
    def __init__(self, num_embeddings, super_embed_dim, padding_idx, *args, **kwargs):  #noqa: D107
        super().__init__(num_embeddings, super_embed_dim, padding_idx, *args, **kwargs)

        # the largest embed dim
        self.super_embed_dim = {
            'encoder': super_embed_dim, 'decoder': super_embed_dim}

        # the current sampled embed dim
        self.sample_embed_dim = {'encoder': None, 'decoder': None}

        self.samples = {'encoder': {}, 'decoder': {}}
        self.profiling = False
        self.reset_parameters()

    def profile(self, mode=True):  #noqa: D102
        self.profiling = mode

    def reset_parameters(self):  #noqa: D102
        super().reset_parameters()
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim ** -0.5)
        nn.init.constant_(self.weight[self.padding_idx], 0)

    def set_sample_config(self, sample_embed_dim, part):  #noqa: D102
        self.sample_embed_dim[part] = sample_embed_dim
        self._sample_parameters(part)

    def _sample_parameters(self, part):
        weight = self.weight[..., :self.sample_embed_dim[part]]
        self.samples[part]['weight'] = weight

        return self.samples

    def sample_parameters(self, part, resample=False):  #noqa: D102
        return self._sample_parameters(part) if self.profiling or resample else self.samples

    def sampled_weight(self, part):  #noqa: D102
        return self.sample_parameters(part)[part]['weight']

    def forward(self, input, part='encoder'):  #noqa: D102
        return F.embedding(
            input,
            self.sampled_weight(part),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class LinearSuper(nn.Linear):  #noqa: D101
    def __init__(self, super_in_dim, super_out_dim, bias=True, uniform_=None, non_linear='linear'):  #noqa: D107
        super().__init__(super_in_dim, super_out_dim, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}

        self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

    def profile(self, mode=True):  #noqa: D102
        self.profiling = mode

    def sample_parameters(self, resample=False):  #noqa: D102
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _reset_parameters(self, bias, uniform_, non_linear):
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(
            self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.)

    def set_sample_config(self, sample_in_dim, sample_out_dim):  #noqa: D102
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim

        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = sample_weight(
            self.weight, self.sample_in_dim, self.sample_out_dim)
        self.samples['bias'] = self.bias
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)
        return self.samples

    def forward(self, x):  #noqa: D102
        self.sample_parameters()
        return F.linear(x, self.samples['weight'], self.samples['bias'])

    def calc_sampled_param_num(self):  #noqa: D102
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel


def sample_weight(weight, sample_in_dim, sample_out_dim):  #noqa: D103
    sample_weight = weight[:, :sample_in_dim]
    sample_weight = sample_weight[:sample_out_dim, :]

    return sample_weight


def sample_bias(bias, sample_out_dim):  #noqa: D103
    sample_bias = bias[:sample_out_dim]

    return sample_bias


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):  #noqa: D103
    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class LayerNormSuper(torch.nn.LayerNorm):  #noqa: D101
    def __init__(self, super_embed_dim):  #noqa: D107
        super().__init__(super_embed_dim)

        # the largest embed dim
        self.super_embed_dim = super_embed_dim

        # the current sampled embed dim
        self.sample_embed_dim = None

        self.samples = {}
        self.profiling = False

    def profile(self, mode=True):  #noqa: D102
        self.profiling = mode

    def sample_parameters(self, resample=False):  #noqa: D102
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _sample_parameters(self):
        self.samples['weight'] = self.weight[:self.sample_embed_dim]
        self.samples['bias'] = self.bias[:self.sample_embed_dim]
        return self.samples

    def set_sample_config(self, sample_embed_dim):  # noqa: D102
        self.sample_embed_dim = sample_embed_dim
        self._sample_parameters()

    def forward(self, x):  # noqa: D102
        self.sample_parameters()
        return F.layer_norm(
            x,
            (self.sample_embed_dim,),
            weight=self.samples['weight'],
            bias=self.samples['bias'],
            eps=self.eps,
        )

    def calc_sampled_param_num(self):  # noqa: D102
        assert 'weight' in self.samples.keys()
        assert 'bias' in self.samples.keys()
        return self.samples['weight'].numel() + self.samples['bias'].numel()


class MultiheadAttentionSuper(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, super_embed_dim, num_heads, is_encoder, super_kdim=None, super_vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False, out_dim=None, qkv_dim=None):  # noqa: D107
        super().__init__()

        # the configs of super arch
        self.super_q_embed_dim = super_embed_dim
        self.super_kv_embed_dim = None

        # the configs of current sampled arch
        self.sample_q_embed_dim = None
        self.sample_kv_embed_dim = None

        if super_kdim is not None:
            assert super_kdim == super_vdim
            self.super_kv_embed_dim = super_kdim
        else:
            self.super_kv_embed_dim = self.super_q_embed_dim

        if qkv_dim is None:
            self.qkv_dim = self.super_q_embed_dim
        else:
            self.qkv_dim = qkv_dim

        # this qkv same dim means the input dim for qkv are the same, not the output dim
        # self.qkv_same_dim = self.kdim == self.super_embed_dim and self.vdim == self.super_embed_dim
        self.qkv_same_dim = self.super_kv_embed_dim == self.super_q_embed_dim
        self.encoder = is_encoder

        # Caution! these actually are the sampled num_heads, head_dim and scaling
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = self.qkv_dim // num_heads
        assert self.head_dim * num_heads == self.qkv_dim, "qkv must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and ' \
                                                             'value to be of the same size'

        if self.qkv_same_dim:
            self.in_proj_weight = Parameter(torch.Tensor(
                3 * self.qkv_dim, self.super_q_embed_dim))
        else:
            self.k_proj_weight = Parameter(torch.Tensor(
                self.qkv_dim, self.super_kv_embed_dim))
            self.v_proj_weight = Parameter(torch.Tensor(
                self.qkv_dim, self.super_kv_embed_dim))
            self.q_proj_weight = Parameter(torch.Tensor(
                self.qkv_dim, self.super_q_embed_dim))

        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * self.qkv_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        if out_dim is None:
            out_dim = self.super_q_embed_dim
        self.out_proj = LinearSuper(
            super_in_dim=self.qkv_dim, super_out_dim=out_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, self.super_q_embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, self.super_q_embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

        self.enable_torch_version = False
        if hasattr(F, "multi_head_attention_forward"):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False
        self.enable_torch_version = False

    def calc_sampled_param_num(self):  # noqa: D102
        assert self.in_proj_weight is not None and self.in_proj_bias is not None
        in_proj_q_weight_numel = self.sample_q_embed_dim * self.qkv_dim
        in_proj_v_weight_numel = in_proj_k_weight_numel = self.sample_kv_embed_dim * self.qkv_dim
        in_proj_bias_numel = self.in_proj_bias.numel()

        # does not count in the output proj because it will be counted in LinearSuper layer
        # out_proj_weight_numel = self.qkv_dim * self.sample_q_embed_dim
        # out_proj_bias_numel = self.

        return in_proj_q_weight_numel + in_proj_k_weight_numel + in_proj_v_weight_numel + in_proj_bias_numel

    def set_sample_config(self, sample_q_embed_dim, sample_attention_heads, sample_kv_embed_dim=None):  # noqa: D102
        self.sample_q_embed_dim = sample_q_embed_dim
        if sample_kv_embed_dim is None:
            self.sample_kv_embed_dim = sample_q_embed_dim
        else:
            self.sample_kv_embed_dim = sample_kv_embed_dim

        self.num_heads = sample_attention_heads
        self.head_dim = self.qkv_dim // self.num_heads
        assert self.head_dim * \
            self.num_heads == self.qkv_dim, "qkv_dim must be divisible by sampled num_heads"
        self.scaling = self.head_dim ** -0.5

        self.out_proj.set_sample_config(
            sample_in_dim=self.qkv_dim, sample_out_dim=self.sample_q_embed_dim)

    def prepare_for_onnx_export_(self):  # noqa: D102
        self.onnx_trace = True

    def reset_parameters(self):  # noqa: D102
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """Input shape: Time x Batch x Channel.

        Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        tgt_len, bsz, embed_dim = query.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.in_proj_k(key)
                v = self.in_proj_v(key)

        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)

        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(
                    bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(
                    bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(
                bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(
                bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None

        if key_padding_mask is not None:
            fil = key_padding_mask.new_ones(
                key_padding_mask.size(0), src_len-key_padding_mask.size(1))
            key_padding_mask = torch.cat((key_padding_mask, fil), dim=1)
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat(
                [k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat(
                [v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(
            attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [
            bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                attn_weights = torch.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    attn_weights.float()
                ).type_as(attn_weights)
            else:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                )
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len)

        attn_weights = fairseq.utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace,
        ).type_as(attn_weights)
        attn_weights = F.dropout(
            attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)

        assert list(attn.size()) == [
            bsz * self.num_heads, tgt_len, self.head_dim]

        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, self.qkv_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(
                tgt_len, bsz, self.qkv_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len)

            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):  # noqa: D102
        return self._in_proj(query, sample_dim=self.sample_q_embed_dim).chunk(3, dim=-1)

    def in_proj_q(self, query):  # noqa: D102
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.qkv_dim, sample_dim=self.sample_q_embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[:self.qkv_dim]
            return F.linear(query, self.q_proj_weight[..., :self.sample_q_embed_dim], bias)

    def in_proj_k(self, key):  # noqa: D102
        if self.qkv_same_dim:
            return self._in_proj(key, start=self.qkv_dim, end=2 * self.qkv_dim, sample_dim=self.sample_kv_embed_dim)
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.qkv_dim:2 * self.qkv_dim]
            return F.linear(key, weight[..., :self.sample_kv_embed_dim], bias)

    def in_proj_v(self, value):  # noqa: D102
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.qkv_dim, sample_dim=self.sample_kv_embed_dim)
        else:
            weight = self.v_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[2 * self.qkv_dim:]
            return F.linear(value, weight[..., :self.sample_kv_embed_dim], bias)

    def _in_proj(self, input, sample_dim, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :sample_dim]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )

    def apply_sparse_mask(self, attn_weights, tgt_len, src_len, bsz):  # noqa: D102
        return attn_weights

    def __repr__(self):  # noqa: D105
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '\tnum_heads:' + str(self.num_heads) + \
            '\t qkv_dim:' + str(self.qkv_dim)
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str
