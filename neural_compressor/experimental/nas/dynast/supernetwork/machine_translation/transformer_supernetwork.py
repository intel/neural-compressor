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

import math

import torch
import torch.nn.functional as F
from torch import nn

from neural_compressor.utils import logger
from neural_compressor.utils.utility import LazyImport

from .modules_supernetwork import (EmbeddingSuper, LayerNormSuper, LinearSuper,
                                   MultiheadAttentionSuper)

fairseq = LazyImport("fairseq")

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


class TransformerSuperNetwork(fairseq.models.BaseFairseqModel):
    """Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)`.

    <https://arxiv.org/abs/1706.03762>

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, task):  #noqa: D107
        super().__init__()

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        encoder_config = {'encoder_embed_dim': 640,
                          'encoder_layers': 6,
                          'encoder_attention_heads': 8,
                          'encoder_ffn_embed_dim': 3072,
                          'encoder_embed_path': None}

        decoder_config = {'decoder_embed_dim': 640,
                          'decoder_layers': 6,
                          'decoder_attention_heads': 8,
                          'decoder_ffn_embed_dim': 3072}

        encoder_embed_tokens = self.build_embedding(
            src_dict, encoder_config['encoder_embed_dim'], encoder_config['encoder_embed_path']
        )
        decoder_embed_tokens = encoder_embed_tokens
        self.share_decoder_input_output_embed = True

        self.encoder = TransformerEncoder(
            encoder_config, src_dict, encoder_embed_tokens)
        self.decoder = TransformerDecoder(
            decoder_config, tgt_dict, decoder_embed_tokens)

    def build_embedding(self, dictionary, embed_dim, path=None):  #noqa: D102
        utils = fairseq.utils

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    def profile(self, mode=True):  #noqa: D102
        for module in self.modules():
            if hasattr(module, 'profile') and self != module:
                module.profile(mode)

    def get_sampled_params_numel(self, config):  #noqa: D102
        self.set_sample_config(config)
        numels = []
        for name, module in self.named_modules():
            if hasattr(module, 'calc_sampled_param_num'):
                # a hacky way to skip the layers that exceed encoder-layer-num or decoder-layer-num
                if (
                    name.split('.')[0] == 'encoder'
                    and eval(name.split('.')[2]) >= config['encoder']['encoder_layer_num']
                ):
                    continue
                if (
                    name.split('.')[0] == 'decoder'
                    and eval(name.split('.')[2]) >= config['decoder']['decoder_layer_num']
                ):
                    continue

                numels.append(module.calc_sampled_param_num())
        return sum(numels)

    def set_sample_config(self, config):  #noqa: D102
        logger.info('[DyNAS-T] Setting active configuration to {}'.format(config))
        self.encoder.set_sample_config(config)
        self.decoder.set_sample_config(config)

    def forward(self,src_tokens,src_lengths,prev_output_token):  #noqa: D102
         encoder_output = self.encoder.forward(src_tokens,src_lengths)
         output = self.decoder(prev_output_token,encoder_output)
         return output


class TransformerEncoder(fairseq.models.FairseqEncoder):
    """Transformer encoder consisting of *args.encoder_layers* layers.

    Each layer is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, encoder_config, dictionary, embed_tokens):  #noqa: D107
        super().__init__(dictionary)
        # the configs of super arch
        self.super_embed_dim = encoder_config['encoder_embed_dim']
        self.super_ffn_embed_dim = [
            encoder_config['encoder_ffn_embed_dim']] * encoder_config['encoder_layers']
        self.super_layer_num = encoder_config['encoder_layers']
        self.super_self_attention_heads = [
            encoder_config['encoder_attention_heads']] * encoder_config['encoder_layers']

        self.super_dropout = 0.3
        self.super_activation_dropout = 0

        self.super_embed_scale = math.sqrt(self.super_embed_dim)

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_ffn_embed_dim = None
        self.sample_layer_num = None
        self.sample_self_attention_heads = None

        self.sample_dropout = None
        self.sample_activation_dropout = None

        self.sample_embed_scale = None

        self.register_buffer('version', torch.Tensor([3]))

        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS

        self.embed_tokens = embed_tokens

        self.embed_positions = fairseq.modules.PositionalEmbedding(
            self.max_source_positions, self.super_embed_dim, self.padding_idx,
            learned=False,
        )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(encoder_config, layer_idx=i)
            for i in range(self.super_layer_num)
        ])

        if False:
            self.layer_norm = LayerNormSuper(self.super_embed_dim)
        else:
            self.layer_norm = None

        self.vocab_original_scaling = False

    def set_sample_config(self, config: dict):  #noqa: D102

        self.sample_embed_dim = config['encoder']['encoder_embed_dim']

        # Caution: this is a list for all layers
        self.sample_ffn_embed_dim = config['encoder']['encoder_ffn_embed_dim']

        self.sample_layer_num = config['encoder']['encoder_layer_num']

        # Caution: this is a list for all layers
        self.sample_self_attention_heads = config['encoder']['encoder_self_attention_heads']

        self.sample_dropout = calc_dropout(
            self.super_dropout, self.sample_embed_dim, self.super_embed_dim)
        self.sample_activation_dropout = calc_dropout(
            self.super_activation_dropout, self.sample_embed_dim, self.super_embed_dim)

        self.sample_embed_scale = math.sqrt(
            self.sample_embed_dim) if not self.vocab_original_scaling else self.super_embed_scale

        self.embed_tokens.set_sample_config(
            sample_embed_dim=self.sample_embed_dim, part='encoder')

        if self.layer_norm is not None:
            self.layer_norm.set_sample_config(
                sample_embed_dim=self.sample_embed_dim)

        for i, layer in enumerate(self.layers):
            # not exceed sample layer number
            if i < self.sample_layer_num:
                layer.set_sample_config(is_identity_layer=False,
                                        sample_embed_dim=self.sample_embed_dim,
                                        sample_ffn_embed_dim_this_layer=self.sample_ffn_embed_dim[i],
                                        sample_self_attention_heads_this_layer=self.sample_self_attention_heads[
                                            i],
                                        sample_dropout=self.sample_dropout,
                                        sample_activation_dropout=self.sample_activation_dropout)
            # exceeds sample layer number
            else:
                layer.set_sample_config(is_identity_layer=True)

    def forward(self, src_tokens, src_lengths):
        """Forward function.

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.sample_embed_scale * \
            self.embed_tokens(src_tokens, part='encoder')
        if self.embed_positions is not None:
            positions = self.embed_positions(src_tokens)

            # sample the positional embedding and add
            x += positions[..., :self.sample_embed_dim]
        x = F.dropout(x, p=self.sample_dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        all_x = []
        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            all_x.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,
            'encoder_out_all': all_x,
            'encoder_padding_mask': encoder_padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        # need to reorder each layer of output
        if 'encoder_out_all' in encoder_out.keys():
            new_encoder_out_all = []
            for encoder_out_one_layer in encoder_out['encoder_out_all']:
                new_encoder_out_all.append(
                    encoder_out_one_layer.index_select(1, new_order))
            encoder_out['encoder_out_all'] = new_encoder_out_all

        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        utils = fairseq.utils
        if isinstance(self.embed_positions, fairseq.modules.SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(
                name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i))

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoder(fairseq.models.FairseqIncrementalDecoder):
    """Transformer decoder consisting of *args.decoder_layers* layers.

    Each layer is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, decoder_config, dictionary, embed_tokens, no_encoder_attn=False):  #noqa: D107
        super().__init__(dictionary)

        # the configs of super arch
        self.super_embed_dim = decoder_config['decoder_embed_dim']
        self.super_ffn_embed_dim = decoder_config['decoder_ffn_embed_dim'] * \
            decoder_config['decoder_layers']
        self.super_layer_num = decoder_config['decoder_layers']
        self.super_self_attention_heads = 8 * \
            [decoder_config['decoder_attention_heads']] * \
            decoder_config['decoder_layers']
        self.super_ende_attention_heads = [
            decoder_config['decoder_attention_heads']] * decoder_config['decoder_layers']
        self.super_arbitrary_ende_attn = [-1] * \
            decoder_config['decoder_layers']

        self.super_dropout = 0.3
        self.super_activation_dropout = 0.0

        self.super_embed_scale = math.sqrt(self.super_embed_dim)

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_ffn_embed_dim = None
        self.sample_layer_num = None
        self.sample_self_attention_heads = None
        self.sample_ende_attention_heads = None
        self.sample_arbitrary_ende_attn = None

        self.sample_dropout = None
        self.sample_activation_dropout = None

        self.sample_embed_scale = None

        # the configs of current sampled arch
        self.register_buffer('version', torch.Tensor([3]))

        self.share_input_output_embed = True

        self.output_embed_dim = decoder_config['decoder_embed_dim']

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        self.embed_tokens = embed_tokens

        self.embed_positions = fairseq.modules.PositionalEmbedding(
            self.max_target_positions, self.super_embed_dim, padding_idx,
            learned=False,
        ) if not False else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(
                decoder_config, layer_idx=i, no_encoder_attn=no_encoder_attn)
            for i in range(self.super_layer_num)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(self.super_embed_dim, self.output_embed_dim, bias=False) \
            if self.super_embed_dim != self.output_embed_dim else None

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(
                len(dictionary), self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0,
                            std=self.output_embed_dim ** -0.5)

        self.layer_norm = None
        self.get_attn = False

        self.vocab_original_scaling = False

    def set_sample_config(self, config: dict):  #noqa: D102

        self.sample_embed_dim = config['decoder']['decoder_embed_dim']
        self.sample_encoder_embed_dim = config['encoder']['encoder_embed_dim']

        # Caution: this is a list for all layers
        self.sample_ffn_embed_dim = config['decoder']['decoder_ffn_embed_dim']

        # Caution: this is a list for all layers
        self.sample_self_attention_heads = config['decoder']['decoder_self_attention_heads']

        # Caution: this is a list for all layers
        self.sample_ende_attention_heads = config['decoder']['decoder_ende_attention_heads']

        self.sample_arbitrary_ende_attn = config['decoder']['decoder_arbitrary_ende_attn']

        self.sample_layer_num = config['decoder']['decoder_layer_num']

        self.sample_dropout = calc_dropout(
            self.super_dropout, self.sample_embed_dim, self.super_embed_dim)
        self.sample_activation_dropout = calc_dropout(
            self.super_activation_dropout, self.sample_embed_dim, self.super_embed_dim)

        self.sample_embed_scale = math.sqrt(
            self.sample_embed_dim) if not self.vocab_original_scaling else self.super_embed_scale

        self.embed_tokens.set_sample_config(
            sample_embed_dim=self.sample_embed_dim, part='decoder')

        if self.layer_norm is not None:
            self.layer_norm.set_sample_config(
                sample_embed_dim=self.sample_embed_dim)

        for i, layer in enumerate(self.layers):
            # not exceed sample layer number
            if i < self.sample_layer_num:
                layer.set_sample_config(is_identity_layer=False,
                                        sample_embed_dim=self.sample_embed_dim,
                                        sample_encoder_embed_dim=self.sample_encoder_embed_dim,
                                        sample_ffn_embed_dim_this_layer=self.sample_ffn_embed_dim[i],
                                        sample_self_attention_heads_this_layer=self.sample_self_attention_heads[
                                            i],
                                        sample_ende_attention_heads_this_layer=self.sample_ende_attention_heads[
                                            i],
                                        sample_dropout=self.sample_dropout,
                                        sample_activation_dropout=self.sample_activation_dropout)
            # exceeds sample layer number
            else:
                layer.set_sample_config(is_identity_layer=True)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        """Forward pass.

        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state)
        x = self.output_layer(x)
        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        """Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if positions is not None:
            positions = positions[..., :self.sample_embed_dim]

        if incremental_state is not None:
            # only take the last token in to the decoder
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.sample_embed_scale * \
            self.embed_tokens(prev_output_tokens, part='decoder')

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.sample_dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        attns = []
        inner_states = [x]

        # decoder layers
        for i, layer in enumerate(self.layers):
            encoder_out_feed = None
            encoder_padding_mask_feed = None

            if encoder_out is not None:
                # only use the last layer
                if i >= self.sample_layer_num or self.sample_arbitrary_ende_attn[i] == -1:
                    encoder_out_feed = encoder_out['encoder_out']
                # concat one second last output layer
                elif self.sample_arbitrary_ende_attn[i] == 1:
                    encoder_out_feed = torch.cat(
                        [encoder_out['encoder_out'], encoder_out['encoder_out_all'][-2]], dim=0)
                elif self.sample_arbitrary_ende_attn[i] == 2:
                    encoder_out_feed = torch.cat(
                        [encoder_out['encoder_out'],
                        encoder_out['encoder_out_all'][-2],
                        encoder_out['encoder_out_all'][-3]],
                        dim=0)
                else:
                    raise NotImplementedError(
                        "arbitrary_ende_attn should in [-1, 1, 2]")

            if encoder_out['encoder_padding_mask'] is not None:
                if i >= self.sample_layer_num or self.sample_arbitrary_ende_attn[i] == -1:
                    encoder_padding_mask_feed = encoder_out['encoder_padding_mask']
                # concat one more
                elif self.sample_arbitrary_ende_attn[i] == 1:
                    encoder_padding_mask_feed = torch.cat(
                        [encoder_out['encoder_padding_mask'], encoder_out['encoder_padding_mask']], dim=1)
                # concat two more
                elif self.sample_arbitrary_ende_attn[i] == 2:
                    encoder_padding_mask_feed = torch.cat(
                        [encoder_out['encoder_padding_mask'],
                        encoder_out['encoder_padding_mask'],
                        encoder_out['encoder_padding_mask']],
                        dim=1)
                else:
                    raise NotImplementedError(
                        "arbitrary_ende_attn should in [-1, 1, 2]")

            x, attn = layer(
                x,
                encoder_out_feed,
                encoder_padding_mask_feed,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(
                    x) if incremental_state is None else None,
            )
            inner_states.append(x)
            attns.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)  # pylint: disable=not-callable

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        if not self.get_attn:
            attns = attns[-1]
        return x, {'attn': attns, 'inner_states': inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.sampled_weight('decoder'))
            else:
                return F.linear(features, self.embed_out[:, :self.sample_embed_dim])
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):  #noqa: D102
        utils = fairseq.utils

        dim = tensor.size(0)
        if (
            not hasattr(self, '_future_mask')
            or self._future_mask is None  # pylint: disable=access-member-before-definition
            or self._future_mask.device != tensor.device  # pylint: disable=access-member-before-definition
            or self._future_mask.size(0) < dim  # pylint: disable=access-member-before-definition
        ):
            self._future_mask = torch.triu(  # pylint: disable=access-member-before-definition
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]  # pylint: disable=access-member-before-definition

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        utils = fairseq.utils
        if isinstance(self.embed_positions, fairseq.modules.SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(
                name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(
                        name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(
                            name, i, new, m)] = state_dict[k]
                        del state_dict[k]

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, encoder_config, layer_idx):  #noqa: D107
        super().__init__()

        utils = fairseq.utils

        # the configs of super arch
        self.super_embed_dim = encoder_config['encoder_embed_dim']
        self.super_ffn_embed_dim_this_layer = encoder_config['encoder_ffn_embed_dim']
        self.super_self_attention_heads_this_layer = encoder_config['encoder_attention_heads']

        self.super_dropout = 0.3
        self.super_activation_dropout = 0

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_self_attention_heads_this_layer = None

        self.sample_dropout = None
        self.sample_activation_dropout = None

        self.is_identity_layer = None

        self.qkv_dim = 512

        self.self_attn = MultiheadAttentionSuper(
            super_embed_dim=self.super_embed_dim, num_heads=self.super_self_attention_heads_this_layer,
            is_encoder=True, dropout=0.1, self_attention=True, qkv_dim=self.qkv_dim,
        )

        self.self_attn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.dropout = 0.1
        self.activation_fn = utils.get_activation_fn(
            activation='relu'
        )
        self.normalize_before = False

        self.fc1 = LinearSuper(super_in_dim=self.super_embed_dim, super_out_dim=self.super_ffn_embed_dim_this_layer,
                               uniform_=None, non_linear='relu')  # init.uniform_
        self.fc2 = LinearSuper(super_in_dim=self.super_ffn_embed_dim_this_layer,
                               super_out_dim=self.super_embed_dim, uniform_=None, non_linear='linear')
        self.final_layer_norm = LayerNormSuper(self.super_embed_dim)

    def set_sample_config(
        self,
        is_identity_layer,
        sample_embed_dim=None,
        sample_ffn_embed_dim_this_layer=None,
        sample_self_attention_heads_this_layer=None,
        sample_dropout=None,
        sample_activation_dropout=None,
    ):  #noqa: D102

        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False

        self.sample_embed_dim = sample_embed_dim
        self.sample_ffn_embed_dim_this_layer = sample_ffn_embed_dim_this_layer
        self.sample_self_attention_heads_this_layer = sample_self_attention_heads_this_layer

        self.sample_dropout = sample_dropout
        self.sample_activation_dropout = sample_activation_dropout

        self.self_attn_layer_norm.set_sample_config(
            sample_embed_dim=self.sample_embed_dim)

        self.self_attn.set_sample_config(sample_q_embed_dim=self.sample_embed_dim,
                                         sample_attention_heads=self.sample_self_attention_heads_this_layer)

        self.fc1.set_sample_config(
            sample_in_dim=self.sample_embed_dim, sample_out_dim=self.sample_ffn_embed_dim_this_layer)
        self.fc2.set_sample_config(
            sample_in_dim=self.sample_ffn_embed_dim_this_layer, sample_out_dim=self.sample_embed_dim)

        self.final_layer_norm.set_sample_config(
            sample_embed_dim=self.sample_embed_dim)

    def upgrade_state_dict_named(self, state_dict, name):
        """Renames keys in state dict.

        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {
            '0': 'self_attn_layer_norm',
            '1': 'final_layer_norm'
        }
        for old, new in layer_norm_map.items():
            for m in ('weight', 'bias'):
                k = '{}.layer_norms.{}.{}'.format(name, old, m)
                if k in state_dict:
                    state_dict[
                        '{}.{}.{}'.format(name, new, m)
                    ] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask=None):
        """Forward pass.

        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if self.is_identity_layer:
            return x
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.byte(), -1e8)
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        # TODO: to formally solve this problem, we need to change fairseq's
        # MultiheadAttention. We will do this later on.
        x, _ = self.self_attn(query=x, key=x, value=x,
                              key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x[:residual.size(0), :, :] = residual + x[:residual.size(0), :, :]
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.sample_activation_dropout,
                      training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):  #noqa: D102
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        decoder_config,
        layer_idx,
        no_encoder_attn=False,
        add_bias_kv=False,
        add_zero_attn=False,
    ):  #noqa: D107
        super().__init__()

        utils = fairseq.utils

        # the configs of super arch
        self.super_embed_dim = decoder_config['decoder_embed_dim']
        self.super_encoder_embed_dim = decoder_config['decoder_embed_dim']
        self.super_ffn_embed_dim_this_layer = decoder_config['decoder_ffn_embed_dim']
        self.super_self_attention_heads_this_layer = decoder_config['decoder_attention_heads']
        self.super_ende_attention_heads_this_layer = decoder_config['decoder_attention_heads']

        self.super_dropout = 0.3
        self.super_activation_dropout = 0

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_encoder_embed_dim = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_self_attention_heads_this_layer = None
        self.sample_ende_attention_heads_this_layer = None
        self.sample_dropout = None
        self.sample_activation_dropout = None
        self.is_identity_layer = None
        self.qkv_dim = 512
        self.layer_idx = layer_idx

        self.self_attn = MultiheadAttentionSuper(
            is_encoder=False,
            super_embed_dim=self.super_embed_dim,
            num_heads=self.super_self_attention_heads_this_layer,
            dropout=0.1,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
            qkv_dim=self.qkv_dim
        )
        self.activation_fn = utils.get_activation_fn(
            activation='relu'
        )
        self.normalize_before = False

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix

        self.self_attn_layer_norm = LayerNormSuper(self.super_embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttentionSuper(
                super_embed_dim=self.super_embed_dim,
                num_heads=self.super_ende_attention_heads_this_layer,
                is_encoder=False,
                super_kdim=self.super_encoder_embed_dim,
                super_vdim=self.super_encoder_embed_dim,
                dropout=0.1,
                encoder_decoder_attention=True,
                qkv_dim=self.qkv_dim
            )
            self.encoder_attn_layer_norm = LayerNormSuper(self.super_embed_dim)

        self.fc1 = LinearSuper(super_in_dim=self.super_embed_dim, super_out_dim=self.super_ffn_embed_dim_this_layer,
                               uniform_=None, non_linear='relu')
        self.fc2 = LinearSuper(super_in_dim=self.super_ffn_embed_dim_this_layer, super_out_dim=self.super_embed_dim,
                               uniform_=None, non_linear='linear')

        self.final_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.need_attn = True

        self.onnx_trace = False

    def set_sample_config(self,
        is_identity_layer,
        sample_embed_dim=None,
        sample_encoder_embed_dim=None,
        sample_ffn_embed_dim_this_layer=None,
        sample_self_attention_heads_this_layer=None,
        sample_ende_attention_heads_this_layer=None,
        sample_dropout=None,
        sample_activation_dropout=None,
    ):  #noqa: D102

        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False

        self.sample_embed_dim = sample_embed_dim
        self.sample_encoder_embed_dim = sample_encoder_embed_dim
        self.sample_ffn_embed_dim_this_layer = sample_ffn_embed_dim_this_layer
        self.sample_self_attention_heads_this_layer = sample_self_attention_heads_this_layer
        self.sample_ende_attention_heads_this_layer = sample_ende_attention_heads_this_layer

        self.sample_dropout = sample_dropout
        self.sample_activation_dropout = sample_activation_dropout

        self.self_attn_layer_norm.set_sample_config(
            sample_embed_dim=self.sample_embed_dim)
        self.encoder_attn_layer_norm.set_sample_config(
            sample_embed_dim=self.sample_embed_dim)

        self.self_attn.set_sample_config(sample_q_embed_dim=self.sample_embed_dim,
                                         sample_attention_heads=self.sample_self_attention_heads_this_layer)
        self.encoder_attn.set_sample_config(
            sample_q_embed_dim=self.sample_embed_dim,
            sample_kv_embed_dim=self.sample_encoder_embed_dim,
            sample_attention_heads=self.sample_ende_attention_heads_this_layer,
        )

        self.fc1.set_sample_config(
            sample_in_dim=self.sample_embed_dim, sample_out_dim=self.sample_ffn_embed_dim_this_layer)
        self.fc2.set_sample_config(
            sample_in_dim=self.sample_ffn_embed_dim_this_layer, sample_out_dim=self.sample_embed_dim)

        self.final_layer_norm.set_sample_config(
            sample_embed_dim=self.sample_embed_dim)

    def prepare_for_onnx_export_(self):  #noqa: D102
        self.onnx_trace = True

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        incremental_state=None,
        prev_self_attn_state=None,
        prev_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        """Forward pass.

        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if self.is_identity_layer:
            return x, None

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(
                self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(
                    incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.sample_dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(
                self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.sample_activation_dropout,
                      training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):  #noqa: D102
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):  #noqa: D102
        self.need_attn = need_attn


def calc_dropout(dropout, sample_embed_dim, super_embed_dim):  #noqa: D103
    return dropout * 1.0 * sample_embed_dim / super_embed_dim


def Embedding(num_embeddings, embedding_dim, padding_idx):  #noqa: D103
    return EmbeddingSuper(num_embeddings, embedding_dim, padding_idx=padding_idx)


def Linear(in_features, out_features, bias=True, uniform_=None, non_linear='linear'):  #noqa: D103
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight) if uniform_ is None else uniform_(  #noqa: D103
        m.weight, non_linear=non_linear)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m
