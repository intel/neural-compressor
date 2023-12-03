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

""" Load a checkpoint file of pretrained transformer to a model in pytorch """

import numpy as np
import tensorflow as tf
import torch

def load_param(checkpoint_file, conversion_table):
    """
    Load parameters in pytorch model from checkpoint file according to conversion_table
    checkpoint_file : pretrained checkpoint model file in tensorflow
    conversion_table : { pytorch tensor in a model : checkpoint variable name }
    """
    for pyt_param, tf_param_name in conversion_table.items():
        tf_param = tf.train.load_variable(checkpoint_file, tf_param_name)

        # for weight(kernel), we should do transpose
        if tf_param_name.endswith('kernel'):
            tf_param = np.transpose(tf_param)

        assert pyt_param.size() == tf_param.shape, \
            'Dim Mismatch: %s vs %s ; %s' % \
                (tuple(pyt_param.size()), tf_param.shape, tf_param_name)
        
        # assign pytorch tensor from tensorflow param
        pyt_param.data = torch.from_numpy(tf_param)


def load_model(model, checkpoint_file):
    """ Load the pytorch model from tensorflow checkpoint file """

    # Embedding layer
    e, p = model.embed, 'bert/embeddings/'
    load_param(checkpoint_file, {
        e.tok_embed.weight: p+"word_embeddings",
        e.pos_embed.weight: p+"position_embeddings",
        e.seg_embed.weight: p+"token_type_embeddings",
        e.norm.gamma:       p+"LayerNorm/gamma",
        e.norm.beta:        p+"LayerNorm/beta"
    })

    # Transformer blocks
    for i in range(len(model.blocks)):
        b, p = model.blocks[i], "bert/encoder/layer_%d/"%i
        load_param(checkpoint_file, {
            b.attn.proj_q.weight:   p+"attention/self/query/kernel",
            b.attn.proj_q.bias:     p+"attention/self/query/bias",
            b.attn.proj_k.weight:   p+"attention/self/key/kernel",
            b.attn.proj_k.bias:     p+"attention/self/key/bias",
            b.attn.proj_v.weight:   p+"attention/self/value/kernel",
            b.attn.proj_v.bias:     p+"attention/self/value/bias",
            b.proj.weight:          p+"attention/output/dense/kernel",
            b.proj.bias:            p+"attention/output/dense/bias",
            b.pwff.fc1.weight:      p+"intermediate/dense/kernel",
            b.pwff.fc1.bias:        p+"intermediate/dense/bias",
            b.pwff.fc2.weight:      p+"output/dense/kernel",
            b.pwff.fc2.bias:        p+"output/dense/bias",
            b.norm1.gamma:          p+"attention/output/LayerNorm/gamma",
            b.norm1.beta:           p+"attention/output/LayerNorm/beta",
            b.norm2.gamma:          p+"output/LayerNorm/gamma",
            b.norm2.beta:           p+"output/LayerNorm/beta",
        })


def load_embedding(embed, checkpoint_file):
    """ Load the pytorch model from tensorflow checkpoint file """

    # Embedding layer
    e, p = embed, 'bert/embeddings/'
    load_param(checkpoint_file, {
        e.tok_embed.weight: p+"word_embeddings",
        e.pos_embed.weight: p+"position_embeddings",
        e.seg_embed.weight: p+"token_type_embeddings",
        e.norm.gamma:       p+"LayerNorm/gamma",
        e.norm.beta:        p+"LayerNorm/beta"
    })
