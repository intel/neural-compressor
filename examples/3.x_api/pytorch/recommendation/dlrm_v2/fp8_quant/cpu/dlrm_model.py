#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Intel Corporation
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

import torch

from torchrec.models.dlrm import SparseArch, InteractionDCNArch, DLRM_DCN
from torchrec.modules.embedding_modules import EmbeddingBagCollection

from typing import List, Optional
import numpy as np


def _calculate_fan_in_and_fan_out(shape):
    # numpy array version
    dimensions = len(shape)
    assert (
        dimensions >= 2
    ), "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"

    num_input_fmaps = shape[1]
    num_output_fmaps = shape[0]
    receptive_field_size = 1
    if len(shape) > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _calculate_correct_fan(shape, mode):
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError(
            "Mode {} not supported, please use one of {}".format(mode, valid_modes)
        )

    fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
    return fan_in if mode == "fan_in" else fan_out


def calculate_gain(nonlinearity, param=None):
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return np.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif (
            not isinstance(param, bool)
            and isinstance(param, int)
            or isinstance(param, float)
        ):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return np.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == "selu":
        return (
            3.0 / 4
        )  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def xavier_norm_(shape: tuple, gain: float = 1.0):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
    std = gain * np.sqrt(2.0 / float(fan_in + fan_out))
    mean = 0.0
    d = np.random.normal(mean, std, size=shape).astype(np.float32)
    return d


def kaiming_uniform_(
    shape: tuple, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
):
    assert 0 not in shape, "Initializing zero-element tensors is a no-op"
    fan = _calculate_correct_fan(shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / np.sqrt(fan)
    bound = np.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return np.random.uniform(-bound, bound, size=shape)

class _LowRankCrossNet(torch.nn.Module):
    def __init__(
        self,
        lr_crossnet,
    ) -> None:
        super().__init__()
        self._num_layers = lr_crossnet._num_layers
        self._in_features = lr_crossnet.bias[0].shape[0]
        self._low_rank = lr_crossnet._low_rank
        self.V_linears = torch.nn.ModuleList()
        self.W_linears = torch.nn.ModuleList()
        for i in range(self._num_layers):
            self.V_linears.append(
                torch.nn.Linear(self._in_features, self._low_rank, bias=False)
            )
            self.W_linears.append(
                torch.nn.Linear(self._low_rank, self._in_features, bias=True)
            )
            self.V_linears[i].weight.data = lr_crossnet.V_kernels[i]
            self.W_linears[i].weight.data = lr_crossnet.W_kernels[i]
            self.W_linears[i].bias.data = lr_crossnet.bias[i]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_0 = input
        x_l = x_0
        for layer in range(self._num_layers):
            x_l_v = self.V_linears[layer](x_l)
            x_l_w = self.W_linears[layer](x_l_v)
            x_l = x_0 * x_l_w + x_l  # (B, N)
        return x_l


def replace_crossnet(dlrm):
    crossnet = dlrm.inter_arch.crossnet
    new_crossnet = _LowRankCrossNet(crossnet)
    dlrm.inter_arch.crossnet = new_crossnet
    del crossnet

class SparseArchCatDense(SparseArch):
    def forward(
        self,
        embedded_dense_features,
        sparse_features,
    ) -> torch.Tensor:
        """
        Args:
            embedded_dense_features: the output of DenseArch.
            sparse_features: the indices/offsets for F embeddingbags in embedding_bag_collection

        Returns:
            torch.Tensor: tensor of shape B X (F + 1) X D.
        """
        (B, _) = embedded_dense_features.shape
        embedding_bag_collection = self.embedding_bag_collection
        indices = tuple([sf["values"] for _, sf in sparse_features.items()])
        offsets = tuple([sf["offsets"] for _, sf in sparse_features.items()])
        embedded_sparse_features: List[torch.Tensor] = []
        for i, embedding_bag in enumerate(
            embedding_bag_collection.embedding_bags.values()
        ):
            for feature_name in embedding_bag_collection._feature_names[i]:
                f = sparse_features[feature_name]
                res = embedding_bag(
                    f["values"],
                    f["offsets"],
                    per_sample_weights=None,
                )
                embedded_sparse_features.append(res)
        to_cat = [embedded_dense_features] + list(embedded_sparse_features)
        out = torch.cat(to_cat, dim=1)
        return out
    

class InteractionDCNArchWithoutCat(InteractionDCNArch):
    def forward(self, concat_dense_sparse: torch.Tensor) -> torch.Tensor:
        """
        Args:
            concat_dense_sparse (torch.Tensor): an input tensor of size B X (F*D + D).

        Returns:
            torch.Tensor: an output tensor of size B X (F*D + D).
        """

        # size B X (F * D + D)
        return self.crossnet(concat_dense_sparse)


class IPEX_DLRM_DCN(DLRM_DCN):
    """
    Recsys model with DCN modified from the original model from "Deep Learning Recommendation
    Model for Personalization and Recommendation Systems"
    (https://arxiv.org/abs/1906.00091). Similar to DLRM module but has
    DeepCrossNet https://arxiv.org/pdf/2008.13535.pdf as the interaction layer.

    The module assumes all sparse features have the same embedding dimension
    (i.e. each EmbeddingBagConfig uses the same embedding_dim).

    The following notation is used throughout the documentation for the models:

    * F: number of sparse features
    * D: embedding_dimension of sparse features
    * B: batch size
    * num_features: number of dense features

    Args:
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define `SparseArch`.
        dense_in_features (int): the dimensionality of the dense input features.
        dense_arch_layer_sizes (List[int]): the layer sizes for the `DenseArch`.
        over_arch_layer_sizes (List[int]): the layer sizes for the `OverArch`.
            The output dimension of the `InteractionArch` should not be manually
            specified here.
        dcn_num_layers (int): the number of DCN layers in the interaction.
        dcn_low_rank_dim (int): the dimensionality of low rank approximation
            used in the dcn layers.
        dense_device (Optional[torch.device]): default compute device.
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        dcn_num_layers: int,
        dcn_low_rank_dim: int,
        dense_device: Optional[torch.device] = None,
    ) -> None:
        # initialize DLRM
        # sparse arch and dense arch are initialized via DLRM
        super().__init__(
            embedding_bag_collection,
            dense_in_features,
            dense_arch_layer_sizes,
            over_arch_layer_sizes,
            dcn_num_layers,
            dcn_low_rank_dim,
            dense_device,
        )

        num_sparse_features: int = len(self.sparse_arch.sparse_feature_names)

        embedding_bag_collection = self.sparse_arch.embedding_bag_collection

        self.sparse_arch = SparseArchCatDense(embedding_bag_collection)

        crossnet = self.inter_arch.crossnet
        self.inter_arch = InteractionDCNArchWithoutCat(
            num_sparse_features=num_sparse_features,
            crossnet=crossnet,
        )

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features,
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): the dense features.
            sparse_features (KeyedJaggedTensor): the sparse features.

        Returns:
            torch.Tensor: logits.
        """
        embedded_dense = self.dense_arch(dense_features)
        concat_sparse_dense = self.sparse_arch(embedded_dense, sparse_features)
        concatenated_dense = self.inter_arch(concat_sparse_dense)
        logits = self.over_arch(concatenated_dense)
        return logits
