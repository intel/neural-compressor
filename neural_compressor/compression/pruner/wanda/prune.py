# !/usr/bin/env python
# -*- coding: utf-8 -*-
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
import transformers

from neural_compressor.adaptor.torch_utils.util import get_hidden_states

from .utils import find_layers, get_module_list, get_tensor_sparsity_ratio, logger, nn, torch


# Define WrappedGPT class
class WrappedGPT:
    """This class wraps a GPT layer for specific operations."""

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        import torch

        if isinstance(self.layer, (nn.Conv1d, nn.Linear, transformers.Conv1D)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples


@torch.no_grad()
def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_module_list(model)

    if hasattr(model, "hf_device_map") and "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    inps = []
    others = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *inp, **kwargs):
            tmp_other = {}
            for arg in kwargs:
                if isinstance(kwargs[arg], torch.Tensor) or arg == "alibi":
                    tmp_other[arg] = kwargs[arg]
            others.append(tmp_other)
            inps.append(list(inp))
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    model.config.use_cache = use_cache

    return inps, others


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1)
    W_mask = W_metric <= thres
    cur_sparsity = (W_mask is True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


@torch.no_grad()
def prune_wanda(
    model,
    dataloader,
    sparsity_ratio,
    prune_n=0,
    prune_m=0,
    nsamples=128,
    use_variant=False,
    device=torch.device("cpu"),
):
    """Prune the model using wanda
    Sij = |Wij| · ||Xj||2."""
    model.to(device)
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # get inputs
    # inps, others = prepare_calibration_input(model, dataloader, device)
    inps, others = get_hidden_states(model, dataloader)
    nsamples = min(nsamples, len(inps))

    # get the module list of the model, blockwise
    layers = get_module_list(model)
    for i in range(len(layers)):
        outs = []
        layer = layers[i]
        subset = find_layers(layer)
        logger.info(f"prune layer {i}, subset list:")
        logger.info(subset)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        # register hook and foward to gather the activations ||Xj||2
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(nsamples):
            with torch.no_grad():
                outs.append(layer(*inps[j], **others[j])[0])
        for h in handles:
            h.remove()

        # start pruning
        for name in subset:
            logger.info(f"pruning layer {i} name {name}")
            # Sij = |Wij| · ||Xj||2
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                # wrapped_layers[name].scaler_row.reshape((1, -1))
                wrapped_layers[name].scaler_row.unsqueeze(0)
            )

            W_mask = torch.zeros_like(W_metric) == 1  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0.0, 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - sparsity_ratio) > 0.001) and (
                        alpha_hist[1] - alpha_hist[0] >= 0.001
                    ):
                        if cur_sparsity > sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    logger.info(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:, : int(W_metric.shape[1] * sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero
            ratio = get_tensor_sparsity_ratio(subset[name].weight.data)
            logger.info(f"finish pruning layer {i} name {name}, sparsity ratio: {ratio}")

        for j in range(nsamples):
            with torch.no_grad():
                out = layer(*inps[j], **others[j])[0]
                outs[j] = [out]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
