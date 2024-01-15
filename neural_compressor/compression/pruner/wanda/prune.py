# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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
from .wrapper import WrappedGPT


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
            model(batch[0].to(next(model.parameters()).device))
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
    cur_sparsity = torch.sum(W_mask == 1.0).data.item() / W_mask.numel()
    return W_mask, cur_sparsity


def move_inps_to_device(inps, others, device):
    for idx in range(len(inps)):
        for i in range(len(inps[idx])):
            inps[idx][i] = inps[idx][i].to(device)
        for arg in others[idx]:
            other = others[idx][arg]
            if isinstance(other, torch.Tensor):
                others[idx][arg] = other.to(device)


@torch.no_grad()
def prune_wanda(
    model,
    dataloader,
    sparsity_ratio,
    prune_n=0,
    prune_m=0,
    nsamples=128,
    use_variant=False,
    device=None,
    low_mem_usage=None,
    dsnot=False,
):
    """Prune the model using wanda
    Sij = |Wij| · ||Xj||2.

    See the original paper: https://arxiv.org/pdf/2306.11695.pdf
    """
    # When the model is too large, we can use low memory usage mode to reduce the GPU memory consumption.
    if low_mem_usage is None:
        low_mem_usage = False
        if "cuda" in str(device):  # pragma: no cover
            current_gpu_index = str(device)
            total_memory = torch.cuda.get_device_properties(current_gpu_index).total_memory
            used_memory = torch.cuda.memory_allocated(current_gpu_index)
            free_space = total_memory - used_memory
            total_params = sum(p.numel() for p in model.parameters())
            n = 2 if "16" in str(model.dtype) else 4
            need_space = total_params * n
            if free_space < need_space:
                logger.info("Low memory usage mode is enabled.")
                low_mem_usage = True

    if device is not None and not low_mem_usage:  # pragma: no cover
        model.to(device)
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # get inputs
    # inps, others = prepare_calibration_input(model, dataloader, device)
    inps, others = get_hidden_states(model, dataloader)
    if low_mem_usage:  # pragma: no cover
        move_inps_to_device(inps, others, device)
    nsamples = min(nsamples, len(inps))

    # get the module list of the model, blockwise
    layers = get_module_list(model)
    for i in range(len(layers)):
        outs = []
        layer = layers[i]
        if low_mem_usage:  # pragma: no cover
            layer.to(device)
        subset = find_layers(layer)
        if len(subset) == 0:
            logger.info(f"skip layer {i}, no op found")
            continue
        logger.info(f"prune layer {i}, subset list:")
        logger.info(subset)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        # register hook and forward to gather the activations ||Xj||2
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(nsamples):
            with torch.no_grad():
                out = layer(*inps[j], **others[j])
                if isinstance(out, (list, tuple)):
                    out = out[0]
                outs.append(out)
        for h in handles:
            h.remove()

        # start pruning
        for name in subset:
            logger.info(f"pruning layer {i} name {name}")
            # Sij = |Wij| · ||Xj||2
            W = subset[name].weight.data
            if isinstance(subset[name], transformers.Conv1D):  # pragma: no cover
                W = W.t()
            W_metric = torch.abs(W) * torch.sqrt(
                # wrapped_layers[name].scaler_row.reshape((1, -1))
                wrapped_layers[name].scaler_row.unsqueeze(0)
            )

            W_mask = torch.zeros_like(W_metric) == 1  ## initialize a mask to be all False
            if prune_n != 0:  # pragma: no cover
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
                    while (abs(cur_sparsity - sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    logger.info(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:  # pragma: no cover
                    # unstructured pruning
                    indices = sort_res[1][:, : int(W_metric.shape[1] * sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            if dsnot:
                from ..dsnot import DSnoT

                logger.info(f"Using dsnot pruning for {name}")
                W_mask = DSnoT(W_metric, sparsity_ratio, wrapped_layers[name])
            if isinstance(subset[name], transformers.Conv1D):  # pragma: no cover
                subset[name].weight.data[W_mask.T] = 0  ## set weights to zero
            else:
                subset[name].weight.data[W_mask] = 0  ## set weights to zero
            ratio = get_tensor_sparsity_ratio(subset[name].weight.data)
            logger.info(f"finish pruning layer {i} name {name}, sparsity ratio: {ratio}")

        for j in range(nsamples):
            with torch.no_grad():
                out = layer(*inps[j], **others[j])
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if "hidden_states" in others[j]:
                    outs[j] = []
                    others[j]["hidden_states"] = out
                else:
                    if len(inps[j]) > 1:
                        outs[j] = inps[j]
                        outs[j][0] = out
                    else:
                        outs[j] = [out]
        inps, outs = outs, inps
        if low_mem_usage:  # pragma: no cover
            layer.to(torch.device("cpu"))

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
