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

from neural_compressor.utils.utility import LazyImport

torch = LazyImport("torch")
from .wanda.wrapper import WrappedGPT


def return_reorder_indice(input_tensor):
    """
    For instance:
    [[1., -2., 3.],
    [-2, 2., -4],
    [5., 6., -7],
    [-6, -7, -4]]
    return indices of
    [[-2.,  3.,  1.],
    [-2., -4.,  2.],
    [-7.,  6.,  5.],
    [-6., -7., -4.]]
    Description: The relative order in the positive number remains unchanged,
    and the relative order in the negative number is flipped.
    """
    positive_tensor = input_tensor.clone()
    negative_tensor = input_tensor.clone()

    positive_mask = positive_tensor > 0
    negative_mask = negative_tensor < 0

    positive_indices = (
        torch.arange(0, input_tensor.shape[1], device=input_tensor.device)
        .to(torch.float64)
        .repeat(input_tensor.shape[0], 1)
    )
    negative_indices = (
        torch.arange(0, input_tensor.shape[1], device=input_tensor.device)
        .to(torch.float64)
        .repeat(input_tensor.shape[0], 1)
    )

    positive_indices[~positive_mask] = float("inf")
    negative_indices[~negative_mask] = float("inf")

    positive_value, _ = torch.sort(positive_indices, dim=1)
    negative_value, _ = torch.sort(negative_indices, dim=1)

    positive_value = torch.flip(positive_value, dims=[1])

    negative_value[negative_value == float("inf")] = 0
    positive_value[positive_value == float("inf")] = 0

    reorder_indice = (positive_value + negative_value).to(torch.int64)

    return reorder_indice


def DSnoT(
    initial_metric,
    sparsity_ratio,
    wrapped_layer,
    prune_n=0,
    prune_m=0,
    pow_of_var_regrowing=1.0,
    max_cycle_time=100,
    without_same_sign=True,
    update_threshold=0.1,
):
    """
    Args:
        pow_of_var_regrowing: The power of variance, float type, default=1.
        max_cycle_time: Max cycle time.
        without_same_sign, without same sign, bool, default True.
        update_threshold: update threshold, float, default 0.1.

    See the original paper: https://arxiv.org/pdf/2310.08915.pdf
    """
    assert isinstance(wrapped_layer, WrappedGPT)
    W = wrapped_layer.layer.weight.data
    if isinstance(wrapped_layer, transformers.Conv1D):
        W = W.t()
    DSnoT_metric = W * wrapped_layer.sum_metric_row.reshape((1, -1))
    weight_mask = torch.zeros_like(initial_metric) == 1

    if prune_n != 0:  # pragma: no cover
        initial_prune_indices = torch.zeros(
            (initial_metric.shape[0], 0),
            dtype=torch.int64,
            device=initial_metric.device,
        )
        initial_res_indices = torch.zeros(
            (initial_metric.shape[0], 0),
            dtype=torch.int64,
            device=initial_metric.device,
        )
        for ii in range(initial_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = initial_metric[:, ii : (ii + prune_m)].float()
                _, tmp_all_indices = torch.sort(tmp, dim=1)
                tmp_all_indices += ii
                res_prune_n = prune_m - prune_n
                tmp_indices, tmp_res_indices = torch.split(
                    tmp_all_indices,
                    split_size_or_sections=[prune_n, res_prune_n],
                    dim=1,
                )

                initial_prune_indices = torch.cat((initial_prune_indices, tmp_indices), dim=1)
                initial_res_indices = torch.cat((initial_res_indices, tmp_res_indices), dim=1)
                weight_mask.scatter_(1, tmp_indices, True)

        metric_for_regrowing = DSnoT_metric.clone()

        metric_for_regrowing.scatter_(1, initial_res_indices, 0)

        reconstruction_error = torch.sum(metric_for_regrowing, dim=1, keepdim=True)
        initialize_error_sign = torch.sign(reconstruction_error)

        if pow_of_var_regrowing:
            metric_for_regrowing /= torch.pow(
                wrapped_layer.var.reshape((1, -1)),
                pow_of_var_regrowing,
            )

        _, regrowing_indices_block = torch.sort(metric_for_regrowing, dim=1, stable=True)

        indice_indice_list_for_regrowing = torch.zeros(
            (reconstruction_error.shape[0], 2),
            device=reconstruction_error.device,
            dtype=torch.long,
        )
        last_one = regrowing_indices_block.shape[-1] - 1
        indice_indice_list_for_regrowing[:, 1] = last_one
        update_num_for_regrowing = torch.ones(
            (reconstruction_error.shape[0], 2),
            device=reconstruction_error.device,
            dtype=torch.long,
        )
        update_num_for_regrowing[:, 1] = -1

        initial_metric.scatter_(1, initial_prune_indices, float("inf"))
        W_metric_max_value = torch.max(initial_metric, dim=1, keepdim=True)[0] + 1

        cycle_time = 1
        update_mask = torch.ones_like(reconstruction_error, dtype=torch.bool)
        while not (torch.all(update_mask == 0.0) or cycle_time > max_cycle_time):
            cycle_time += 1

            # regrowing
            indice_of_indice_indice_list_for_regrowing = (reconstruction_error > 0).int().to(torch.int64)
            indice_indice_for_regrowing = torch.gather(
                indice_indice_list_for_regrowing,
                1,
                indice_of_indice_indice_list_for_regrowing,
            )
            regrowing_indice = torch.gather(
                regrowing_indices_block,
                1,
                indice_indice_for_regrowing.to(torch.int64),
            )
            regrowing_metric = DSnoT_metric.gather(1, regrowing_indice.to(torch.int64))
            recover_block_start_indice = regrowing_indice - regrowing_indice % prune_m
            recover_block_indices = (
                torch.arange(0, prune_m, device=recover_block_start_indice.device).repeat(
                    recover_block_start_indice.shape[1], 1
                )
                + recover_block_start_indice
            )
            pruning_block = torch.gather(initial_metric, 1, recover_block_indices.to(torch.int64))
            pruning_wanda_metric, pruning_indice = torch.topk(pruning_block, 1, dim=1, largest=False)
            pruning_indice += recover_block_start_indice
            pruning_metric = DSnoT_metric.gather(1, pruning_indice.to(torch.int64))
            reconstruction_error_after = reconstruction_error + pruning_metric - regrowing_metric
            update_mask = (
                update_mask
                & (initialize_error_sign == torch.sign(reconstruction_error_after))
                & (abs(reconstruction_error) > update_threshold)
            )
            initial_metric.scatter_(1, pruning_indice, W_metric_max_value)
            weight_mask.scatter_(1, pruning_indice, update_mask)
            weight_mask.scatter_(1, regrowing_indice, ~update_mask)
            reconstruction_error += torch.where(
                update_mask,
                pruning_metric,
                torch.zeros_like(pruning_metric),
            )
            reconstruction_error -= torch.where(
                update_mask,
                regrowing_metric,
                torch.zeros_like(regrowing_metric),
            )
            indice_indice_list_for_regrowing.scatter_(
                1,
                indice_of_indice_indice_list_for_regrowing,
                indice_indice_for_regrowing
                + update_num_for_regrowing.gather(1, indice_of_indice_indice_list_for_regrowing),
            )
    else:
        _, sorted_initial_indice = torch.sort(initial_metric, dim=-1, stable=True)
        sparsity_num = int(initial_metric.shape[-1] * sparsity_ratio)
        res_sparsity_num = sorted_initial_indice.shape[-1] - sparsity_num

        initial_prune_indices, initial_res_indices = torch.split(
            sorted_initial_indice,
            split_size_or_sections=[sparsity_num, res_sparsity_num],
            dim=1,
        )

        weight_mask.scatter_(1, initial_prune_indices, True)
        metric_for_regrowing = DSnoT_metric.clone()
        wanda_metric = torch.abs(W) * torch.sqrt(wrapped_layer.scaler_row.reshape((1, -1)))
        metric_for_regrowing.scatter_(1, initial_res_indices, 0)
        reconstruction_error = torch.sum(metric_for_regrowing, dim=1, keepdim=True)
        initialize_error_sign = torch.sign(reconstruction_error)
        if pow_of_var_regrowing:
            metric_for_regrowing /= torch.pow(
                wrapped_layer.var.reshape((1, -1)),
                pow_of_var_regrowing,
            )

        _, regrowing_indices_block = torch.sort(metric_for_regrowing, dim=1, stable=True)
        wanda_metric.scatter_(1, initial_prune_indices, float("inf"))
        wanda_res_indices, _ = torch.split(
            torch.sort(wanda_metric, dim=1, stable=True)[1],
            split_size_or_sections=[res_sparsity_num, sparsity_num],
            dim=1,
        )
        reorder_indice_of_pruning_indice = return_reorder_indice(torch.gather(DSnoT_metric, 1, wanda_res_indices))
        pruning_indices_block = torch.gather(wanda_res_indices, 1, reorder_indice_of_pruning_indice)

        indice_indice_list_for_regrowing = torch.zeros(
            (reconstruction_error.shape[0], 2),
            device=reconstruction_error.device,
            dtype=torch.long,
        )
        last_one = regrowing_indices_block.shape[-1] - 1
        indice_indice_list_for_regrowing[:, 1] = last_one

        update_num_for_regrowing = torch.ones(
            (reconstruction_error.shape[0], 2),
            device=reconstruction_error.device,
            dtype=torch.long,
        )
        update_num_for_regrowing[:, 1] = -1

        indice_indice_list_for_pruning = torch.zeros(
            (reconstruction_error.shape[0], 2),
            device=reconstruction_error.device,
            dtype=torch.long,
        )
        last_one = pruning_indices_block.shape[-1] - 1
        indice_indice_list_for_pruning[:, 1] = last_one

        update_num_for_pruning = torch.ones(
            (reconstruction_error.shape[0], 2),
            device=reconstruction_error.device,
            dtype=torch.long,
        )
        update_num_for_pruning[:, 1] = -1

        update_mask = torch.ones_like(reconstruction_error, dtype=torch.bool)
        cycle_time = 0
        while not (torch.all(update_mask == 0.0) or cycle_time >= max_cycle_time):
            cycle_time += 1

            # regrowing
            indice_of_indice_indice_list_for_regrowing = (reconstruction_error > 0).int().to(torch.int64)

            indice_indice_for_regrowing = torch.gather(
                indice_indice_list_for_regrowing,
                1,
                indice_of_indice_indice_list_for_regrowing,
            )

            regrowing_indice = torch.gather(
                regrowing_indices_block,
                1,
                indice_indice_for_regrowing.to(torch.int64),
            )

            regrowing_metric = DSnoT_metric.gather(1, regrowing_indice.to(torch.int64))

            indice_indice_list_for_regrowing.scatter_(
                1,
                indice_of_indice_indice_list_for_regrowing,
                indice_indice_for_regrowing
                + update_num_for_regrowing.gather(1, indice_of_indice_indice_list_for_regrowing),
            )

            # pruning
            indice_of_indice_indice_list_for_pruning = (reconstruction_error < 0).int().to(torch.int64)

            indice_indice_for_pruning = torch.gather(
                indice_indice_list_for_pruning,
                1,
                indice_of_indice_indice_list_for_pruning,
            )

            pruning_indice = torch.gather(
                pruning_indices_block,
                1,
                indice_indice_for_pruning.to(torch.int64),
            )

            pruning_metric = DSnoT_metric.gather(1, pruning_indice.to(torch.int64))

            indice_indice_list_for_pruning.scatter_(
                1,
                indice_of_indice_indice_list_for_pruning,
                indice_indice_for_pruning + update_num_for_pruning.gather(1, indice_of_indice_indice_list_for_pruning),
            )

            # change mask
            reconstruction_error_after = reconstruction_error + pruning_metric - regrowing_metric

            if without_same_sign:
                update_mask = update_mask & (abs(reconstruction_error) > update_threshold)
            else:
                update_mask = (
                    update_mask
                    & (abs(reconstruction_error) > update_threshold)
                    & (initialize_error_sign == torch.sign(reconstruction_error_after))
                )

            weight_mask.scatter_(1, pruning_indice, update_mask)
            weight_mask.scatter_(1, regrowing_indice, ~update_mask)

            reconstruction_error += torch.where(
                update_mask,
                pruning_metric,
                torch.zeros_like(pruning_metric),
            )
            reconstruction_error -= torch.where(
                update_mask,
                regrowing_metric,
                torch.zeros_like(regrowing_metric),
            )

    return weight_mask
