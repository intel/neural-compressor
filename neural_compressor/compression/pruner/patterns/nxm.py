"""NxM patterns."""

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
import numpy as np

from ..utils import logger, nn, safe_get_data, safe_get_grad, safe_get_shape, tf, torch
from .base import KerasBasePattern, ProgressivePatternUtils, PytorchBasePattern, SparsityInfo, register_pattern


@register_pattern("ptNxM")
class PytorchPatternNxM(PytorchBasePattern):
    """Pruning Pattern.

    A Pattern class derived from BasePattern. In this pattern, the weights in a NxM block will be pruned or kept
    during one pruning step.

    Args:
        config: A config dict object that contains the pattern information.

    Attributes:
            block_size: A list of two integers representing the height and width of the block.
            Please note that the vertical direction of a Linear layer's weight refers to the output channel.
                because PyTorch's tensor matmul has a hidden transpose operation.
    """

    def __init__(self, config, modules):
        """Initialize the basic pruning unit of NXM pattern."""
        super().__init__(config, modules)
        pattern = self.pattern.split("_")[-1]
        self.N = pattern.split("x")[0]
        self.M = pattern.split("x")[1]
        if self.N == "channel":  # channel-wise pruning mode
            self.block_size = ["channel", int(self.M)]
        elif self.M == "channel":  # channel-wise pruning mode
            self.block_size = [int(self.N), "channel"]
        else:
            self.block_size = [int(pattern.split("x")[0]), int(pattern.split("x")[1])]
        self.total_params_cnt = -1

        self.block_size = self.get_block_size_dict()
        self.check_layer_validity()

    def get_block_size_dict(self):
        """Calculate the zero elements' ration in pre_masks.

        Args:
            data: Dict{"layer_name": Tensor} that stores weights or scores.

        Returns:
            A dict. Dict{"layer_name": [block_size_1, block_size_2]} containing block shapes of each layer.
                In channel-wise pruning different layers can have different pruning patterns.
        """
        datas = self.modules
        block_sizes_dict = {}
        for key in datas.keys():
            block_sizes_dict[key] = self.block_size
            if not (self.N == "channel" or self.M == "channel"):
                continue
            if isinstance(datas[key], torch.nn.Module):
                param = datas[key].weight
                shape = safe_get_shape(param)
                # shape = datas[key].weight.shape
            else:
                shape = datas[key].shape
            if self.N == "channel":  # support "channelxM" format
                block_sizes_dict[key] = [shape[0], self.block_size[1]]
            if self.M == "channel":
                block_sizes_dict[key] = [self.block_size[0], shape[1]]

        return block_sizes_dict

    def check_layer_validity(self):
        """Check if a layer is valid for this block_size."""
        block_sizes = self.block_size
        datas = self.modules
        for key in datas.keys():
            param = datas[key].weight
            data = safe_get_data(param)
            data = self._reshape_orig_to_2dims(data)
            shape = data.shape
            block_size = block_sizes[key]
            if shape[0] % block_size[0] != 0 or shape[1] % block_size[1] != 0:  # only consider input channel
                self.invalid_layers.append(key)
                logger.warning(f"{key} shape {data.shape} cannot be divided by {self.pattern}")

    def get_reduced_masks_from_data(self, data, key):
        """Obtain the unpruned weights and reshape according to the block_size.

        Args:
            data: Input.
            key: The layer name.

        Returns:
            The unpruned weights.
        """
        assert key not in self.invalid_layers
        block_size = self.block_size[key]
        data = self._reshape_orig_to_2dims(data)
        shape = data.shape
        new_shape = [shape[0] // block_size[0], block_size[0], shape[1] // block_size[1], block_size[1]]
        data = data.reshape(new_shape)
        data = data.sum(-1).sum(1)
        reduced_mask = data != 0
        return reduced_mask

    def get_sparsity_ratio(self, pre_masks, return_dict=False):
        """Please note that the zero cnt and total cnt are all block_wise for supporting channel-wise pruning.

        Args:
            pre_masks: Dict{"layer_name": Tensor} representing the masks generated after the last pruning step.
            return_dict: A bool determining whether to return more information like zero_cnt and total_cnt.

        Returns:
            A float representing the zero elements' ratio in pre_masks.
        """
        zero_cnt = 0
        total_cnt = 0
        for key in pre_masks.keys():
            if key in self.invalid_layers:
                continue
            reduced_mask = (
                pre_masks[key].float() if self.block else self.get_reduced_masks_from_data(pre_masks[key].float(), key)
            )
            zero_cnt += int(torch.sum(reduced_mask == 0.0).data.item())
            total_cnt += int(reduced_mask.numel())
        if total_cnt == 0:
            sparsity_ratio = 0.0
        else:
            sparsity_ratio = float(zero_cnt) / total_cnt
        if return_dict:
            return {"sparsity_ratio": sparsity_ratio, "zero_cnt": zero_cnt, "total_cnt": total_cnt}
        else:
            return sparsity_ratio

    def _reshape_orig_to_2dims(self, data):
        """Process layers that are not two-dimensional(e.g conv layer).

        Args:
            data: Input.

        Return:
            Reshaped data.
        """
        # TODO: need to verify whether it's ok for transposed conv
        from ..utils import FLATTEN_DIM2

        if len(data.shape) == 2:
            return data
        elif len(data.shape) == 4:
            if isinstance(data, np.ndarray):
                data = np.transpose(data, (0, 2, 3, 1))
            else:
                data = data.permute(0, 2, 3, 1)  # cout,k,k,cin
            data = data.reshape(data.shape[0], -1)
        elif len(data.shape) == 3:
            if isinstance(data, np.ndarray):
                data = np.transpose(data, (0, 2, 1))
            else:
                data = data.permute(0, 2, 1)  # cout,k,cin
            data = data.reshape(data.shape[0], -1)
        # TODO(Yi) support handle 1-dim (flatten param from DeepSpeed or FSDP)
        elif len(data.shape) == 1:  # pragma: no cover
            data = data.reshape(-1, FLATTEN_DIM2)
        else:
            raise NotImplementedError(
                f"Currently only support reshape data with 1,3,4-dims, but got shape {data.shape}"
            )
        return data

    def _reshape_2dims_to_orig(self, data, orig_shape):
        """Recover layers that are not two-dimensional(e.g conv layer).

        Args:
            data: Input.
            orig_shape: Target shape.

        Returns:
            Reshaped data.
        """
        if len(orig_shape) == 2:
            return data
        elif len(orig_shape) == 4:
            data = data.reshape(orig_shape[0], orig_shape[2], orig_shape[3], orig_shape[1])
            if isinstance(data, np.ndarray):  # pragma: no cover
                data = np.transpose(data, (0, 3, 1, 2))
            else:
                data = data.permute(0, 3, 1, 2)
        elif len(orig_shape) == 3:
            data = data.reshape(orig_shape[0], orig_shape[2], orig_shape[1])
            if isinstance(data, np.ndarray):  # pragma: no cover
                data = np.transpose(data, (0, 2, 1))
            else:
                data = data.permute(0, 2, 1)
        elif len(orig_shape) == 1:
            data = data.reshape(-1)
        else:
            raise NotImplementedError(
                f"Currently only support reshape data with 1,3,4-dims, but got shape {data.shape}"
            )
        return data

    def reshape_orig_to_pattern(self, data, key):
        """Reshape the data(s1,s2) to [s1/N,N,s2/M,M].

        Args:
            data: The input.
            key: The layer name.

        Returns:
            Reshaped input tensor.
        """
        block_size = self.block_size[key]
        data = self._reshape_orig_to_2dims(data)
        shape = data.shape
        new_shape = [shape[0] // block_size[0], block_size[0], shape[1] // block_size[1], block_size[1]]
        data = data.reshape(new_shape)
        return data

    def reshape_reduced_to_orig(self, data, key, orig_shape):
        """Reshape the data [s1/N,s2/M] to [s1,s2], also permute dims for conv layer.

        Args:
            data: Input.
            key: The layer name.
            orig_shape: The original shape of the layer.

        Returns:
            Data of its original shape.
        """
        if key in self.invalid_layers:
            return data
        if self.keep_mask_layers.get(key, False):
            return data
        block_size = self.block_size[key]
        data = data.repeat_interleave(block_size[0], dim=0).repeat_interleave(block_size[1], dim=-1)
        data = self._reshape_2dims_to_orig(data, orig_shape)
        return data

    def reduce_score(self, score, key, force=False):
        """Recalculate the pruning score after reducing the data.

        Args:
            scores: A Tensor that stores the pruning score of weights.

        Returns:
            The reduced pruning score.
        """
        if not force:
            if key in self.invalid_layers:
                return score
            if self.keep_mask_layers.get(key, False):
                return score
        self.keep_mask_layers[key] = False
        new_score = self.reshape_orig_to_pattern(score, key)
        # sum or mean is quite different for per channel pruning
        new_score = self.reduce_tensor(self.reduce_tensor(new_score, dim=-1), dim=1)
        return new_score

    def reduce_scores(self, scores):  # pragma: no cover
        """Recalculate the pruning scores after reducing the data.

        Args:
            scores: A dict{"layer_name": Tensor} that stores the pruning scores of weights.

        Returns:
            The reduced pruning scores.
        """
        new_scores = {}
        for key in scores.keys():
            if key in self.invalid_layers:  # pragma: no cover
                continue
            if self.keep_mask_layers.get(key, False):  # pragma: no cover
                continue
            self.keep_mask_layers[key] = False
            current_score = scores[key]
            current_score = self.reshape_orig_to_pattern(current_score, key)
            # sum or mean is quite different for per channel pruning
            if current_score.dtype == torch.bool:
                current_score = current_score.float()
            current_score_sum = self.reduce_tensor(self.reduce_tensor(current_score, dim=-1), dim=1)
            new_scores[key] = current_score_sum
        return new_scores

    def get_mask_per_threshold(self, score, threshold, block_size):
        """Get the mask per threshold."""
        zero = torch.tensor([False]).to(score.device)
        one = torch.tensor([True]).to(score.device)
        mask = torch.where(score <= threshold, zero, one)
        if not self.block:
            mask = mask.repeat_interleave(block_size[0], dim=0).repeat_interleave(block_size[1], dim=-1)
        else:
            mask = mask.float()
        return mask

    def get_masks_global(self, scores, cur_target_sparsity_ratio, pre_masks, keep_exact_sparsity_ratio=True):
        """Generate masks for layers.

        Gather all layer's scores together and calculate a common threshold.
        This threshold will be applied to all layers.

        Args:
            scores: A dict{"layer_name": Tensor} that stores the pruning scores of weights.
            cur_target_sparsity_ratio: A float representing the model's sparsity after pruning.
            pre_masks: A dict{"layer_name": Tensor} that stores the masks generated at the last pruning step.
            max_sparsity_ratio_per_op: A float representing the maximum sparsity that one layer can reach.
            keep_pre_masks: A bool representing if the masks should remain unchanged.

        Returns:
            A dict with the identical size as pre_masks and its 0/1 values are updated.
                1 means unpruned and 0 means pruned.
        """
        # keep the masks if the layer exceed max sparsity ratio
        masks = pre_masks
        k_blockwise = self.update_residual_cnt(masks, cur_target_sparsity_ratio)
        if k_blockwise <= 0:
            return masks
        new_scores = scores  # if self.block else self.reduce_scores(scores)
        not_exceed_layers = []
        residual_k = k_blockwise
        if self.min_sparsity_ratio_per_op > 0:
            sparsity_infos_perlayer, _ = self.get_sparsity_ratio_each_layer(masks)

        while True:
            new_not_exceed_layers = [key for key in new_scores.keys() if not self.keep_mask_layers.get(key, False)]
            if not_exceed_layers == new_not_exceed_layers or len(new_not_exceed_layers) == 0:
                break
            not_exceed_layers = new_not_exceed_layers
            global_scores = torch.cat([torch.flatten(new_scores[key]) for key in not_exceed_layers])
            if residual_k < 1:  # pragma: no cover
                break
            threshold, _ = torch.kthvalue(global_scores, residual_k)

            for key in not_exceed_layers:
                block_size = self.block_size[key]
                score = new_scores[key]
                mask = self.get_mask_per_threshold(score, threshold, block_size)
                info = self.get_sparsity_ratio({key: mask}, return_dict=True)
                zero_cnt = info["zero_cnt"]
                total_cnt = info["total_cnt"]
                current_sparsity_ratio = float(zero_cnt) / total_cnt
                key_new_sparsity = SparsityInfo(zero_cnt, total_cnt, current_sparsity_ratio)
                need_adjust, adjust_ratio = self.adjust_ratio(
                    masks,
                    key,
                    key_new_sparsity,
                    self.max_sparsity_ratio_per_op,
                    self.min_sparsity_ratio_per_op,
                    self.target_sparsity_ratio,
                )
                if need_adjust:
                    # uptade status
                    self.keep_mask_layers[key] = True
                    masks[key] = self.get_single_mask_per_target_ratio(new_scores[key], adjust_ratio)
                    if not self.block:
                        masks[key] = masks[key].repeat_interleave(block_size[0], 0).repeat_interleave(block_size[1], -1)
                    if keep_exact_sparsity_ratio:
                        zero_cnt = self.get_sparsity_ratio({key: masks[key]}, return_dict=True)["zero_cnt"]
                        residual_k -= zero_cnt
                else:
                    masks[key] = mask

                if not self.block:
                    masks[key] = masks[key].bool()
            if not keep_exact_sparsity_ratio:  # pragma: no cover
                break

        for key in masks.keys():
            if key in self.invalid_layers:
                continue
            param = self.modules[key].weight
            orig_shape = safe_get_shape(param)
            # orig_shape = self.modules[key].weight.shape
            # if len(orig_shape) == 4 or len(orig_shape) == 3 :  # need to permute
            mask = masks[key]
            # orig_shape = scores[key].shape
            mask = self._reshape_2dims_to_orig(mask, orig_shape)
            masks[key] = mask
            layer_ratio = torch.sum(masks[key] == 0.0).data.item() / masks[key].numel()
            logger.info(f"{key} sparsity is {layer_ratio}")
        return masks

    def get_pattern_lock_masks(self, modules):
        """Obtain masks from original weight map by masking the zero-valued weights.

        Args:
            modules: A dict{"layer_name": Tensor} that stores weights.

        Returns:
            A dict with the identical size as modules, containing pattern lock masks.
        """
        pattern_lock_masks = {}
        for key in modules.keys():
            param = modules[key].weight
            data = safe_get_data(param)
            ori_shape = safe_get_shape(param)
            # ori_shape = weight.shape
            if key in self.invalid_layers:
                mask = torch.ones(ori_shape, device=param.device)
                pattern_lock_masks[key] = mask
                continue
            reduced_mask = self.get_reduced_masks_from_data(data, key)
            mask = self.reshape_reduced_to_orig(reduced_mask, key, ori_shape)
            pattern_lock_masks[key] = mask

        return pattern_lock_masks

    def register_block_masks(self):
        """Register the block mask parameters and get the mask gradients.

        Args:
            modules: A dict{"layer_name": Tensor} that stores weights.

        Returns:
            A dict containing block masks.
        """
        masks = {}
        for key in self.modules.keys():
            if key in self.invalid_layers:
                continue  # No corresponding block mask, skip.
            module = self.modules[key]
            weight = module.weight
            if type(module).__name__ not in ["Linear"]:
                logger.warning(
                    f"Currently only support Linear block mask pruning," f"{type(module).__name__} won't be pruned."
                )
                continue
            block_mask = self.get_reduced_masks_from_data(weight.detach(), key).to(dtype=weight.dtype)
            masks[key] = block_mask

        return masks

    def mask_block_weights(self, masks):
        """Achieve weight pruning by multiplying the reshaped weights and block masks."""
        for key in masks.keys():
            if key in self.invalid_layers:
                continue
            module = self.modules[key]
            block_size = self.block_size[key]
            # org_shape = module.weight.shape
            org_shape = safe_get_shape(module.weight)
            mask = (
                masks[key]
                .data.repeat_interleave(block_size[0], dim=0)
                .repeat_interleave(block_size[1], dim=-1)
                .to(module.weight.device)
            )
            reshaped_weight = self._reshape_orig_to_2dims(module.weight.data) * mask
            module.weight.data = self._reshape_2dims_to_orig(reshaped_weight, org_shape)

    def update_progressive_masks(self, pre_masks, cur_masks, scores, progressive_step, progressive_configs):
        """Generate the progressive masks.

        Args:
            pre_masks: Dict{"layer_name": Tensor} that stores the masks generated after the last pruning step.
            cur_masks: Dict{"layer_name": Tensor} that stores the current masks.
            scores: A dict{"layer_name": Tensor} that stores the pruning scores of weights.
            progressive_step: An integer representing the number of current step in progressive pruning.
            progressive_configs: A dict that stores configurations of progressive pruning.

        Returns:
            A dict{"layer_name": Tensor} that stores the masks generated in progressive pruning.
        """
        score_or_linear = progressive_configs["progressive_type"]  # "scores" or "linear"
        new_scores = {}
        for key in scores.keys():
            if scores[key].shape != pre_masks[key].shape:
                block_size = self.block_size[key]
                data = scores[key].repeat_interleave(block_size[0], dim=0).repeat_interleave(block_size[1], dim=-1)
                data = self._reshape_2dims_to_orig(data, pre_masks[key].shape)
                new_scores[key] = data
            else:
                new_scores[key] = scores[key]
        if score_or_linear == "scores":
            return ProgressivePatternUtils.update_progressive_masks_scores_order(
                pre_masks, cur_masks, new_scores, progressive_step, progressive_configs
            )
        elif score_or_linear == "linear":
            return ProgressivePatternUtils.update_progressive_masks_linear_order(
                pre_masks, cur_masks, new_scores, progressive_step, progressive_configs, self.block_size
            )
        else:
            raise NotImplementedError

    def fasterprune(self, gpt, blocksize=128, percdamp=0.01):
        import transformers

        sparsity = self.target_sparsity_ratio
        W = gpt.module.weight.data.clone()
        dev = gpt.dev
        rows = gpt.rows
        columns = gpt.columns
        H = gpt.H
        module = gpt.module
        if isinstance(module, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(module, transformers.Conv1D):
            W = W.t()
        W = W.float()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(rows, device=dev)
        damp = percdamp * torch.mean(torch.diag(H))  # λI
        diag = torch.arange(columns, device=dev)
        H[diag, diag] += damp  # H = (X*X.t() + λI)
        H = torch.linalg.cholesky(H)  # the default is lower triangle
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, columns, blocksize):
            i2 = min(i1 + blocksize, columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
            thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
            mask1 = tmp <= thresh

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if isinstance(module, transformers.Conv1D):
            W = W.t()
        param = module.weight
        param_shape = safe_get_shape(param)
        module.weight.data = W.reshape(param_shape).to(dtype=module.weight.data.dtype)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@register_pattern("kerasNxM")
class KerasPatternNxM(KerasBasePattern):
    """Pruning Pattern.

    A Pattern class derived from BasePattern. In this pattern, the weights in a NxM block will be pruned or kept
    during one pruning step.

    Args:
        config: A config dict object that contains the pattern information.

    Attributes:
            block_size: A list of two integers representing the height and width of the block.
            Please note that the vertical direction of a Linear layer's weight refers to the output channel.
                because PyTorch's tensor matmul has a hidden transpose operation.
    """

    def __init__(self, config, modules):
        """Initialize the basic pruning unit of NXM pattern."""
        super().__init__(config, modules)
        pattern = self.pattern.split("_")[-1]
        self.N = pattern.split("x")[0]
        self.M = pattern.split("x")[1]
        if self.N == "channel":  # channel-wise pruning mode
            self.block_size = ["channel", int(self.M)]
        elif self.M == "channel":  # channel-wise pruning mode
            self.block_size = [int(self.N), "channel"]
        else:
            self.block_size = [int(pattern.split("x")[0]), int(pattern.split("x")[1])]
        self.total_params_cnt = -1

        self.block_size = self.get_block_size_dict()
        self.check_layer_validity()

    def get_block_size_dict(self):
        """Calculate the zero elements' ration in pre_masks.

        Args:
            data: Dict{"layer_name": Tensor} that stores weights or scores.

        Returns:
            A dict. Dict{"layer_name": [block_size_1, block_size_2]} containing block shapes of each layer.
                In channel-wise pruning different layers can have different pruning patterns.
        """
        datas = self.modules
        block_sizes_dict = {}
        for key in datas.keys():
            block_sizes_dict[key] = self.block_size
            if not (self.N == "channel" or self.M == "channel"):
                continue
            shape = datas[key].shape
            if self.N == "channel":  # support "channelxM" format
                block_sizes_dict[key] = [shape[0], self.block_size[1]]
            if self.M == "channel":
                block_sizes_dict[key] = [self.block_size[0], shape[1]]

        return block_sizes_dict

    def check_layer_validity(self):
        """Check if a layer is valid for this block_size."""
        block_sizes = self.block_size
        datas = self.modules
        for key in datas.keys():
            data = datas[key].get_weights()[0]
            data = self._reshape_orig_to_2dims(data)
            shape = data.shape
            block_size = block_sizes[key]
            if shape[0] % block_size[0] != 0 or shape[1] % block_size[1] != 0:  ## only consider input channel
                self.invalid_layers.append(key)
                logger.warning(f"{key} shape {data.shape} cannot be divided by {self.pattern}")

    def get_reduced_masks_from_data(self, data, key):
        """Obtain the unpruned weights and reshape according to the block_size.

        Args:
            data: Input.
            key: The layer name.

        Returns:
            The unpruned weights.
        """
        assert key not in self.invalid_layers
        block_size = self.block_size[key]
        data = self._reshape_orig_to_2dims(data)
        shape = data.shape
        new_shape = [shape[0] // block_size[0], block_size[0], shape[1] // block_size[1], block_size[1]]
        data = data.reshape(new_shape)
        data = data.sum(-1).sum(1)
        reduced_mask = data != 0
        return reduced_mask

    def get_sparsity_ratio(self, pre_masks, return_dict=False):
        """Please note that the zero cnt and total cnt are all block_wise for supporting channel-wise pruning.

        Args:
            pre_masks: Dict{"layer_name": Tensor} representing the masks generated after the last pruning step.
            return_dict: A bool determining whether to return more information like zero_cnt and total_cnt.

        Returns:
            A float representing the zero elements' ratio in pre_masks.
        """
        zero_cnt = 0
        total_cnt = 0
        for key in pre_masks.keys():
            if key in self.invalid_layers:
                continue
            if not isinstance(pre_masks[key], np.ndarray):
                pre_masks[key] = pre_masks[key].numpy()
            reduced_mask = pre_masks[key] if self.block else self.get_reduced_masks_from_data(pre_masks[key], key)
            zero_cnt += int(np.sum(reduced_mask == 0.0))
            total_cnt += int(reduced_mask.size)
        if total_cnt == 0:
            sparsity_ratio = 0.0
        else:
            sparsity_ratio = float(zero_cnt) / total_cnt
        if return_dict:
            return {"sparsity_ratio": sparsity_ratio, "zero_cnt": zero_cnt, "total_cnt": total_cnt}
        else:
            return sparsity_ratio

    def _reshape_orig_to_2dims(self, data):
        """Process layers that are not two-dimensional(e.g conv layer).

        Args:
            data: Input.

        Returns:
            Reshaped data.
        """
        if len(data.shape) == 4:
            if isinstance(data, np.ndarray):
                data = np.transpose(data, (0, 2, 3, 1))
            else:
                data = data.permute(0, 2, 3, 1)
            data = data.reshape(data.shape[0], -1)
        return data

    def _reshape_2dims_to_orig(self, data, orig_shape):
        """Recover layers that are not two-dimensional(e.g conv layer).

        Args:
            data: Input.
            orig_shape: Target shape.

        Returns:
            Reshaped data.
        """
        if len(orig_shape) == 4:
            data = data.reshape(orig_shape[0], orig_shape[2], orig_shape[3], orig_shape[1])
            if isinstance(data, np.ndarray):
                data = np.transpose(data, (0, 3, 1, 2))
            else:
                data = data.permute(0, 3, 1, 2)
        return data

    def reshape_orig_to_pattern(self, data, key):
        """Reshape the data(s1,s2) to [s1/N,N,s2/M,M].

        Args:
            data: The input.
            key: The layer name.

        Returns:
            Reshaped input tensor.
        """
        block_size = self.block_size[key]
        data = self._reshape_orig_to_2dims(data)
        shape = data.shape
        new_shape = [shape[0] // block_size[0], block_size[0], shape[1] // block_size[1], block_size[1]]
        data = data.reshape(new_shape)
        return data

    def reshape_reduced_to_orig(self, data, key, orig_shape):
        """Reshape the data [s1/N,s2/M] to [s1,s2], also permute dims for conv layer.

        Args:
            data: Input.
            key: The layer name.
            orig_shape: The original shape of the layer.

        Returns:
            Data of its original shape.
        """
        if key in self.invalid_layers:
            return data
        if self.keep_mask_layers.get(key, False):
            return data
        block_size = self.block_size[key]
        data = data.repeat_interleave(block_size[0], dim=0).repeat_interleave(block_size[1], dim=-1)
        data = self._reshape_2dims_to_orig(data, orig_shape)
        return data

    def reduce_scores(self, scores):
        """Recalculate the pruning scores after reducing the data.

        Args:
            scores: A dict{"layer_name": Tensor} that stores the pruning scores of weights.

        Returns:
            The reduced pruning scores.
        """
        new_scores = {}
        for key in scores.keys():
            if key in self.invalid_layers:
                continue
            if self.keep_mask_layers.get(key, False):
                continue
            self.keep_mask_layers[key] = False
            current_score = scores[key]
            current_score = self.reshape_orig_to_pattern(current_score, key)
            current_score_sum = self.reduce_tensor(self.reduce_tensor(current_score, dim=-1), dim=1)
            new_scores[key] = current_score_sum
        return new_scores

    def get_mask_per_threshold(self, score, threshold, block_size):
        """Get the mask per threshold."""
        zero = tf.convert_to_tensor([0.0])
        one = tf.convert_to_tensor([1.0])
        mask = tf.where(score <= threshold, zero, one)
        if not self.block:
            mask = tf.repeat(mask, repeats=block_size[0], axis=0)
            mask = tf.repeat(mask, repeats=block_size[1], axis=-1)
        mask = mask.numpy()
        return mask

    def get_masks_global(self, scores, cur_target_sparsity_ratio, pre_masks, keep_exact_sparsity_ratio=True):
        """Generate masks for layers.

        Gather all layer's scores together and calculate a common threshold.
        This threshold will be applied to all layers.

        Args:
            scores: A dict{"layer_name": Tensor} that stores the pruning scores of weights.
            cur_target_sparsity_ratio: A float representing the model's sparsity after pruning.
            pre_masks: A dict{"layer_name": Tensor} that stores the masks generated at the last pruning step.
            max_sparsity_ratio_per_op: A float representing the maximum sparsity that one layer can reach.
            keep_pre_masks: A bool representing if the masks should remain unchanged.

        Returns:
            A dict with the identical size as pre_masks and its 0/1 values are updated.
                1 means unpruned and 0 means pruned.
        """
        masks = pre_masks
        k_blockwise = self.update_residual_cnt(masks, cur_target_sparsity_ratio)
        if k_blockwise <= 0:
            return masks
        new_scores = scores if self.block else self.reduce_scores(scores)
        not_exceed_layers = []
        residual_k = k_blockwise
        if self.min_sparsity_ratio_per_op > 0:
            sparsity_infos_perlayer, _ = self.get_sparsity_ratio_each_layer(masks)

        while True:
            new_not_exceed_layers = [key for key in new_scores.keys() if not self.keep_mask_layers.get(key, False)]
            if not_exceed_layers == new_not_exceed_layers or len(new_not_exceed_layers) == 0:
                break
            not_exceed_layers = new_not_exceed_layers
            global_scores = np.concatenate([tf.reshape(new_scores[key], [-1]).numpy() for key in not_exceed_layers])
            threshold = np.partition(global_scores, kth=residual_k)[residual_k]

            for key in not_exceed_layers:
                block_size = self.block_size[key]
                score = new_scores[key]
                mask = self.get_mask_per_threshold(score, threshold, block_size)
                info = self.get_sparsity_ratio({key: mask}, return_dict=True)
                zero_cnt = info["zero_cnt"]
                total_cnt = info["total_cnt"]
                current_sparsity_ratio = float(zero_cnt) / total_cnt
                key_new_sparsity = SparsityInfo(zero_cnt, total_cnt, current_sparsity_ratio)
                need_adjust, adjust_ratio = self.adjust_ratio(
                    masks,
                    key,
                    key_new_sparsity,
                    self.max_sparsity_ratio_per_op,
                    self.min_sparsity_ratio_per_op,
                    self.target_sparsity_ratio,
                )
                if need_adjust:
                    # uptade status
                    self.keep_mask_layers[key] = True
                    masks[key] = self.get_single_mask_per_target_ratio(new_scores[key], adjust_ratio)
                    if not self.block:
                        masks[key] = tf.repeat(masks[key], repeats=block_size[0], axis=0)
                        masks[key] = tf.repeat(masks[key], repeats=block_size[1], axis=-1)
                    if keep_exact_sparsity_ratio:
                        zero_cnt = self.get_sparsity_ratio({key: masks[key]}, return_dict=True)["zero_cnt"]
                        residual_k -= zero_cnt
                else:
                    masks[key] = mask
            if not keep_exact_sparsity_ratio:
                break

        for key in masks.keys():
            if key in self.invalid_layers:
                continue
            if len(scores[key].shape) == 4:  # need to permute
                mask = masks[key]
                orig_shape = scores[key].shape
                mask = self._reshape_2dims_to_orig(mask, orig_shape)
                masks[key] = mask
            layer_ratio = np.sum(masks[key] == 0.0) / masks[key].size
            logger.info(f"{key} sparsity is {layer_ratio}")
        return masks
