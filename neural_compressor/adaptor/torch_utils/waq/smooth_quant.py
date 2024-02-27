#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

import copy
import json

try:
    from neural_compressor.utils.utility import LazyImport

    torch = LazyImport("torch")
    from neural_compressor.utils import logger
except:
    import logging

    import torch

    logger = logging.getLogger()
from collections import UserDict, defaultdict

import numpy
from tqdm import tqdm

from .calibration import Calibration
from .graph_trace import GraphTrace
from .utils import *


class TorchSmoothQuant:
    """Fake input channel quantization, for more details please refer to
    [1] SmoothQuant: Accurate and Efficient
    Post-Training Quantization for Large Language Models
    [2] SPIQ: Data-Free Per-Channel Static Input Quantization
    Currently, we only handle the layers whose smooth scale could be absorbed, we will support other layers later.

    We only support inplace mode which means the model weights will be changed, you can call recover function
    to recover the weights if needed
    """

    def __init__(self, model, dataloader=None, example_inputs=None, q_func=None, traced_model=None):
        """
        :param model: Torch model :param dataloader: Calibration dataloader :param traced_model: A specific model
        shares the same architecture as the model and could be traced by torch.jit. If not supplied, we use model
        instead.
        """
        self.model = model
        if not isinstance(self.model, torch.nn.Module):
            return
        device, dtype = self._get_device()
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self.dtype = dtype
        self.dataloader = dataloader
        self.example_inputs = example_inputs
        self.q_func = q_func
        self.input_maxes = {}
        self.input_mins = {}
        self.input_maxes_abs = {}
        self.traced_model = traced_model
        if self.traced_model is None:
            self.traced_model = self.model
        self.weight_scale_info = {}
        self.absorb_scales_info = {}
        self.insert_mul = False
        self.allow_absorb = True
        self.record_max_info = False
        self.max_value_info = {}  # to record max values for alpha tune
        self.absorb_to_layer = {}
        self.weight_max_lb = 1e-5  ##weight max low bound
        self.weight_scale_dict = {}
        self.sq_scale_info = {}
        self.max_value_info = {}
        self.need_calibration = False

    def _get_device(self):
        """Get the model device
        :return:Model device."""
        for _, p in self.model.named_parameters():
            return p.data.device, p.data.dtype

    def _scale_layer_weight(self, layer_name, scale, alpha=0.5, input_minmax=None):  ##input channel
        """Scale the layer weights at input channel, depthwise conv output channel
        :param layer_name: The layer name
        :param scale: The scale to be multiplied
        :param alpha: alpha for SQLinearWrapper
        :param input_minmax: input_minmax for SQLinearWrapper
        :return:"""
        layer = get_module(self.model, layer_name)
        if self.insert_mul:
            from ..model_wrapper import SQLinearWrapper

            layer = get_module(self.model, layer_name)
            if isinstance(layer, SQLinearWrapper):
                layer._recover_sq_linear()
                set_module(self.model, layer_name, layer.sq_linear)  ##recover
            else:
                new_module = SQLinearWrapper(layer, 1.0 / scale, input_minmax, alpha)
                set_module(self.model, layer_name, new_module)
        elif self.allow_absorb:
            scale = reshape_scale_as_weight(layer, scale)
            layer.weight = torch.nn.Parameter(layer.weight * scale)
        return scale

    def _absorb_scales(self, layer_name, scale):  ##output channel
        """Absorb the scale to the layer at output channel
        :param layer_name: The module name
        :param scale: The scale to be absorbed
        :param alpha_key: The alpha passed to SQLinearWrapper
        :return:"""
        if self.insert_mul or not self.allow_absorb:
            return  # absorb is updated in SQLinearWrapper in def _scale_layer_weight

        ##if self.allow absorb
        layer = get_module(self.model, layer_name)
        if layer.__class__.__name__ == "WrapperLayer":
            layer = layer.orig_layer
        if (
            isinstance(layer, torch.nn.BatchNorm2d)
            or isinstance(layer, torch.nn.GroupNorm)
            or isinstance(layer, torch.nn.InstanceNorm2d)
        ):
            if layer.affine:
                layer.weight *= scale
                layer.bias *= scale
            else:
                layer.affine = True
                weight = torch.ones(layer.num_features, device=self.device, dtype=self.dtype) * scale
                layer.weight = torch.nn.Parameter(weight, requires_grad=False)
                bias = torch.zeros(layer.num_features, device=self.device, dtype=self.dtype)
                layer.bias = torch.nn.Parameter(bias, requires_grad=False)
        elif isinstance(layer, torch.nn.LayerNorm):
            if layer.elementwise_affine:
                layer.weight *= scale
                layer.bias *= scale
            else:
                layer.elementwise_affine = True
                weight = torch.ones(layer.num_features, device=self.device, dtype=self.dtype) * scale
                layer.weight = torch.nn.Parameter(torch.ones(weight, requires_grad=False))
                bias = torch.zeros(layer.num_features, device=self.device, dtype=self.dtype)
                layer.bias = torch.nn.Parameter(bias, requires_grad=False)

        elif isinstance(layer, torch.nn.Conv2d):
            ##the order could not be changed
            if hasattr(layer, "bias") and (layer.bias is not None):
                layer.bias *= scale
            scale = scale.view(scale.shape[0], 1, 1, 1)
            layer.weight *= scale

        elif isinstance(layer, torch.nn.Linear):
            if hasattr(layer, "bias") and (layer.bias is not None):
                layer.bias *= scale
            scale = scale.view(scale.shape[0], 1)
            layer.weight *= scale

        elif layer.__class__.__name__ == "LlamaRMSNorm" or layer.__class__.__name__ == "T5LayerNorm":  ##quite tricky
            layer.weight *= scale

        else:
            logger.warning(
                f"found unsupported layer {type(layer)}, try to multiply scale to "
                f"weight and bias directly, this may introduce accuracy issue, please have a check "
            )
            if hasattr(layer, "weight") and layer.weight is not None:
                layer.weight *= scale
            if hasattr(layer, "bias") and layer.bias is not None:
                layer.bias *= scale

    def _export_sq_info(self, absorb_to_layer, input_maxes, alpha=0.5):
        from ..model_wrapper import SQLinearWrapper

        absorb_to_input_maxes = {}
        for key in absorb_to_layer.keys():
            layer_name = absorb_to_layer[key][0]
            absorb_to_input_maxes[key] = input_maxes[layer_name]

        for index, key in enumerate(absorb_to_layer.keys()):
            alpha_tmp = alpha[key] if isinstance(alpha, dict) else alpha
            layer_names = absorb_to_layer[key]
            weights = []
            for layer_name in layer_names:
                weight = reshape_in_channel_to_last(layer_name, self.model)
                weights.append(weight)
            weight_max_per_channel = torch.max(torch.abs(torch.cat(weights, dim=0)), dim=0)[0]

            weight_max_per_channel = weight_max_per_channel.clamp(min=self.weight_max_lb)

            input_max = absorb_to_input_maxes[key]
            layer_names = absorb_to_layer[key]
            # weight_scale = cal_scale(input_max, weights, alpha_tmp)
            input_minmax = [self.input_mins[layer_names[0]].to("cpu"), self.input_maxes[layer_names[0]].to("cpu")]
            abs_input_max = torch.max(torch.abs(input_minmax[0]), torch.abs(input_minmax[1]))
            input_power = torch.pow(abs_input_max, alpha_tmp)
            weight_power = torch.pow(weight_max_per_channel, 1 - alpha_tmp)
            weight_scale = torch.clip(input_power / weight_power, min=1e-5)

            input_scale = 1.0 / weight_scale

            self.max_value_info[key] = {
                "alpha": alpha_tmp,
                "input_minmax": input_minmax,
                "weight_max": weight_max_per_channel,
                "absorbed_layer": layer_names,
            }  # max_value_info is used for pytorch backend and sq_scale_info is used for ipex backend.
            # the input of layers with same absorb layer is the same.
            for op_name in layer_names:
                module = copy.deepcopy(get_module(self.model, op_name))
                new_module = SQLinearWrapper(module, 1.0 / weight_scale, input_minmax, alpha_tmp)
                self.sq_scale_info[op_name] = {}
                self.sq_scale_info[op_name] = {
                    "alpha": alpha_tmp,
                    "input_scale_for_mul": input_scale.to("cpu"),
                    "input_scale_after_mul": new_module.scale,
                    "input_zero_point_after_mul": new_module.zero_point,
                    "input_dtype": new_module.dtype,
                    "weight_scale_after_mul": new_module._get_weight_scale(),
                }

    def _cal_scales(self, absorb_to_layer, input_maxes, alpha=0.5):
        """Cal the adjust scales
        :param absorb_to_layer: A dict mapping absorb layer to smooth quantized layer
        :param input_maxes: The channel-wise input max info for layers
        :param alpha: Alpha value to balance the quantization difficulty of activation and weight, a float of a dict
        :return:"""
        absorb_to_input_maxes = {}
        for key in absorb_to_layer.keys():
            layer_name = absorb_to_layer[key][0]
            absorb_to_input_maxes[key] = input_maxes[layer_name]

        weight_scales_info = {}
        absorb_scales_info = {}
        for index, key in enumerate(absorb_to_layer.keys()):
            alpha_tmp = alpha[key] if isinstance(alpha, dict) else alpha

            input_max = absorb_to_input_maxes[key]
            layer_names = absorb_to_layer[key]
            weights = []
            for layer_name in layer_names:
                weight = reshape_in_channel_to_last(layer_name, self.model)
                weights.append(weight)
            scale = cal_scale(input_max, weights, alpha_tmp)
            absorb_scales_info[key] = 1.0 / scale
            absorb_scales_info[key][scale == 0] = 0
            layer_names = absorb_to_layer[key]
            for layer_name in layer_names:
                ##self._scale_layer_weight(layer_name, scale)
                weight_scales_info[layer_name] = scale
        return absorb_scales_info, weight_scales_info

    def _adjust_parameters(self, absorb_to_layer, input_maxes, alpha=0.5):
        """Adjust the weights and biases
        :param absorb_to_layer: A dict mapping absorb layer to smooth quantized layer
        :param input_maxes: The channel-wise input max info for layers
        :param alpha: Alpha value to balance the quantization difficulty of activation and weight, a float of a dict
        :return:"""
        absorb_scales_info, weight_scales_info = self._cal_scales(absorb_to_layer, input_maxes, alpha)
        if not absorb_scales_info or not weight_scales_info:
            return weight_scales_info, absorb_scales_info
        for index, key in enumerate(absorb_to_layer.keys()):
            if isinstance(alpha, float):
                alpha_tmp = alpha
            elif isinstance(alpha, dict):
                alpha_tmp = alpha[key]
            absorb_scale = absorb_scales_info[key]
            self._absorb_scales(key, absorb_scale)
            layer_names = absorb_to_layer[key]
            for layer_name in layer_names:
                input_minmax = [self.input_mins[layer_names[0]], self.input_maxes[layer_names[0]]]
                self._scale_layer_weight(layer_name, weight_scales_info[layer_name], alpha_tmp, input_minmax)
        return weight_scales_info, absorb_scales_info

    def _check_need_calibration(self, alpha, percentile, op_types, scales_per_op, calib_iter):
        """
        check need calibration or not
        :param alpha: current alpha
        :param percentile: current percentile
        :param op_types: current op_types
        :param scales_per_op: current scales_per_op
        :param calib_iter:: current scales_per_op
        :return:
        """
        need_calib = True
        from peft import PeftModel

        is_peft, is_auto = isinstance(self.model, PeftModel), alpha == "auto"
        if len(self.input_maxes) == 0:  ## the first time
            need_calib = True
            self.alpha = alpha
            self.percentile = percentile
            self.op_types = op_types
            self.scales_per_op = scales_per_op
            self.calib_iter = calib_iter
            return False if (is_auto and not is_peft) else need_calib

        if (
            self.percentile == percentile
            and self.op_types == op_types
            and self.scales_per_op == scales_per_op
            and self.calib_iter == calib_iter
        ):
            if isinstance(alpha, float) or self.alpha == "auto":
                need_calib = False

        self.alpha, self.percentile, self.calib_iter = alpha, percentile, calib_iter
        self.op_types, self.scales_per_op = op_types, scales_per_op
        return need_calib

    @torch.no_grad()
    def _parse_absorb_to_layers(self, op_types, folding):
        str_op_types = [i.__name__ for i in op_types]
        self_absorb_layers = {}
        if self.insert_mul:
            self_absorb_layers = self._get_all_layer_names(op_types)  # TODO: only support linear now.
            # fetch modules with the same input
            group_modules = self._trace(str_op_types, skip_unsupported_layers=False)
            if group_modules is not None:
                # use one input for qkv
                for k, v in group_modules.items():
                    for i in v:
                        if i in self_absorb_layers:
                            self_absorb_layers.pop(i)
                    self_absorb_layers[v[0]] = v
                logger.debug(f"self_absorb_layers:{self_absorb_layers}")
        if self.allow_absorb:
            self.absorb_to_layer, no_absorb_layers = self._trace(str_op_types)
            if self.absorb_to_layer is None and no_absorb_layers is None:
                return None

        # remove self.self_absorb_layers if it exists in self.absorb_to_layer
        for k, v in self.absorb_to_layer.items():
            for i in v:
                if i in self_absorb_layers:
                    self_absorb_layers.pop(i)
        self.absorb_to_layer.update(self_absorb_layers)

        if self.absorb_to_layer is None and no_absorb_layers is None:
            logger.warning(
                "sorry, could not trace the model, smooth quant is ignored."
                "If you are using huggingface model,"
                "you could set torchscript to True "
            )
            return None

        # Check if input_maxes match self.absorb_to_layer
        # (due to self._get_all_layer_names use layer tree instead of forward_path)
        if not folding and self.need_calibration:
            if len(self.input_mins) == 0:  ##there are some modules not used in forward
                calib = Calibration(self.model, self.dataloader, self.q_func, self.device)  ##
                input_mins, input_maxes = calib.calibrate(
                    1, op_types
                )  ##TODO if using qfunc for calibration, it will calibrate twice
            # use qfunc to calibrate, the input min could be used for fixed alpha transformation
            self.input_mins = input_mins
            self.input_maxes = input_maxes
            diff_modules = set(self.absorb_to_layer.keys()).difference(input_mins.keys())
            for d in diff_modules:
                del self.absorb_to_layer[d]
        return self.absorb_to_layer

    @torch.no_grad()
    def transform(
        self,
        alpha=0.5,
        folding=False,
        percentile=100,
        op_types=[torch.nn.Linear, torch.nn.Conv2d],
        scales_per_op=False,
        calib_iter=100,
        weight_clip=True,
        auto_alpha_args={
            "init_alpha": 0.5,
            "alpha_min": 0.0,
            "alpha_max": 1.0,
            "alpha_step": 0.1,
            "shared_criterion": "mean",
            "n_samples": 32,  ##512 for cuda, 128 for cpu?
        },
    ):
        """The main entry of smooth quant
        :param alpha: Alpha value to balance the quantization difficulty of activation and weight, please refer
        to the paper for more details
        :param folding: whether insert mul(False) or just allow foldable layers(True) for SmoothQuant
        :param percentile: Not supported now
        :param op_types: The op typed to be smooth quantized
        :param scales_per_op: Not supported now
        :param calib_iter: Data size for calibration
        :param weight_clip: Whether to clip weight_max when calculating scales.

        :param auto_alpha_args: Hyperparameters used to set the alpha search space in SQ auto-tuning.
            By default, the search space is 0.0-1.0 with step_size 0.1.
            do_blockwise: Whether to do blockwise auto-tuning.
        :param init_alpha: A hyperparameter that is used in SQ auto-tuning; by default it is 0.5.
        :return: A FP32 model with the same architecture as the orig model but with different weight which will be
        benefit to quantization.
        """

        if not isinstance(self.model, torch.nn.Module):
            logger.warning("smoothquant is ignored since the model is not a torch module")
            return self.model

        if isinstance(alpha, float) and (alpha < 0):
            logger.warning("reset alpha to >=0")
            alpha = numpy.clip(alpha, 0.0)

        if folding:
            self.insert_mul, self.allow_absorb = False, True
        else:
            self.insert_mul, self.allow_absorb = True, False
        self.weight_clip = weight_clip

        self.revert()
        self.need_calibration = self._check_need_calibration(alpha, percentile, op_types, scales_per_op, calib_iter)
        if self.need_calibration:
            self.input_mins, self.input_maxes = {}, {}
        self.absorb_to_layer = self._parse_absorb_to_layers(
            op_types, folding
        )  ##need to forward to check modules not used in forward
        if len(self.input_mins) != 0:  ##this is from _parse_absorb_to_layers, ugly code to support q_func
            input_maxes_abs = {}
            for key in self.input_mins.keys():
                input_maxes_abs[key] = torch.max(torch.abs(self.input_mins[key]), torch.abs(self.input_maxes[key]))
            if self.q_func:
                self.need_calibration = False  # Avoid double-calibration in fixed-value alpha SQ.

        if self.absorb_to_layer is None:
            logger.warning("empty absorb_to_layer, smoothquant is ignored ")
            return self.model
        example_inputs = self._get_example_input()
        if alpha == "auto":  ##TODO need to polish later
            from . import auto_alpha
            from .utils import TUNERS

            auto_alpha_version = "version1"
            auto_alpha_tuner = TUNERS[auto_alpha_version](
                self.model,
                self.dataloader,
                self.absorb_to_layer,
                op_types=op_types,
                device=self.device,
                q_func=self.q_func,
                folding=folding,
                example_inputs=self.example_inputs,
                **auto_alpha_args,
            )
            self.alpha = auto_alpha_tuner.tune()
            input_maxes_abs = auto_alpha_tuner.input_maxes_abs
            self.input_mins, self.input_maxes = auto_alpha_tuner.input_mins, auto_alpha_tuner.input_maxes
            if auto_alpha_tuner.loss_type == "blockwise":
                self.block_names = auto_alpha_tuner.block_names

        elif self.need_calibration:
            calib = Calibration(self.model, self.dataloader, self.q_func, self.device)
            self.input_mins, self.input_maxes = calib.calibrate(calib_iter, op_types)
            input_maxes_abs = {}
            for key in self.input_mins.keys():
                input_maxes_abs[key] = torch.max(torch.abs(self.input_mins[key]), torch.abs(self.input_maxes[key]))

        if example_inputs is not None:
            out_pre_sq = model_forward_per_sample(self.model, example_inputs, self.device)

        if folding:
            self._save_scale = False  ##TODO remove it later

        if self.record_max_info:
            self._export_sq_info(self.absorb_to_layer, input_maxes_abs, self.alpha)
            # # max_info is recorded in self.max_value_info
            # self._adjust_parameters(self.absorb_to_layer, input_maxes_abs, alpha)
            self.model._smoothquant_optimized = False
            return self.model

        self.weight_scale_info, self.absorb_scales_info = self._adjust_parameters(
            self.absorb_to_layer, input_maxes_abs, self.alpha
        )
        self.model._smoothquant_optimized = True

        if example_inputs is not None:
            # Check mathematical equivalency
            out_post_sq = model_forward_per_sample(self.model, example_inputs, self.device)
            if not self.output_is_equal(out_post_sq, out_pre_sq):
                logger.warning(
                    "Mathematical equivelancy of Smoothquant is not preserved. "
                    "Please kindly report this issue to https://github.com/intel/neural-compressor."
                )
        else:
            logger.warning(" Could not get example input, equivelancy check is skipped")

        return self.model

    def output_is_equal(self, out1, out2, atol=1e-04):
        try:
            if isinstance(out1, tuple):
                return all(torch.all(torch.isclose(out1[i], out2[i], atol=atol)) for i in range(len(out1)))
            elif isinstance(out1, dict):
                return all(torch.all(torch.isclose(out1[k], out2[k], atol=atol)) for k in out1.keys())
            elif isinstance(out1, torch.Tensor):
                return torch.all(torch.isclose(out1, out2, atol=atol))
            return False
        except:
            logger.warning(
                "Automatically check failed, Please check equivelancy manually "
                "between out_pre_sq and out_post_sq if necessary."
            )
            return True

    @torch.no_grad()
    def revert(self):
        """Revert the model weights
        :return:"""
        for key in self.weight_scale_info:
            self._scale_layer_weight(key, 1.0 / self.weight_scale_info[key])
        for key in self.absorb_scales_info:
            self._absorb_scales(key, 1.0 / self.absorb_scales_info[key])
        self.weight_scale_info = {}  ##clear the data
        self.absorb_scales_info = {}

    def _get_all_layer_names(self, op_types=[torch.nn.Linear]):
        """Try the model to find the layers which can be smooth quantized.

        :param op_types: The op types to be smooth quantized
        :return:
        self_absorb_layer: A dict, absorb layer name (itself): layers to be smooth quantized
        """
        self_absorb_layer = {}
        op_types = [torch.nn.Linear]  # TODO： only support SQLinearWrapper
        for name, module in self.model.named_modules():
            if isinstance(module, tuple(op_types)):
                self_absorb_layer[name] = [name]
        return self_absorb_layer

    def _get_example_input(self):
        if self.dataloader is None and self.example_inputs is None:
            return None
        if self.example_inputs is None:
            try:
                for idx, (input, label) in enumerate(self.dataloader):
                    self.example_inputs = input
                    break
            except:
                for idx, input in enumerate(self.dataloader):
                    self.example_inputs = input
                    break

        return self.example_inputs

    def _trace(self, op_types, skip_unsupported_layers=True):
        """Try the model to find the layers which can be smooth quantized.

        :param op_types: The op types to be smooth quantized
        :return:
        absorb_to_layer: A dict, absorb layer name:layers to be smooth quantized
        no_absorb_layers: A list saving the layers which could not find the absorb layer
        """

        tg = GraphTrace()
        self._get_example_input()
        absorb_to_layer, no_absorb_layers = tg.get_absorb_to_layer(
            self.traced_model,
            self.example_inputs,
            op_types,
            skip_unsupported_layers=skip_unsupported_layers,
        )
        if not skip_unsupported_layers:
            return absorb_to_layer
        if absorb_to_layer is None and no_absorb_layers is None:
            logger.warning(
                "sorry, could not trace the model, smooth quant is skipped."
                "If you are using huggingface model,"
                "you could set torchscript to True "
                "when loading the model or set the return_dict to False"
            )
        elif absorb_to_layer == {}:
            logger.warning("could not find any layer to be absorbed")
        else:
            to_absorb_cnt = 0
            for key, item in absorb_to_layer.items():
                to_absorb_cnt += len(item)
            logger.info(
                f" {to_absorb_cnt} out of {to_absorb_cnt + len(no_absorb_layers)} "
                f"layers could be absorbed in smooth quant"
            )
        return absorb_to_layer, no_absorb_layers
