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
from abc import abstractmethod
from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator
import torch
from . import ScaleIdentity
from ..common import QuantTensorType
from ..fp_utils import mmse_scale_multi, get_fullscale, mmse_scale, calc_maxabs_scale, invert_scale

class ScalesMethod:

    curr_device = auto_detect_accelerator()

    def __init__(self, round_scale_method, params, device_for_scale, fullscale=None, is_dynamic=False):
        self.round_scale_method = round_scale_method
        self.params = params
        self.hp_dtype = self.params["hp_dtype"]
        self.lp_dtype = self.params["lp_dtype"]
        self.device = torch.device(self.curr_device.name())
        self.fullscale = fullscale if fullscale is not None else get_fullscale(self.lp_dtype, device_for_scale)
        self.scale = None
        self.is_dynamic = is_dynamic

    def __repr__(self):
        return f"{self.__class__.__name__}(round_scale_method={self.round_scale_method.__class__.__name__}, params={self.params}, " \
            f"fullscale={self.fullscale}, scale={self.scale}, is_dynamic={self.is_dynamic})"

    def calc_invert_scales(self):
        return invert_scale(self.scale)

    @abstractmethod
    def calc_scales(self, tensor, tensor_type, **additional_kwargs):
        raise NotImplementedError("`calc_scales` function is not implemented")


class MaxAbsPts(ScalesMethod):
    def __init__(self, round_scale_method, params, device_for_scales, backoff, fullscale=None):
        super().__init__(round_scale_method, params, device_for_scales, fullscale)
        self.backoff = backoff

    def calc_scales(self, tensor, tensor_type, **additional_kwargs):
        if tensor_type == QuantTensorType.CONST:
            max_abs_input =  torch.max(
                torch.abs(tensor.detach())).to(dtype=self.hp_dtype, device=self.device)
        else:
            max_abs_input = torch.tensor(tensor, dtype=self.hp_dtype ,device=self.device).max()
        scale = calc_maxabs_scale(max_abs_input, self.fullscale, self.backoff)
        self.scale = self.round_scale_method.calc(scale)
        return self.scale


## MulAdditionalScales Get 2 input scales, and return their multiplication.
# used for linear and matmul outputs
class MulAdditionalScales(ScalesMethod):
    def __init__(self, round_scale_method, params, device_for_scales, is_dynamic=False):
        super().__init__(round_scale_method, params, device_for_scales, is_dynamic=is_dynamic)

    def calc_scales(self, tensor, tensor_type, **additional_kwargs):
        input0 = additional_kwargs["input0"]
        input1 = additional_kwargs["input1"]
        self.scale = input0 * input1
        return self.scale


class MulAdditionalDynamicScales(MulAdditionalScales):
    def __init__(self, round_scale_method, params, device_for_scales):
        super().__init__(round_scale_method, params, device_for_scales, is_dynamic=True)

    def calc_scales(self, tensor, tensor_type, **additional_kwargs):
        if tensor_type != QuantTensorType.DYNAMIC:
            return None
        return super().calc_scales(tensor, tensor_type, **additional_kwargs)


## UseFirstAdditionalScales Get 2 input scales, and return the first one.
# used for linear smooth quant output
class UseFirstAdditionalScales(ScalesMethod):
    def __init__(self, round_scale_method, params, device_for_scales):
        super().__init__(round_scale_method, params, device_for_scales)

    def calc_scales(self, tensor, tensor_type, **additional_kwargs):
        input0 = additional_kwargs["input0"]
        self.scale = input0
        return self.scale

## DummyScales always return 1.0 as scale,
# used when running with dummy measurement (prepare_model_with_dummy_measurement)
class DummyScales(ScalesMethod):
    def calc_scales(self, tensor, tensor_type, **additional_kwargs):
        self.scale = torch.tensor(1.0).to("hpu")
        return self.scale

class MaxAbsPcs(ScalesMethod):

    def __init__(self, round_scale_method, params, device_for_scales, backoff, fullscale=None, dim=1, keepdim=False, is_dynamic=False):
        super().__init__(round_scale_method, params, device_for_scales, fullscale, is_dynamic=is_dynamic)
        self.backoff = backoff
        self.dim = dim
        self.keepdim = keepdim

    def calc_scales(self, tensor, tensor_type, **additional_kwargs):
        if tensor_type in [QuantTensorType.CONST, QuantTensorType.DYNAMIC]:
            max_abs_input = torch.amax(torch.abs(tensor), dim=self.dim, keepdim=self.keepdim)
            # on dynamic quantization we don't need to reshape
            if self.dim != -1:
                max_abs_input = max_abs_input.reshape([-1, 1])
            if self.dim == 1:
                max_abs_input = max_abs_input.flatten()
        else:
            max_abs_input = torch.tensor(tensor, dtype=self.hp_dtype ,device=self.device).max()
        scale = calc_maxabs_scale(max_abs_input, self.fullscale, self.backoff)
        scale = self.round_scale_method.calc(scale)
        self.scale = scale
        return scale


## InputChannelScale used for input channel in PCS mode
class InputChannelScale(ScalesMethod):
    def __init__(self, round_scale_method, params, device_for_scales, in_channel_size):
        super().__init__(round_scale_method, params, device_for_scales)
        self.in_channel_size = in_channel_size

    def calc_scales(self, tensor, tensor_type, **additional_kwargs):
        input_in_ch = torch.ones([self.in_channel_size, 1], dtype=self.hp_dtype, device=self.device)
        return  input_in_ch.flatten()

class FixedScale(ScalesMethod):
    
    def __init__(self, round_scale_method, params, device_for_scales):
        super().__init__(round_scale_method, params, device_for_scales)
        self.round_scale_method = round_scale_method

    def calc_scales(self, tensor, tensor_type, **additional_kwargs):
        self.scale = torch.tensor(self.round_scale_method.calc(tensor), dtype=self.hp_dtype, device=self.device)
        return self.scale


class OptScalesPts(ScalesMethod):

    def __init__(self, round_scale_method, optional_scales_list, params, device_for_scales, backoff):
        super().__init__(round_scale_method, params, device_for_scales)
        self.round_scale_method = round_scale_method
        self.optional_scales_list = optional_scales_list
        self.backoff = backoff

    def calc_scales(self, tensor, tensor_type, **additional_kwargs):
        self.scale = self.round_scale_method.calc(mmse_scale(tensor, self.optional_scales_list, self.lp_dtype, self.hp_dtype))
        return self.scale

class OptScalesPcs(ScalesMethod):
    def __init__(self, round_scale_method, optional_scales_list, params, device_for_scales, backoff):
        super().__init__(round_scale_method, params, device_for_scales)
        self.round_scale_method = round_scale_method
        self.optional_scales_list = optional_scales_list
        self.backoff = backoff
        self.device_for_scales = device_for_scales

    def calc_scales(self, tensor, tensor_type, **additional_kwargs):
        maxabs_scales_as_ref = MaxAbsPcs(self.round_scale_method, self.params, self.device_for_scales,
                                         self.backoff).calc_scales(tensor, tensor_type)
        const_opt_scale_out_ch = mmse_scale_multi(
            torch.transpose(tensor, 0, 1),
            maxabs_scales_as_ref,
            self.optional_scales_list,
            self.lp_dtype,
            self.hp_dtype,
        ).unsqueeze(1)
        self.scale = self.round_scale_method.calc(const_opt_scale_out_ch).flatten()
        return self.scale


class InputSmoothQuantMaxAbs(ScalesMethod):
    def __init__(self, round_scale_method, weight, params, device_for_scales, backoff):
        super().__init__(round_scale_method, params, device_for_scales)
        self.round_scale_method = round_scale_method
        self.weight = weight
        self.alpha = params["alpha"]
        self.backoff = backoff
        self.device_for_scales = device_for_scales

    def calc_scales(self, tensor, tensor_type, **additional_kwargs):
        weight_scale_in_ch = MaxAbsPcs(ScaleIdentity(), self.params, self.device_for_scales, 1.0, 1.0, dim=0).calc_scales(
            self.weight, QuantTensorType.CONST)
        input_range = torch.tensor(tensor, dtype=self.hp_dtype, device=self.device)
        input_scale = MaxAbsPts(ScaleIdentity(), self.params, self.device_for_scales, 1.0, 1.0).calc_scales(tensor,
                                                                                                            QuantTensorType.MEASUREMENTS)
        input_scale = (input_scale ** self.alpha) / (weight_scale_in_ch ** (1 - self.alpha))
        input_scale = self.round_scale_method.calc(input_scale)
        input_range_post = input_range / input_scale
        input_scale_post = calc_maxabs_scale(input_range_post.max(), self.fullscale, self.backoff)
        input_scale_post = self.round_scale_method.calc(input_scale_post)
        input_scale = input_scale * input_scale_post
        self.scale = input_scale
        return self.scale

class InputSmoothQuantOpt(ScalesMethod):
    def __init__(self, round_scale_method, weight, params, device_for_scales, backoff, backoff_weight):
        super().__init__(round_scale_method, params, device_for_scales)
        self.round_scale_method = round_scale_method
        self.weight = weight
        self.alpha = params["alpha"]
        self.backoff = backoff
        self.backoff_weight = backoff_weight
        self.device_for_scales = device_for_scales

    def calc_scales(self, tensor, tensor_type, **additional_kwargs):
        weight_scale_in_ch = MaxAbsPcs(ScaleIdentity(), self.params, self.device_for_scales, self.backoff_weight,
                                       self.fullscale, dim=0).calc_scales(self.weight, QuantTensorType.CONST)
        input_scale = MaxAbsPts(ScaleIdentity(), self.params, self.device_for_scales, self.backoff,
                                self.fullscale).calc_scales(tensor, QuantTensorType.MEASUREMENTS)
        input_scale = (input_scale ** self.alpha) / (weight_scale_in_ch ** (1 - self.alpha))
        input_scale = self.round_scale_method.calc(input_scale)
        self.scale = input_scale
        return self.scale


class WeightIchSmoothQuant(ScalesMethod):
    def __init__(self, round_scale_method, params, device_for_scales):
        super().__init__(round_scale_method, params, device_for_scales)

    def calc_scales(self, tensor, tensor_type, **additional_kwargs):
        self.scale = 1 / tensor
        return self.scale


class MaxAbsDynamicPcs(MaxAbsPcs):

    def __init__(self, round_scale_method, params, device_for_scales, backoff, fullscale=None):
        super().__init__(round_scale_method, params, device_for_scales, backoff, fullscale, dim=-1, keepdim=True, is_dynamic=True)

    # returns either None when not supplying dynamic tensor (during init..)
    # or calculates the scale
    def calc_scales(self, tensor, tensor_type, **additional_kwargs):
        if tensor_type != QuantTensorType.DYNAMIC:
            return None
        return super().calc_scales(tensor, tensor_type, **additional_kwargs)
