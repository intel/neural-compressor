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
from .round_scales_function import ScaleIdentity
#TODO [SW-224612]: Use cguid to calc scales and remove is_calc_scale_with_cguid
from ..common import QuantTensorType, is_calc_scale_with_cguid
from ..fp_utils import mmse_scale_multi, get_fullscale, mmse_scale, calc_maxabs_scale, invert_scale, calculate_scale_maxabs, ScaleCalculationMaxMode
from ...utils.logger import logger

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
        #TODO [SW-224612]: Use cguid to calc scales and remove check
        self.calc_scale_with_cguid = is_calc_scale_with_cguid() if is_dynamic else False
        logger.trace("%s %s", self.__class__.__name__, self.__dict__)

    def __repr__(self):
        return f"{self.__class__.__name__}(round_scale_method={self.round_scale_method.__class__.__name__}, params={self.params}, " \
            f"fullscale={self.fullscale}, scale={self.scale}, is_dynamic={self.is_dynamic})"

    def calc_invert_scales(self):
        return self.invert_scales(self.scale)

    def invert_scales(self, scale=None):
        return invert_scale(scale)

    @abstractmethod
    def calc_scales(self, tensor, tensor_type, **additional_kwargs):
        raise NotImplementedError("`calc_scales` function is not implemented")


class MaxAbsMethod(ScalesMethod):
    def __init__(self, round_scale_method, params, device_for_scales, backoff, fullscale=None, is_dynamic=False):
        super().__init__(round_scale_method, params, device_for_scales, fullscale, is_dynamic)
        self.backoff = backoff
        self.calc_maxabs_method_dict = self.get_maxabs_methods_dict()
        logger.trace("%s %s", self.__class__.__name__, self.__dict__)

    def get_maxabs_methods_dict(self):
        return {QuantTensorType.CONST: self.calc_maxabs_for_real_tensor,
                        QuantTensorType.MEASUREMENTS: self.calc_maxabs_for_measurement}

    @abstractmethod
    def calc_maxabs_for_real_tensor(self, tensor):
        raise NotImplementedError("`calc_maxabs_for_real_tensor` function is not implemented")

    def calc_maxabs_for_measurement(self, tensor):
        return torch.tensor(tensor, dtype=self.hp_dtype, device=self.device).max()

    def _calculate_maxabs_scale(self, tensor, tensor_type, **additional_kwargs):
        max_abs_tensor = self.calc_maxabs_method_dict[tensor_type](tensor)
        #TODO [SW-224612]: Use cguid to calc scales and remove check
        if not self.calc_scale_with_cguid:
            max_abs_tensor = calc_maxabs_scale(max_abs_tensor, self.fullscale, self.backoff)

        scale = self.round_scale_method.calc(max_abs_tensor)
        return scale

    def calc_scales(self, tensor, tensor_type, **additional_kwargs):
        self.scale = self._calculate_maxabs_scale(tensor, tensor_type, **additional_kwargs)
        return self.scale

class MaxAbsPts(MaxAbsMethod):
    def __init__(self, round_scale_method, params, device_for_scales, backoff, fullscale=None, is_dynamic=False):
        super().__init__(round_scale_method, params, device_for_scales, backoff, fullscale, is_dynamic)
        logger.trace("%s %s", self.__class__.__name__, self.__dict__)

    def calc_maxabs_for_real_tensor(self, tensor):
        #TODO [SW-224612]: Use cguid to calc scales and remove check
        if self.calc_scale_with_cguid:
            max_abs_tensor = calculate_scale_maxabs(tensor, ScaleCalculationMaxMode.MAX_ABS_PTS_CALCULATION, fullscale=self.fullscale, backoff=self.backoff)
        else:
            max_abs_tensor =  torch.max(
                torch.abs(tensor.detach())).to(dtype=self.hp_dtype, device=self.device)
        return max_abs_tensor


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


class MaxAbsPcs(MaxAbsMethod):

    def __init__(self, round_scale_method, params, device_for_scales, backoff, fullscale=None, dim=1, keepdim=False, is_dynamic=False):
        super().__init__(round_scale_method, params, device_for_scales, backoff, fullscale, is_dynamic=is_dynamic)
        self.dim = dim
        self.keepdim = keepdim
        logger.trace("%s %s", self.__class__.__name__, self.__dict__)

    def calc_maxabs_for_real_tensor_no_reshape(self, tensor):
        #TODO [SW-224612]: Use cguid to calc scales and remove check
        if self.calc_scale_with_cguid:
            max_abs_tensor = calculate_scale_maxabs(
                tensor,
                ScaleCalculationMaxMode.MAX_ABS_PCS_CALCULATION,
                reduceAxis=self.dim,
                reduceKeepdim=self.keepdim,
                fullscale=self.fullscale,
                backoff=self.backoff,
            )
        else:
            max_abs_tensor = torch.amax(torch.abs(tensor), dim=self.dim, keepdim=self.keepdim)
        return max_abs_tensor

    def calc_maxabs_for_real_tensor(self, tensor):
        max_abs_tensor = self.calc_maxabs_for_real_tensor_no_reshape(tensor).reshape([-1, 1])
        if self.dim == 1:
            max_abs_tensor = max_abs_tensor.flatten()
        return max_abs_tensor


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


class MaxAbsDynamicPcs(MaxAbsPcs):

    def __init__(self, round_scale_method, params, device_for_scales, backoff, fullscale=None):
        super().__init__(round_scale_method, params, device_for_scales, backoff, fullscale, -1, True, True)
        logger.trace("%s %s", self.__class__.__name__, self.__dict__)

    def get_maxabs_methods_dict(self):
        maxabs_methods_dict = super().get_maxabs_methods_dict()
        maxabs_methods_dict[QuantTensorType.DYNAMIC] = self.calc_maxabs_for_real_tensor_no_reshape
        return maxabs_methods_dict

    def calc_scales(self, tensor, tensor_type, **additional_kwargs):
        # In dynamic quantization the scale is changed each time,
        # and setting scale as a member is not supported in hpu graphs and torch.compile
        # (it can break the graph)
        return self._calculate_maxabs_scale(tensor, tensor_type, **additional_kwargs)


class MaxAbsDynamicPts(MaxAbsPts):

    def __init__(self, round_scale_method, params, device_for_scales, backoff, fullscale=None):
        super().__init__(round_scale_method, params, device_for_scales, backoff, fullscale, True)
        logger.trace("%s %s",self.__class__.__name__, self.__dict__)

    def get_maxabs_methods_dict(self):
        maxabs_methods_dict = super().get_maxabs_methods_dict()
        maxabs_methods_dict[QuantTensorType.DYNAMIC] = self.calc_maxabs_for_real_tensor
        return maxabs_methods_dict

    def calc_scales(self, tensor, tensor_type, **additional_kwargs):
        # In dynamic quantization the scale is changed each time,
        # and setting scale as a member is not supported in hpu graphs and torch.compile
        # (it can break the graph)
        return self._calculate_maxabs_scale(tensor, tensor_type, **additional_kwargs)
