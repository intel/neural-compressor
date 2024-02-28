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

from .utils import *


class Calibration:
    def __init__(self, model, dataloder=None, q_func=None, device="cpu"):
        self.model = model
        self.dataloader = dataloder
        self.q_func = q_func
        self.device = device

    @torch.no_grad()
    def _save_input_pc_hook(self, name):
        """A forward hook to save input max of a module
        :param name: the module name
        :return: A hook function."""

        def save_input_hook(module, inputs, outputs):
            input = inputs[0]
            ##TODO check input channel is correct
            if len(module.weight.shape) == 4:  ##conv3d or conv1d not supported now, need better way
                input = input.permute(0, 2, 3, 1)
            input = input.reshape(-1, input.shape[-1])
            max_tensor = torch.max(input, dim=0)[0]
            min_tensor = torch.min(input, dim=0)[0]
            if name not in self.input_maxes.keys():
                self.input_mins[name], self.input_maxes[name] = min_tensor, max_tensor
            else:
                self.input_mins[name] = torch.min(self.input_mins[name], min_tensor)
                self.input_maxes[name] = torch.max(self.input_maxes[name], max_tensor)

        return save_input_hook

    @torch.no_grad()
    def _add_min_max_observer(self, modules):
        """
        :param modules: the modules which the observer will insert to
        :return:
        """
        self.hook_handles = []
        for key in modules.keys():
            hook_func = self._save_input_pc_hook(key)
            hook_handle = modules[key].register_forward_hook(hook_func)
            self.hook_handles.append(hook_handle)

    @torch.no_grad()
    def _remove_observer(self):
        """Remove the observer from the model
        :return:"""
        for hook_handle in self.hook_handles:
            hook_handle.remove()

    @torch.no_grad()
    def _dump_min_max(self, calib_iter=100):
        """Dump min max per channel information, the min max value will be saved in input_maxes attribute
        :param calibration_method: only support min_max currently
        :param calib_iter: Sample size for calibration
        :return:"""
        logger.info("Calibrating...")
        if self.q_func:
            self.q_func(self.model)
        else:
            assert self.dataloader, "Please set dataloader for calibration."
            model_forward(self.model, self.dataloader, calib_iter, self.device)

    @torch.no_grad()
    def calibrate(self, calib_iter, op_types=[torch.nn.Conv2d, torch.nn.Linear]):  ##TODO transformers.conv1d
        """
        :param absorb_to_layer: A dict,key is the absorb layer, val is a list of the to be smoothed layer
        :param calib_iter: Data size for calibration
        :return: A dict that saved the layer name and the channel-wise max value info
        """
        ##hook all the module
        self.input_mins = {}
        self.input_maxes = {}

        hook_modules = {}
        for n, module in self.model.named_modules():
            if isinstance(module, tuple(op_types)):
                hook_modules[n] = module

        self._add_min_max_observer(hook_modules)

        self._dump_min_max(calib_iter=calib_iter)
        self._remove_observer()
        return self.input_mins, self.input_maxes
