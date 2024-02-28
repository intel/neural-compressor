#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
"""Layer wise quantization."""
import os
import shutil
from collections import UserDict
from copy import deepcopy

from accelerate.utils import set_module_tensor_to_device
from torch.quantization import convert, prepare
from tqdm import tqdm

from neural_compressor.adaptor.torch_utils.waq import TorchSmoothQuant
from neural_compressor.config import default_workspace

from ..model_wrapper import QDQLayer
from .utils import (
    _get_path,
    clean_module_weight,
    get_named_children,
    load_tensor,
    load_tensor_from_shard,
    logger,
    register_weight_hooks,
    torch,
    update_module,
)

TMP_DIR = os.path.join(default_workspace, "lwq_tmpdir")


def mk_tmp_dir():
    os.makedirs(TMP_DIR, exist_ok=True)


def del_tmp_dir():
    shutil.rmtree(TMP_DIR)


def forward_wrapper(model, input, device="cpu"):
    if isinstance(input, dict) or isinstance(input, UserDict):
        if device == "cpu":
            output = model(**input)
        else:  # pragma: no cover
            for inp in input.keys():
                input[inp] = input[inp].to(device) if isinstance(input[inp], torch.Tensor) else input[inp]
            output = model(**input)
    elif isinstance(input, list) or isinstance(input, tuple):  # pragma: no cover
        if device == "cpu":
            output = model(*input)
        else:  # pragma: no cover
            input = [inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in input]  # pylint: disable=E1133
            output = model(*input)
    else:  # pragma: no cover
        if device == "cpu" or not isinstance(input, torch.Tensor):
            output = model(input)
        else:  # pragma: no cover
            input = input.to(device)  # pylint: disable=no-member
            output = model(input)
    return output


class LayerWiseQuant:
    """Layer wise quantization.

    Layer-by-layer quantize the model, in order to save memomery.
    """

    def __init__(
        self,
        q_model,
        pretrained_model_name_or_path,
        op_cfgs,
        calib_data,
        smooth_quant=False,
        output_dir=None,
        device="cpu",
        alpha=0.5,
    ):
        """Init LayerWiseQuant."""
        # self.q_model = load_empty_model(pretrained_model_name_or_path, cls)
        self.q_model = q_model
        self.fp32_model = deepcopy(self.q_model)
        self.path = _get_path(pretrained_model_name_or_path)
        self.op_cfgs = op_cfgs
        self.calib_data = calib_data
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        self.modules = get_named_children(self.q_model)
        self.device = device
        self._handle = {}
        self.quantized_layers = {}

        self.smooth_quant = smooth_quant
        self.alpha = float(alpha)
        assert self.alpha > 0 and self.alpha < 1, f"alpha should be in range (0, 1), but got {alpha}"
        if smooth_quant:
            self.init_sq()

    def init_sq(self):
        handles = register_weight_hooks(self.fp32_model, self.path)

        # traced_model = torch.jit.trace(self.fp32_model, torch.randint(0, 100, (1, 3)))
        op_types = ["Linear"]
        sq = TorchSmoothQuant(self.fp32_model, self.calib_data)
        sq._check_need_calibration(
            self.alpha, percentile=99.999, op_types=["Linear", "Conv2d"], scales_per_op=False, calib_iter=100
        )
        absorb_to_layer = sq._get_all_layer_names()
        assert absorb_to_layer is not None, "if you are using huggingface model,"
        "you could set torchscript to True when loading the model or set the return_dict to False"
        self.absorb_layers = []
        self.scale_weight_layers = []
        group_modules = sq._trace(op_types, skip_unsupported_layers=False)
        if group_modules:
            for k, v in group_modules.items():
                # use one input for qkv
                for i in v:
                    if i in absorb_to_layer:
                        absorb_to_layer.pop(i)
                        absorb_to_layer[v[0]] = v
            for k, v in absorb_to_layer.items():
                self.absorb_layers.append(k)
                self.scale_weight_layers += v
        else:
            logger.warning("Cannot trace the model, smooth quant may be not used.")
        self._remove_hooks(handles)

    def quantize(self, clean_weight=True):
        """The main entry of layer wise quantization."""
        mk_tmp_dir()
        self._layer_wise_quantize(self.calib_data)
        if self.output_dir:
            self._save(self.output_dir, clean_weight=clean_weight)
        else:
            self._convert(clean_weight=clean_weight)
        del_tmp_dir()
        return self.q_model

    def _layer_wise_quantize(self, calib_data):
        for idx, (name, module) in enumerate(self.modules):
            qconfig = self.op_cfgs.module_name_qconfigs.get(name + ".module")
            # if qconfig:
            if module.__class__.__name__ in ["Linear"]:
                module = QDQLayer(module)
                self.modules[idx] = (name, module)
                update_module(self.q_model, name, module)
                # module.qconfig = self.qconfig
                module.qconfig = qconfig
                self.quantized_layers[name] = -1
        self._regist_hooks()

        self.q_model.eval()
        with torch.no_grad():
            if isinstance(calib_data, torch.Tensor):
                self.q_model(calib_data)
            else:
                try:
                    pbar = tqdm(enumerate(calib_data), total=len(calib_data), desc="layer_wise quant")
                    for idx, input in pbar:
                        forward_wrapper(self.q_model, input, self.device)
                except Exception:  # pragma: no cover
                    pbar = tqdm(enumerate(calib_data), total=len(calib_data), desc="layer_wise quant")
                    for idx, (input, label) in pbar:
                        forward_wrapper(self.q_model, input, self.device)
        self._remove_hooks()

    def _save(self, path=None, clean_weight=True):  # pragma: no cover
        if path is None:
            path = TMP_DIR
        for name, module in self.modules:
            self._load_state_dict(name, TMP_DIR)
            new_module = convert(module, inplace=False)
            torch.save(new_module, os.path.join(path, f"{name}.pt"))
            del new_module
            if clean_weight:
                clean_module_weight(module)
        torch.save(self.fp32_model, os.path.join(path, "model_arch.pt"))

    def _convert(self, clean_weight=False):
        for name, module in self.modules:
            self._load_state_dict(name, TMP_DIR)
            convert(module, inplace=True)
            if clean_weight:
                clean_module_weight(module)

    def _regist_hooks(self):
        def forward_pre_hook(name):
            def load_value(param_name):
                if "lm_head" in param_name and getattr(self.q_model.config, "tie_word_embeddings", True):
                    input_embeddings = self.q_model.get_input_embeddings()
                    for name, module in self.modules:
                        if module == input_embeddings:
                            param_name = name + "." + param_name.split(".")[-1]
                prefix = self.q_model.base_model_prefix
                if "pytorch_model.bin.index.json" in os.listdir(self.path):
                    value = load_tensor_from_shard(self.path, param_name, prefix)
                else:
                    value = load_tensor(os.path.join(self.path, "pytorch_model.bin"), param_name, prefix)
                return value

            def hook(module, input):
                file_path = os.path.join(TMP_DIR, f"{name}.pt")
                if os.path.exists(file_path):
                    self._load_state_dict(name, TMP_DIR)
                else:
                    if isinstance(module, QDQLayer):
                        for n, _ in module.module.named_parameters():
                            value = load_value(name + "." + n)
                            set_module_tensor_to_device(self.q_model, name + ".module." + n, self.device, value)
                        if self.smooth_quant:
                            self._adjust_parameters(module, name, input[0])
                        prepare(module, inplace=True)
                    else:
                        for n, p in module.named_parameters():
                            param_name = name + "." + n
                            value = load_value(param_name)
                            # from hf transformers.modeling_utils._load_state_dict_into_meta_model
                            set_module_tensor_to_device(self.q_model, param_name, self.device, value)

            return hook

        def forward_hook(name):
            def hook(module, input, output):
                file_path = os.path.join(TMP_DIR, f"{name}.pt")
                if os.path.exists(TMP_DIR):
                    torch.save(module.state_dict(), file_path)
                clean_module_weight(module)

            return hook

        for name, module in self.modules:
            self._handle[name] = [module.register_forward_pre_hook(forward_pre_hook(name))]
            self._handle[name] += [module.register_forward_hook(forward_hook(name))]

    def _remove_hooks(self, handles=None):
        if handles is None:
            handles = self._handle
        for handle in handles.values():
            [h.remove() for h in handle]

    def _adjust_parameters(self, module, name, input):
        input = input.reshape(-1, input.shape[-1])
        max_tensor = torch.max(input, dim=0)[0]
        min_tensor = torch.min(input, dim=0)[0]
        input_max = torch.max(torch.abs(max_tensor), torch.abs(min_tensor))

        input_power = torch.pow(input_max, self.alpha)
        weights = module.module.weight
        weight_max_per_channel = torch.max(torch.abs(weights), dim=0)[0]
        weight_power = torch.pow(weight_max_per_channel, 1 - self.alpha)
        scale = torch.clip(input_power / weight_power, min=1e-5)
        scale[input_power == 0] = 1.0

        if name in self.absorb_layers:
            module.input_scale = 1.0 / scale
            self.quantized_layers[name] = 1.0 / scale
        if name in self.scale_weight_layers:
            module.module.weight = torch.nn.Parameter(weights * scale)

    def _load_state_dict(self, module_name, weight_path):
        file_path = os.path.join(weight_path, f"{module_name}.pt")
        state_dict = torch.load(file_path)
        for n, p in state_dict.items():
            set_module_tensor_to_device(self.q_model, f"{module_name}.{n}", self.device, p)
