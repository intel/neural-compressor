#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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
import gc
import time
import shutil
from copy import deepcopy
from tqdm import tqdm

from .utils import torch
from torch.quantization import prepare, convert
from accelerate.utils import set_module_tensor_to_device
from .utils import _get_path, get_named_children, update_module, load_tensor_from_shard, load_tensor

from neural_compressor.config import default_workspace

TMP_DIR = f'{default_workspace}/layer_wise_quant_tmp_dir_{time.time()}'


def mk_tmp_dir():
    os.makedirs(TMP_DIR, exist_ok=True)


def del_tmp_dir():
    shutil.rmtree(TMP_DIR)


class QDQLayer(torch.nn.Module):
    def __init__(self, module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quant = torch.ao.quantization.QuantStub()
        self.module = module
        self.dequant = torch.ao.quantization.DeQuantStub()
    
    def forward(self, X):
        X = self.quant(X)
        X = self.module(X)
        X = self.dequant(X)
        return X


class LayerWiseQuant:
    """Layer wise quantization.
    Layer-by-layer quantize the model, in order to save memomery.
    """
    def __init__(self, q_model, pretrained_model_name_or_path, op_cfgs,
                 output_dir=None, device='cpu'):
        """Init LayerWiseQuant."""
        # self.q_model = load_shell(pretrained_model_name_or_path, cls)
        self.q_model = q_model
        self.fp32_model = deepcopy(self.q_model)
        self.path = _get_path(pretrained_model_name_or_path)
        self.op_cfgs = op_cfgs
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        self.modules = get_named_children(self.q_model)
        self.device = device
        self._handle = {}
    
    def quantize(self, calib_data, clean_weight=True):
        """The main entry of layer wise quantization."""
        mk_tmp_dir()
        self._layer_wise_quantize(calib_data)
        if self.output_dir:
            self._save(self.output_dir, clean_weight=clean_weight)
        else:
            self._convert(clean_weight=clean_weight)
        del_tmp_dir()
        return self.q_model

    def _layer_wise_quantize(self, calib_data):
        for idx, (name, module) in enumerate(self.modules):
            qconfig = self.op_cfgs.module_name_qconfigs.get(name + '.module')
            # if qconfig:
            if module.__class__.__name__ in ['Linear']:
                module = QDQLayer(module)
                self.modules[idx] = (name, module)
                update_module(self.q_model, name, module)
                # module.qconfig = self.qconfig
                module.qconfig = qconfig
        self._regist_hooks()

        self.q_model.eval()
        with torch.no_grad():
            if isinstance(calib_data, torch.Tensor):
                self.q_model(calib_data)
            elif isinstance(calib_data, torch.utils.data.dataloader.DataLoader):
                pbar = tqdm(enumerate(calib_data), total=len(calib_data))
                try:
                    for idx, input in pbar:
                        pbar.set_description(f'iter {idx}')
                        self.q_model(**input)
                except Exception:
                    for idx, (input, label) in pbar:
                        self.q_model(**input)
            else:
                self.q_model(**calib_data)
        self._remove_hooks()

    def _save(self, path=None, clean_weight=True):
        if path is None:
            path = TMP_DIR
        for name, module in self.modules:
            self._load_state_dict(name, TMP_DIR)
            new_module = convert(module, inplace=False)
            torch.save(new_module, os.path.join(path, f'{name}.pt'))
            del new_module
            if clean_weight:
                self._clean_weight(module, name)
        torch.save(self.fp32_model, os.path.join(path, 'model_arch.pt'))

    def _convert(self, clean_weight=False):
        for name, module in self.modules:
            self._load_state_dict(name, TMP_DIR)
            convert(module, inplace=True)
            if clean_weight:
                self._clean_weight(module, name)

    def _regist_hooks(self):
        def forward_pre_hook(name):
            def load_value(param_name):
                if 'lm_head' in param_name and getattr(self.q_model.config, "tie_word_embeddings", True):
                    input_embeddings = self.q_model.get_input_embeddings()
                    for name, module in self.modules:
                        if module == input_embeddings:
                            param_name = name + '.' + param_name.split('.')[-1]
                prefix = self.q_model.base_model_prefix
                if 'pytorch_model.bin.index.json' in os.listdir(self.path):
                    value = load_tensor_from_shard(self.path, param_name, prefix)
                else:
                    value = load_tensor(os.path.join(self.path, 'pytorch_model.bin'), param_name, prefix)
                return value

            def hook(module, input):
                file_path = os.path.join(TMP_DIR, f'{name}.pt')
                if os.path.exists(file_path):
                    self._load_state_dict(name, TMP_DIR)
                else:
                    if isinstance(module, QDQLayer):
                        for n, _ in module.module.named_parameters():
                            value = load_value(name + '.' + n)
                            set_module_tensor_to_device(self.q_model, name + '.module.' + n, self.device, value)
                        prepare(module, inplace=True)
                    else:
                        for n, p in module.named_parameters():
                            param_name = name + '.' + n
                            value = load_value(param_name)
                            # from hf transformers.modeling_utils._load_state_dict_into_meta_model
                            set_module_tensor_to_device(self.q_model, param_name, self.device, value)
            return hook
 
        def forward_hook(name):
            def hook(module, input, output):
                file_path = os.path.join(TMP_DIR, f'{name}.pt')
                if os.path.exists(TMP_DIR):
                    torch.save(module.state_dict(), file_path)
                self._clean_weight(module, name)
            return hook

        for name, module in self.modules:
            self._handle[name] = [module.register_forward_pre_hook(forward_pre_hook(name))]
            self._handle[name] += [module.register_forward_hook(forward_hook(name))]

    def _remove_hooks(self):
        for handle in self._handle.values():
            [h.remove() for h in handle]

    def _clean_weight(self, module, name):
        if isinstance(module, QDQLayer):
            submodule = module.module
        else:
            submodule = module
        
        for n, m in submodule.named_parameters():
            is_buffer = n in submodule._buffers
            old_value = getattr(submodule, n)
            with torch.no_grad():
                if is_buffer:
                    submodule._buffers[n] = torch.zeros([0], device="meta")
                else:
                    param_cls = type(submodule._parameters[n])
                    kwargs = submodule._parameters[n].__dict__
                    new_value = torch.zeros([0], device="meta")
                    new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to("meta")
                    submodule._parameters[n] = new_value
        gc.collect()

    
    def _load_state_dict(self,  module_name, weight_path):
        file_path = os.path.join(weight_path, f'{module_name}.pt')
        state_dict = torch.load(file_path)
        for n, p in state_dict.items():
            set_module_tensor_to_device(self.q_model, f'{module_name}.{n}', self.device, p)
