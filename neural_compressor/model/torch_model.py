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

import copy
import os
import inspect
from collections import OrderedDict
from abc import abstractmethod
from neural_compressor.utils.utility import LazyImport, compute_sparsity
from neural_compressor.utils import logger
from neural_compressor.conf.dotdict import deep_get, deep_set
from neural_compressor.conf import config as cfg
from neural_compressor.model.base_model import BaseModel

torch = LazyImport('torch')
yaml = LazyImport('yaml')
json = LazyImport('json')
np = LazyImport('numpy')


class PyTorchBaseModel(torch.nn.Module, BaseModel):
    def __init__(self, model, **kwargs):
        torch.nn.Module.__init__(self)
        self._model = model
        assert isinstance(model, torch.nn.Module), "model should be pytorch nn.Module."
        self.handles = []
        self.tune_cfg= None
        self.q_config = None
        self._workspace_path = ''
        self.is_quantized = False
        self.kwargs = kwargs if kwargs else None

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @property
    def model(self):
        """ Getter to model """
        return self._model

    @model.setter
    def model(self, model):
        """ Setter to model """
        self._model = model

    def register_forward_pre_hook(self):
        self.handles.append(
                self._model.register_forward_pre_hook(self.generate_forward_pre_hook()))

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

    def generate_forward_pre_hook(self):
        # skip input argument 'self' in forward
        self.input_args = OrderedDict().fromkeys(
                inspect.getfullargspec(self._model.forward).args[1:], None)
        # a wrapper is needed to insert self into the actual hook
        def actual_forward_pre_hook(module, input):
            args, _, _, values = inspect.getargvalues(inspect.stack()[1].frame)
            # intersection update kw arguments
            self.input_args.update(values['kwargs'])
            # update arguments
            for (single_input, single_arg) in zip(values['input'],
                    list(self.input_args.keys())[:len(values['input'])]):
                self.input_args[single_arg] = single_input
        return actual_forward_pre_hook

    def framework(self):
        return 'pytorch'

    def get_all_weight_names(self):
        """Get weight names

        Args:

        Returns:
            names (list): list of weight names

        """
        names = []
        for name, param in self._model.named_parameters():
            names.append(name)
        return names

    def get_weight(self, tensor_name):
        """ Get weight value

        Args:
            tensor_name (string): weight name

        Returns:
            (tensor): weight tensor

        """
        state_dict = self._model.state_dict()
        for name, tensor in state_dict.items():
            if tensor_name == name:
                return tensor.cpu()

    def update_weights(self, tensor_name, new_tensor):
        """ Update weight value

        Args:
            tensor_name (string): weight name
            new_tensor (ndarray): weight value

        Returns:

        """
        # TODO: copy tensor option to new tensor is better
        device = next(self._model.parameters()).device 
        new_tensor = torch.tensor(new_tensor).float()
        module_index = '.'.join(tensor_name.split('.')[:-1])
        module = dict(self._model.named_modules())[module_index]
        setattr(module, tensor_name.split('.')[-1], torch.nn.Parameter(new_tensor.to(device)))

    def update_gradient(self, grad_name, new_grad):
        """ Update grad value

        Args:
            grad_name (string): grad name
            new_grad (ndarray): grad value

        Returns:

        """
        device = next(self._model.parameters()).device
        new_grad = torch.tensor(new_grad).float().to(device)
        params = [p for n,p in self._model.named_parameters() if n == grad_name]
        assert len(params) == 1, "lpot can only update grad of one tensor at one time"
        param = params[0]
        param.grad.copy_(new_grad)

    def prune_weights_(self, tensor_name, mask):
        """ Prune weight in place according to tensor_name with mask

        Args:
            tensor_name (string): weight name
            mask (tensor): pruning mask

        Returns:

        """
        state_dict = self._model.state_dict()
        for name in state_dict:
            if name == tensor_name:
                state_dict[name].masked_fill_(mask.to(state_dict[name].device), 0.)

    def get_inputs(self, input_name=None):
        """Get inputs of model

        Args:
            input_name: name of input tensor

        Returns:
            tensor: input tensor
        """
        return self.input_args[input_name].cpu()

    def get_gradient(self, input_tensor):
        """ Get gradients of specific tensor

        Args:
            input_tensor (string or tensor): weight name or a tensor

        Returns:
            (ndarray): gradient tensor array
        """
        if isinstance(input_tensor, str):
            for name, tensor in self._model.named_parameters():
                if name == input_tensor:
                    assert tensor.grad is not None, 'Please call backward() before get_gradient'
                    return np.array(tensor.grad.cpu())
        elif isinstance(input_tensor, torch.Tensor):
            assert input_tensor.grad is not None, 'Please call backward() before get_gradient'
            return np.array(input_tensor.grad.cpu())
        else:
            logger.error("Expect str or torch.Tensor in get_gradient, " \
                         "but get {}.".format(type(input_tensor)))

    def report_sparsity(self):
        """ Get sparsity of the model

        Args:

        Returns:
            df (DataFrame): DataFrame of sparsity of each weight
            total_sparsity (float): total sparsity of model

        """
        import pandas as pd
        df = pd.DataFrame(columns=['Name', 'Shape', 'NNZ (dense)', 'NNZ (sparse)', "Sparsity(%)",
                                   'Std', 'Mean', 'Abs-Mean'])
        pd.set_option('display.precision', 2)
        # TODO: need to specify modules(Conv2d, Linear, etc.) instead of dims
        param_dims = [2, 4]
        params_size = 0
        sparse_params_size = 0
        model_params = dict(self._model.state_dict())
        for name, param in model_params.items():
            # '_packed_params._packed_params' and dtype is specific for quantized module
            if '_packed_params._packed_params' in name and isinstance(param, tuple):
                param = param[0]
            if hasattr(param, 'dtype') and param.dtype in [torch.qint8, torch.quint8]:
                param = param.dequantize()
            if hasattr(param, 'dim') and param.dim() in param_dims \
              and any(type in name for type in ['weight', 'bias', '_packed_params']):
                param_size, sparse_param_size, dense_param_size = compute_sparsity(
                    param.detach().cpu().numpy())
                density = dense_param_size / param_size
                params_size += param_size
                sparse_params_size += sparse_param_size
                df.loc[len(df.index)] = ([
                    name,
                    list(param.shape),
                    dense_param_size,
                    sparse_param_size,
                    (1 - density) * 100,
                    param.std().item(),
                    param.mean().item(),
                    param.abs().mean().item()
                ])

        total_sparsity = sparse_params_size / params_size * 100

        df.loc[len(df.index)] = ([
            'Total sparsity:',
            params_size,
            "-",
            int(sparse_params_size),
            total_sparsity,
            0, 0, 0])
        return df, total_sparsity

class PyTorchModel(PyTorchBaseModel):
    """Build PyTorchModel object

    Args:
        model (pytorch model): model path
    """

    def __init__(self, model, **kwargs):
        super(PyTorchModel, self).__init__(model, **kwargs)

    @property
    def workspace_path(self):
        return self._workspace_path

    @workspace_path.setter
    def workspace_path(self, path):
        from ..adaptor.pytorch import _cfg_to_qconfig, _propagate_qconfig
        workspace_path = path
        weights_file = os.path.join(os.path.abspath(os.path.expanduser(workspace_path)),
                                    'best_model.pt')
        assert os.path.exists(
            weights_file), "weight file %s didn't exist" % weights_file
        self._model = copy.deepcopy(self._model.eval())
        stat_dict = torch.load(weights_file)
        tune_cfg = stat_dict.pop('best_configure')
        op_cfgs = _cfg_to_qconfig(tune_cfg)
        _propagate_qconfig(self._model, op_cfgs)
        # sanity check common API misusage
        if not any(hasattr(m, 'qconfig') and m.qconfig for m in self._model.modules()):
            logger.warn("None of the submodule got qconfig applied. Make sure you "
                        "passed correct configuration through `qconfig_dict` or "
                        "by assigning the `.qconfig` attribute directly on submodules")
        torch.quantization.add_observer_(self._model)
        torch.quantization.convert(self._model, inplace=True)
        self._model.load_state_dict(stat_dict)

    def save(self, root=None):
        if not root:
            root = cfg.default_workspace
        root = os.path.abspath(os.path.expanduser(root))
        os.makedirs(root, exist_ok=True)
        try:
            stat_dict = self._model.state_dict()
            if self.tune_cfg:
                stat_dict['best_configure'] = self.tune_cfg
            torch.save(stat_dict, os.path.join(root, "best_model.pt"))
            logger.info("Save config file and weights of quantized model to {}.".format(root))
        except IOError as e:
            logger.error("Fail to save configure file and weights due to {}.".format(e))

    def quantized_state_dict(self):
        try:
            stat_dict = self._model.state_dict()
            stat_dict['best_configure'] = self.tune_cfg
        except IOError as e:
            logger.error("Fail to dump configure and weights due to {}.".format(e))
        return stat_dict

    def load_quantized_state_dict(self, stat_dict):
        from ..utils.pytorch import load
        self._model = load(stat_dict, self._model)

    @property
    def graph_info(self):
        from ..adaptor.pytorch import get_ops_recursively
        op_map = {}
        get_ops_recursively(self._model, '', op_map)
        return op_map


class PyTorchFXModel(PyTorchModel):
    """Build PyTorchFXModel object

    Args:
        model (onnx model): model path
    """

    def __init__(self, model, **kwargs):
        super(PyTorchFXModel, self).__init__(model, **kwargs)


class PyTorchIpexModel(PyTorchBaseModel):   # pragma: no cover
    """Build PyTorchIpexModel object

    Args:
        model (onnx model): model path
    """

    def __init__(self, model, **kwargs):
        super(PyTorchIpexModel, self).__init__(model, **kwargs)
        self.ipex_config_path = None

    @property
    def graph_info(self):
        ''' return {Node: Node_type} like {'conv0': 'conv2d'} '''
        pass

    @property
    def workspace_path(self):
        return self._workspace_path

    @workspace_path.setter
    def workspace_path(self, path):
        self._workspace_path = path
        tune_cfg_file = os.path.join(os.path.abspath(os.path.expanduser(path)),
                                     'best_configure.json')
        assert os.path.exists(
            tune_cfg_file), "tune configure file %s didn't exist" % tune_cfg_file

        with open(tune_cfg_file, 'r') as f:
            self.tune_cfg= json.load(f)

    def save(self, root=None):
        if not root:
            root = cfg.default_workspace
        root = os.path.abspath(os.path.expanduser(root))
        os.makedirs(root, exist_ok=True)
        try:
            with open(os.path.join(root, "best_configure.json"), 'w') as f:
                json.dump(self.tune_cfg, f)
            logger.info("Save config file of quantized model to {}.".format(root))
        except IOError as e:
            logger.error("Fail to save configure file and weights due to {}.".format(e))
