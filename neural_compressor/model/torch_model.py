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
import sys
from collections import OrderedDict, UserDict
from abc import abstractmethod
from ..adaptor.torch_utils.util import input2tuple
from neural_compressor.utils.utility import LazyImport, compute_sparsity
from neural_compressor.utils import logger
from neural_compressor.conf.dotdict import deep_get, deep_set
from neural_compressor.conf import config as cfg
from neural_compressor.model.base_model import BaseModel

torch = LazyImport('torch')
yaml = LazyImport('yaml')
json = LazyImport('json')
np = LazyImport('numpy')
onnx = LazyImport('onnx')
ort = LazyImport('onnxruntime')
ortq = LazyImport('onnxruntime.quantization')


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
        else:   # pragma: no cover
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
            if self.q_config:
                stat_dict['best_configure'] = self.q_config
            torch.save(stat_dict, os.path.join(root, "best_model.pt"))
            logger.info("Save config file and weights of quantized model to {}.".format(root))
        except IOError as e:   # pragma: no cover
            logger.error("Fail to save configure file and weights due to {}.".format(e))

    def quantized_state_dict(self):
        try:
            stat_dict = self._model.state_dict()
            stat_dict['best_configure'] = self.q_config
        except IOError as e:   # pragma: no cover
            logger.error("Fail to dump configure and weights due to {}.".format(e))
        return stat_dict

    def load_quantized_state_dict(self, stat_dict):
        from ..utils.pytorch import load
        self.q_config = stat_dict['best_configure']
        self._model = load(stat_dict, self._model)

    @property
    def graph_info(self):
        from ..adaptor.pytorch import get_ops_recursively
        op_map = {}
        get_ops_recursively(self._model, '', op_map)
        return op_map

    def export_to_jit(self, example_inputs=None):
        if example_inputs is not None:
            if isinstance(input, dict) or isinstance(input, UserDict):
                example_inputs = tuple(example_inputs.values())
        else:
            logger.warning("Please provide example_inputs for jit.trace")
        try:
            jit_model = torch.jit.trace(
                self._model,
                example_inputs,
            )
        except:
            jit_model = torch.jit.trace(
                self._model,
                example_inputs,
                strict=False
            )
        info = "JIT Model exported"
        logger.info("*"*len(info))
        logger.info(info)
        logger.info("*"*len(info))
        return jit_model

    def export_to_fp32_onnx(
        self,
        save_path='fp32-model.onnx',
        example_inputs=torch.rand([1, 1, 1, 1]),
        opset_version=14,
        dynamic_axes={"input": {0: "batch_size"},
                      "output": {0: "batch_size"}},
        do_constant_folding=True,
        verbose=True,
        fp32_model=None,
    ):
        example_input_names = ['input']
        if isinstance(input, dict) or isinstance(input, UserDict):
            example_input_names = list(input.keys())
        model = self.model
        if fp32_model:
            model = fp32_model
        torch.onnx.export(
            model,
            input2tuple(example_inputs),
            save_path,
            opset_version=opset_version,
            input_names=example_input_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=do_constant_folding,
        )
        if verbose:
            info = "The FP32 ONNX Model exported to path: {0}".format(save_path)
            logger.info("*"*len(info))
            logger.info(info)
            logger.info("*"*len(info))

    def export_to_bf16_onnx(self, 
        save_path='bf16-model.onnx', 
        example_inputs = torch.rand([1, 1, 1, 1]),
        opset_version=14, 
        dynamic_axes={"input": {0: "batch_size"},
                      "output": {0: "batch_size"}},
        do_constant_folding=True,
        verbose=True,
    ):
        fp32_path = save_path + '.tmp' if save_path else 'bf16-model.onnx.tmp'
        self.export_to_fp32_onnx(
            save_path=fp32_path,
            example_inputs = example_inputs,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            do_constant_folding=do_constant_folding,
            verbose=False,
        )
        import onnx
        model = onnx.load(fp32_path)
        bf16_type_list = ['MatMul', 'Gemm', 'Conv', 'Gather']
        bf16_tensor_name_list = []

        for node in model.graph.node:
            if node.op_type in bf16_type_list:
                for inp in node.input:
                    bf16_tensor_name_list.append(inp)

        from onnx import TensorProto, helper, numpy_helper
        original_initializer = copy.deepcopy(model.graph.initializer)
        for tensor in original_initializer:
            if tensor.name in bf16_tensor_name_list:
                bf16_tensor = helper.make_tensor(
                    name=tensor.name,
                    data_type=TensorProto.BFLOAT16,
                    dims=tensor.dims,
                    vals=numpy_helper.to_array(tensor),
                )
                model.graph.initializer.remove(tensor)
                model.graph.initializer.append(bf16_tensor)
        onnx.save(model, save_path)
        os.remove(fp32_path)

        if verbose:
            info = "The BF16 ONNX Model is exported to path: {0}".format(save_path)
            logger.info("*"*len(info))
            logger.info(info)
            logger.info("*"*len(info))

    def export_to_int8_onnx(
        self,
        save_path='int8-model.onnx',
        example_inputs = torch.rand([1, 1, 1, 1]),
        example_input_names = 'input',
        opset_version=14,
        dynamic_axes={"input": {0: "batch_size"},
                    "output": {0: "batch_size"}},
        do_constant_folding=True,
        quant_format='QDQ',
        dtype='S8S8',
        fp32_model=None,
        calib_dataloader=None,
    ): 
        if 'U8U8' in dtype:   # pragma: no cover
            activation_type = ortq.QuantType.QUInt8
            weight_type = ortq.QuantType.QUInt8
        elif 'S8S8' in dtype:
            activation_type = ortq.QuantType.QInt8
            weight_type = ortq.QuantType.QInt8
        elif 'U8S8' in dtype:
            activation_type = ortq.QuantType.QUInt8
            weight_type = ortq.QuantType.QInt8
        else:   # pragma: no cover 
            # Gather requires weight type be the same as activation.
            # So U8S8(acitvation|weight) option is not workable for best performance.
            logger.error("Right now, we don't support dtype: {}, \
                          please use U8U8/U8S8/S8S8.".format(dtype))
            sys.exit(0)
        logger.info("Weight type: {}.".format(weight_type))
        logger.info("Activation type: {}.".format(activation_type))

        assert self.q_config is not None, \
            "No quantization configuration found, " + \
            "please use the model generated by INC quantizer"
        if 'dynamic' in self.q_config['approach']:
            op_types_to_quantize=['MatMul', 'Gather', "LSTM", 'Conv']
            pytorch_op_types_to_quantize=['Linear', 'Embedding', "LSTM", 
                                            'Conv1d', 'Conv2d']
            addition_op_to_quantize = list(ortq.registry.IntegerOpsRegistry.keys())
        else:
            op_types_to_quantize=['MatMul', 'Gather', 'Conv']
            pytorch_op_types_to_quantize=['Linear', 'Embedding', 'Conv1d', 'Conv2d']
            if quant_format == 'QDQ':
                addition_op_to_quantize = list(ortq.registry.QDQRegistry.keys())
                addition_op_to_quantize.remove('Relu') # ValueError: x not in list
            else:
                addition_op_to_quantize = list(ortq.registry.QLinearOpsRegistry.keys())

        if 'U8S8' in dtype:
            op_types_to_quantize.remove('Gather')
            pytorch_op_types_to_quantize.remove('Embedding')

        if quant_format == 'QDQ' and opset_version < 13:   # pragma: no cover 
            opset_version = 13
            logger.warning("QDQ format requires opset_version >= 13, " + 
                            "we reset opset_version={} here".format(opset_version))
        all_op_types_to_quantize = op_types_to_quantize + addition_op_to_quantize

        # pylint: disable=E1101
        fp32_path = save_path + '.tmp' if save_path else 'int8-model.onnx.tmp'
        self.export_to_fp32_onnx(
            save_path=fp32_path,
            example_inputs = example_inputs,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            do_constant_folding=do_constant_folding,
            verbose=False,
            fp32_model=fp32_model
        )
        model = onnx.load(fp32_path)
        from neural_compressor.adaptor.onnxrt import ONNXRTAdaptor
        # pylint: disable=E1120
        inc_model = ONNXRTAdaptor._replace_gemm_with_matmul(model)
        model = inc_model.model
        onnx.save(model, fp32_path)

        # Get weight name from onnx initializer
        weight_name_list = []
        for tensor in model.graph.initializer:
            weight_name_list.append(tensor.name)

        # Match weight name with onnx node name
        quantize_nodes = []
        tmp_node_mapping = {}
        module_node_mapping = {}
        for node in model.graph.node:
            if node.op_type not in op_types_to_quantize:
                for inp in node.input:
                    if inp in weight_name_list and 'weight' in inp:
                        tmp_node_mapping.update({node.output[0] : inp.split('.weight')[0]})
                    elif inp in tmp_node_mapping:
                        tmp_node_mapping.update({node.output[0] : tmp_node_mapping[inp]})
            else:
                for inp in node.input:
                    if inp in weight_name_list and 'weight' in inp:
                        module_node_mapping.update({inp.split('.weight')[0] : node.name})
                    elif inp in tmp_node_mapping:
                        module_node_mapping.update({tmp_node_mapping[inp]: node.name})

            # Save all quantizable node name
            if node.op_type in all_op_types_to_quantize:
                quantize_nodes.append(node.name)

        # Match pytorch module name with onnx node name for fallbacked fp32 module
        for k, v in self.q_config['op'].items():   # pragma: no cover
            if k[1] not in pytorch_op_types_to_quantize or 'int8' in v['weight']['dtype']:
                continue
            k_0 = k[0].split('.module')[0] if k[0] not in module_node_mapping else k[0]
            if k_0 in module_node_mapping:
                fallback_op = module_node_mapping[k_0]
                quantize_nodes.remove(fallback_op)

        # Quantization
        quant_format = ortq.QuantFormat.QOperator if quant_format != 'QDQ' else ortq.QuantFormat.QDQ

        if 'dynamic' in self.q_config['approach']:
            ortq.quantize_dynamic(fp32_path,
                                save_path,
                                per_channel=True,
                                weight_type=weight_type,
                                nodes_to_quantize=quantize_nodes,
                                nodes_to_exclude=[],
                                #op_types_to_quantize=op_types_to_quantize,
                                extra_options={})
        else:
            from ..adaptor.torch_utils.onnx import DataReader
            # pylint: disable=E1101
            assert calib_dataloader is not None, \
                "Please provice the calibration dataloader used in static quantization"
            if not isinstance(calib_dataloader, ortq.CalibrationDataReader):
                sample_size=self.q_config['calib_sampling_size']
                calib_datareader = DataReader(calib_dataloader, sample_size=sample_size)
            ortq.quantize_static(fp32_path,
                                save_path,
                                calib_datareader,
                                quant_format=quant_format,
                                per_channel=True,
                                weight_type=weight_type,
                                activation_type=activation_type,
                                nodes_to_quantize=quantize_nodes,
                                nodes_to_exclude=[],
                                #op_types_to_quantize=op_types_to_quantize,
                                extra_options={})

        os.remove(fp32_path)
        info = "The INT8 ONNX Model is exported to path: {0}".format(save_path)
        logger.info("*"*len(info))
        logger.info(info)
        logger.info("*"*len(info))


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
            self.tune_cfg = json.load(f)

    def save(self, root=None):
        if not root:
            root = cfg.default_workspace
        root = os.path.abspath(os.path.expanduser(root))
        os.makedirs(root, exist_ok=True)
        try:
            with open(os.path.join(root, "best_configure.json"), 'w') as f:
                json.dump(self.tune_cfg, f, indent = 4)
            logger.info("Save config file of quantized model to {}.".format(root))
        except IOError as e:
            logger.error("Fail to save configure file and weights due to {}.".format(e))
