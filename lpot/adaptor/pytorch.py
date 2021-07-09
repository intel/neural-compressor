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
from lpot.experimental import quantization
import os
from collections import OrderedDict
from distutils.version import StrictVersion
import yaml
from functools import partial
from lpot.utils.utility import dump_elapsed_time
from .adaptor import adaptor_registry, Adaptor
from ..utils.utility import LazyImport, CpuInfo
from ..utils import logger
from .query import QueryBackendCapability

torch = LazyImport('torch')
ipex = LazyImport('intel_pytorch_extension')
json = LazyImport('json')

REDUCE_RANGE = False if CpuInfo().vnni else True
logger.debug("reduce range:")
logger.debug(REDUCE_RANGE)

PT18_VERSION = StrictVersion("1.8")
PT17_VERSION = StrictVersion("1.7")
PT16_VERSION = StrictVersion("1.6")

def get_torch_version():
    try:
        torch_version = torch.__version__.split('+')[0]
    except ValueError as e:
        assert False, 'Got an unknown version of torch: {}'.format(e)
    version = StrictVersion(torch_version)
    return version


def get_ops_recursively(model, prefix, ops={}):
        """This is a helper function for `graph_info`,
           and it will get all ops from model.
        Args:
            model (object): input model
            prefix (string): prefix of op name
            ops (dict): dict of ops from model {op name: type}.
        Returns:
            None
        """
        version = get_torch_version()
        if version < PT17_VERSION:
            white_list = \
               (set(torch.quantization.default_mappings.DEFAULT_MODULE_MAPPING.values()) |
                set(torch.quantization.default_mappings.DEFAULT_QAT_MODULE_MAPPING.values()) |
                set(torch.quantization.default_mappings.DEFAULT_DYNAMIC_MODULE_MAPPING.values()) |
                set(torch.quantization.default_mappings.DEFAULT_MODULE_MAPPING.keys()) |
                set(torch.quantization.default_mappings.DEFAULT_QAT_MODULE_MAPPING.keys()) |
                set(torch.quantization.default_mappings.DEFAULT_DYNAMIC_MODULE_MAPPING.keys()) |
                torch.quantization.default_mappings._INCLUDE_QCONFIG_PROPAGATE_LIST)
        elif version < PT18_VERSION:
            white_list = torch.quantization.get_compare_output_module_list()
        else:
            white_list = torch.quantization.get_default_compare_output_module_list()

        for name, child in model.named_children():
            op_name = prefix + '.' + name if prefix != '' else name
            if type(child) in white_list and not isinstance(child, torch.nn.Sequential) and \
               type(child) != torch.quantization.stubs.DeQuantStub:
                ops[op_name] = unify_op_type_mapping[str(child.__class__.__name__)] \
                    if str(child.__class__.__name__) in unify_op_type_mapping else \
                    str(child.__class__.__name__)
            get_ops_recursively(child, op_name, ops)


def _cfg_to_qconfig(tune_cfg, observer_type='post_training_static_quant'):
    """Convert tune configure to quantization config for each op.

        Args:
            tune_cfg (dict): dictionary of tune configure for each op
            observer_type (str, optional): specify observer type, Default is 'ptq_static',
                                           options: 'ptq_dynamic', 'qat'.

        Returns:
            op_qcfgs (dict): dictionary of quantization configure for each op

        tune_cfg should be a format like below:
        {
          'fuse': {'int8': [['CONV2D', 'RELU', 'BN'], ['CONV2D', 'RELU']],
                   'fp32': [['CONV2D', 'RELU', 'BN']]},
          'calib_iteration': 10,
          'op': {
             ('op1', 'CONV2D'): {
               'activation':  {'dtype': 'uint8',
                               'algorithm': 'minmax',
                               'scheme':'sym',
                               'granularity': 'per_tensor'},
               'weight': {'dtype': 'int8',
                          'algorithm': 'kl',
                          'scheme':'asym',
                          'granularity': 'per_channel'}
             },
             ('op2', 'RELU): {
               'activation': {'dtype': 'int8',
               'scheme': 'asym',
               'granularity': 'per_tensor',
               'algorithm': 'minmax'}
             },
             ('op3', 'CONV2D'): {
               'activation':  {'dtype': 'fp32'},
               'weight': {'dtype': 'fp32'}
             },
             ...
          }
        }
    """
    op_qcfgs = OrderedDict()
    for key in tune_cfg['op']:
        value = tune_cfg['op'][key]
        assert isinstance(value, dict)
        assert 'activation' in value
        if value['activation']['dtype'] == 'fp32':
            if 'weight' in value:
                assert (value['weight']['dtype'] == 'fp32')
            op_qcfgs[key[0]] = None
        else:
            if 'weight' in value:
                weight = value['weight']
                scheme = weight['scheme']
                granularity = weight['granularity']
                algorithm = weight['algorithm']
                dtype = weight['dtype']
                if observer_type == 'quant_aware_training':
                    weights_fake_quantize = _fake_quantize(algorithm, scheme, granularity, dtype)
                else:
                    weights_observer = _observer(algorithm, scheme, granularity, dtype)
            else:
                if observer_type == 'quant_aware_training':
                    weights_fake_quantize = torch.quantization.default_weight_fake_quant
                else:
                    weights_observer = torch.quantization.default_per_channel_weight_observer

            activation = value['activation']
            scheme = activation['scheme']
            granularity = activation['granularity']
            algorithm = activation['algorithm']
            dtype = activation['dtype']
            if observer_type == 'quant_aware_training':
                activation_fake_quantize = _fake_quantize(algorithm, scheme, granularity, dtype)
            else:
                activation_observer = \
                    _observer(algorithm, scheme, granularity, dtype, observer_type)

            if observer_type == 'quant_aware_training':
                qconfig = torch.quantization.QConfig(
                    activation=activation_fake_quantize, weight=weights_fake_quantize)
            elif observer_type == 'post_training_static_quant':
                if key[1] in ['Embedding', 'EmbeddingBag']:     # pragma: no cover
                    qconfig = torch.quantization.QConfigDynamic(
                        activation=activation_observer, weight=weights_observer)
                else:
                    qconfig = torch.quantization.QConfig(
                        activation=activation_observer, weight=weights_observer)
            else:
                version = get_torch_version()
                if version < PT16_VERSION:
                    qconfig = torch.quantization.QConfigDynamic(weight=weights_observer)
                else:
                    qconfig = torch.quantization.QConfigDynamic(
                        activation=activation_observer, weight=weights_observer)

            op_qcfgs[key[0]] = qconfig

    return op_qcfgs


def _cfgs_to_fx_cfgs(op_cfgs, observer_type='post_training_static_quant'):    # pragma: no cover
    """Convert quantization config to a format that meets the requirements of torch.fx.

        Args:
            op_cfgs (dict): dictionary of quantization configure for each op
            observer_type (str, optional): specify observer type, Default is 'ptq_static',
                                           options: 'ptq_dynamic', 'qat'.

        Returns:
            fx_op_cfgs (dict): dictionary of quantization configure that meets
                               the requirements of torch.fx

    example: fx_op_cfgs = {"": default_qconfig,
                           "module_name": [("layer4.1.conv2", per_channel_weight_qconfig)]}
    """
    if observer_type == 'post_training_dynamic_quant':
        model_qconfig = torch.quantization.default_dynamic_qconfig
    elif observer_type == 'quant_aware_training':
        model_qconfig = torch.quantization.QConfig(
                            activation=torch.quantization.FakeQuantize.with_args(
                                    dtype=torch.quint8,
                                    qscheme=torch.per_tensor_affine,
                                    reduce_range=REDUCE_RANGE),
                            weight=torch.quantization.default_weight_fake_quant)
    else:
        model_qconfig = torch.quantization.QConfig(
                            activation=torch.quantization.MinMaxObserver.with_args(
                                    reduce_range=REDUCE_RANGE),
                            weight=torch.quantization.default_per_channel_weight_observer)

    fx_op_cfgs = dict()
    fx_op_cfgs[""] = model_qconfig
    op_tuple_cfg_list = []
    for key, value in op_cfgs.items():
        op_tuple = (key, value)
        op_tuple_cfg_list.append(op_tuple)
    fx_op_cfgs["module_name"] = op_tuple_cfg_list

    return fx_op_cfgs


def _observer(algorithm, scheme, granularity, dtype, observer_type='post_training_static_quant'):
    """Construct an observer module, In forward, observer will update the statistics of
       the observed Tensor. And they should provide a `calculate_qparams` function
       that computes the quantization parameters given the collected statistics.

    Args:
        algorithm (string): What algorithm for computing the quantization parameters based on.
        scheme (string): Quantization scheme to be used.
        granularity (string): What granularity to computing the quantization parameters,
                              per channel or per tensor.
        dtype (string): Quantized data type
        observer_type (string): Observer type, default is 'post_training_static_quant'.

    Returns:
        oberser (object)
    """
    if observer_type == 'post_training_dynamic_quant' and get_torch_version() >= PT16_VERSION:
        return torch.quantization.default_dynamic_quant_observer
    if scheme == 'placeholder':         # pragma: no cover
        return torch.quantization.PlaceholderObserver
    if algorithm == 'minmax':
        if granularity == 'per_channel':
            observer = torch.quantization.PerChannelMinMaxObserver
            if scheme == 'sym':
                qscheme = torch.per_channel_symmetric
            elif scheme == 'asym_float':
                qscheme = torch.per_channel_affine_float_qparams
            else:
                qscheme = torch.per_channel_affine
        else:
            assert granularity == 'per_tensor'
            observer = torch.quantization.MinMaxObserver
            if scheme == 'sym':
                qscheme = torch.per_tensor_symmetric
            else:
                assert scheme == 'asym'
                qscheme = torch.per_tensor_affine
    else:
        assert algorithm == 'kl'
        observer = torch.quantization.HistogramObserver
        assert granularity == 'per_tensor'
        if scheme == 'sym':
            qscheme = torch.per_tensor_symmetric
        else:
            assert scheme == 'asym'
            qscheme = torch.per_tensor_affine

    if dtype == 'int8':
        dtype = torch.qint8
    else:
        assert dtype == 'uint8'
        dtype = torch.quint8

    return observer.with_args(qscheme=qscheme, dtype=dtype,
                              reduce_range=(REDUCE_RANGE and scheme == 'asym'))


def _fake_quantize(algorithm, scheme, granularity, dtype):
    """Construct a fake quantize module, In forward, fake quantize module will update
       the statistics of the observed Tensor and fake quantize the input.
       They should also provide a `calculate_qparams` function
       that computes the quantization parameters given the collected statistics.

    Args:
        algorithm (string): What algorithm for computing the quantization parameters based on.
        scheme (string): Quantization scheme to be used.
        granularity (string): What granularity to computing the quantization parameters,
                              per channel or per tensor.
        dtype (sting): Quantized data type

    Return:
        fake quantization (object)
    """
    fake_quant = torch.quantization.FakeQuantize
    if algorithm == 'minmax':
        if granularity == 'per_channel':
            observer = torch.quantization.MovingAveragePerChannelMinMaxObserver
            if scheme == 'sym':
                qscheme = torch.per_channel_symmetric
            else:
                assert scheme == 'asym'
                qscheme = torch.per_channel_affine
        else:
            assert granularity == 'per_tensor'
            observer = torch.quantization.MovingAverageMinMaxObserver
            if scheme == 'sym':
                qscheme = torch.per_tensor_symmetric
            else:
                assert scheme == 'asym'
                qscheme = torch.per_tensor_affine
    else:
        assert algorithm == 'kl'
        observer = torch.quantization.HistogramObserver
        assert granularity == 'per_tensor'
        if scheme == 'sym':
            qscheme = torch.per_tensor_symmetric
        else:
            assert scheme == 'asym'
            qscheme = torch.per_tensor_affine

    if dtype == 'int8':
        qmin = -128
        qmax = 127
        dtype = torch.qint8
    else:
        assert dtype == 'uint8'
        qmin = 0
        qmax = 255
        dtype = torch.quint8

    return fake_quant.with_args(observer=observer, quant_min=qmin, quant_max=qmax,
                                dtype=dtype, qscheme=qscheme,
                                reduce_range=(REDUCE_RANGE and scheme == 'asym'))


def _propagate_qconfig(model, op_qcfgs, is_qat_convert=False, white_list=None,
                       approach='post_training_static_quant'):
    """Propagate qconfig through the module hierarchy and assign `qconfig`
       attribute on each leaf module

    Args:
        model (object): input model
        op_qcfgs (dict): dictionary that maps from name or type of submodule to
                         quantization configuration, qconfig applies to all submodules of a
                         given module unless qconfig for the submodules are specified (when
                         the submodule already has qconfig attribute)
        is_qat_convert (bool): flag that specified this function is used to QAT prepare
                               for pytorch 1.7 or above.
        white_list (list): list of quantizable op types in pytorch
    Return:
        None, module is modified inplace with qconfig attached
    """
    fallback_ops = []
    version = get_torch_version()
    # there is accuracy issue in quantized LayerNorm op and embedding op in pytorch <1.8.1,
    # so remove it here
    if version < PT17_VERSION and white_list is None:
        white_list = \
            torch.quantization.default_mappings.DEFAULT_DYNAMIC_MODULE_MAPPING \
            if approach == 'post_training_dynamic_quant' else \
            torch.quantization.default_mappings.DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST - \
            {torch.nn.LayerNorm, torch.nn.InstanceNorm3d, torch.nn.Embedding}
    elif version < PT18_VERSION and white_list is None:
        white_list = \
            torch.quantization.quantization_mappings.get_dynamic_quant_module_mappings() \
            if approach == 'post_training_dynamic_quant' else \
            torch.quantization.quantization_mappings.get_qconfig_propagation_list() - \
            {torch.nn.LayerNorm, torch.nn.InstanceNorm3d, torch.nn.Embedding}
    elif white_list is None:
        white_list = \
            torch.quantization.quantization_mappings.get_default_dynamic_quant_module_mappings() \
            if approach == 'post_training_dynamic_quant' else \
            torch.quantization.quantization_mappings.get_default_qconfig_propagation_list() - \
            {torch.nn.LayerNorm, torch.nn.InstanceNorm3d, torch.nn.Embedding}

    _propagate_qconfig_recursively(model, '', op_qcfgs, white_list=white_list)

    if approach != 'post_training_dynamic_quant':
        for k, v in op_qcfgs.items():
            if v is None and not is_qat_convert:
                fallback_ops.append(k)

        if fallback_ops and not is_qat_convert:
            _fallback_quantizable_ops_recursively(model, '', fallback_ops, white_list=white_list)


def _propagate_qconfig_recursively(model, prefix, op_qcfgs, white_list, qconfig_parent=None):
    """This is a helper function for `propagate_qconfig`

    Args:
        model (object): input model
        prefix (string): prefix of op name
        op_qcfgs (dict): dictionary that maps from name or type of submodule to
                        quantization configuration
        white_list (list): list of quantizable op types in pytorch
        qconfig_parent (object, optional): qconfig of parent module

    Returns:
        None
    """
    for name, child in model.named_children():
        model_qconfig = qconfig_parent
        op_name = prefix + name
        if op_name in op_qcfgs:
            child.qconfig = op_qcfgs[op_name]
            model_qconfig = op_qcfgs[op_name]
        elif type(child) in white_list and type(child) != torch.nn.Sequential:
            if model_qconfig is None:
                model_qconfig = torch.quantization.QConfig(
                        activation=torch.quantization.MinMaxObserver.with_args(
                                reduce_range=REDUCE_RANGE),
                        weight=torch.quantization.default_per_channel_weight_observer)
            child.qconfig = model_qconfig
        _propagate_qconfig_recursively(
            child, op_name + '.', op_qcfgs, white_list, model_qconfig)


def _find_quantized_op_num(model, white_list, op_count=0):
    """This is a helper function for `_fallback_quantizable_ops_recursively`

    Args:
        model (object): input model
        white_list (list): list of quantizable op types in pytorch
        op_count (int, optional): count the quantizable op quantity in this module

    Returns:
        the quantizable op quantity in this module
    """
    quantize_op_num = op_count
    for name_tmp, child_tmp in model.named_children():
        if type(child_tmp) in white_list \
            and not (isinstance(child_tmp, torch.quantization.QuantStub)
                     or isinstance(child_tmp, torch.quantization.DeQuantStub)):
            quantize_op_num += 1
        else:
            quantize_op_num = _find_quantized_op_num(
                child_tmp, white_list, quantize_op_num)
    return quantize_op_num


def _fallback_quantizable_ops_recursively(model, prefix, fallback_ops, white_list=None):
    """Handle all fallback ops(fp32 ops)

    Args:
        model (object): input model
        prefix (string): the prefix of op name
        fallback_ops (list): list of fallback ops(fp32 ops)
        white_list (list): list of quantizable op types in pytorch

    Returns:
        None
    """
    class DequantQuantWrapper(torch.nn.Module):
        """A wrapper class that wraps the input module, adds DeQuantStub and
           surround the call to module with call to dequant.
           this is used by fallback layer when the data type of quantized op
           is  input:int8/output:int8.

           This is used by the fallback utility functions to add the dequant and
           quant modules, before `convert` function `QuantStub` will just be observer,
           it observes the input tensor, after `convert`, `QuantStub`
           will be swapped to `nnq.Quantize` which does actual quantization. Similarly
           for `DeQuantStub`.
        """

        def __init__(self, module, observer=None):
            super(DequantQuantWrapper, self).__init__()
            if not module.qconfig and observer:
                weights_observer = observer('minmax', 'asym', 'per_channel', 'int8')
                activation_observer = observer('minmax', 'sym', 'per_tensor', 'uint8')
                module.qconfig = torch.quantization.QConfig(
                    activation=activation_observer, weight=weights_observer)
            self.add_module('quant', torch.quantization.QuantStub(module.qconfig))
            self.add_module('dequant', torch.quantization.DeQuantStub())
            self.add_module('module', module)
            version = get_torch_version()
            if version >= PT18_VERSION:
                self.dequant.qconfig = module.qconfig
            module.qconfig = None
            self.train(module.training)

        def forward(self, X):
            X = self.dequant(X)
            X = self.module(X)
            return self.quant(X)

        def add(self, x, y):
            # type: (Tensor, Tensor) -> Tensor
            x = self.dequant(x)
            y = self.dequant(y)
            r = self.module.add(x, y)
            return self.quant(r)

        def add_scalar(self, x, y):
            # type: (Tensor, float) -> Tensor
            x = self.dequant(x)
            r = self.module.add_scalar(x, y)
            return self.quant(r)

        def mul(self, x, y):
            # type: (Tensor, Tensor) -> Tensor
            x = self.dequant(x)
            y = self.dequant(y)
            r = self.module.mul(x, y)
            return self.quant(r)

        def mul_scalar(self, x, y):
            # type: (Tensor, float) -> Tensor
            x = self.dequant(x)
            r = self.module.mul_scalar(x, y)
            return self.quant(r)

        def cat(self, x, dim=0):
            # type: (List[Tensor], int) -> Tensor
            X = [self.dequant(x_) for x_ in x]
            r = self.module.cat(X, dim)
            return self.quant(r)

        def add_relu(self, x, y):
            # type: (Tensor, Tensor) -> Tensor
            x = self.dequant(x)
            y = self.dequant(y)
            r = self.module.add_relu(x, y)
            return self.quant(r)

    for name, child in model.named_children():
        op_name = prefix + name
        if op_name in fallback_ops:
            child.qconfig = None
            quantize_op_num = _find_quantized_op_num(model, white_list=white_list)
            if quantize_op_num == 1:
                found = False
                for name_tmp, child_tmp in model.named_children():
                    if isinstance(
                            child_tmp, torch.quantization.QuantStub) or isinstance(
                            child_tmp, torch.quantization.DeQuantStub):
                        model._modules[name_tmp] = torch.nn.Identity()
                        found = True
                if not found:
                    model._modules[name] = DequantQuantWrapper(
                        child, observer=_observer)
            else:
                model._modules[name] = DequantQuantWrapper(
                    child, observer=_observer)
        else:
            _fallback_quantizable_ops_recursively(
                child, op_name + '.', fallback_ops, white_list=white_list)


@adaptor_registry
class TemplateAdaptor(Adaptor):
    """Tample adaptor of PyTorch framework.

    Args:
        framework_specific_info (dict): dictionary of tuning configure from yaml file.
    """
    def __init__(self, framework_specific_info):
        super(TemplateAdaptor, self).__init__(framework_specific_info)
        import torch.quantization as tq

        self.version = get_torch_version()

        # set torch random seed
        random_seed = framework_specific_info['random_seed']
        torch.manual_seed(random_seed)

        self.device = framework_specific_info['device']
        self.q_dataloader = framework_specific_info['q_dataloader']
        self.benchmark = framework_specific_info['benchmark'] \
            if 'benchmark' in framework_specific_info else False
        self.workspace_path = framework_specific_info['workspace_path']
        self.is_baseline = True if not self.benchmark else False
        self.query_handler = None
        self.approach = ''
        self.pre_optimized_model = None

        if 'approach' in framework_specific_info:
            self.approach = framework_specific_info['approach']
            if framework_specific_info['approach'] == "post_training_static_quant":
                if self.version < PT17_VERSION:
                    self.q_mapping = tq.default_mappings.DEFAULT_MODULE_MAPPING
                elif self.version < PT18_VERSION:
                    self.q_mapping = \
                        tq.quantization_mappings.get_static_quant_module_mappings()
                else:
                    self.q_mapping = \
                        tq.quantization_mappings.get_default_static_quant_module_mappings()
            elif framework_specific_info['approach'] == "quant_aware_training":
                if self.version < PT17_VERSION:
                    self.q_mapping = tq.default_mappings.DEFAULT_QAT_MODULE_MAPPING
                elif self.version < PT18_VERSION:
                    self.q_mapping = \
                        tq.quantization_mappings.get_qat_module_mappings()
                else:
                    self.q_mapping = \
                        tq.quantization_mappings.get_default_qat_module_mappings()
            elif framework_specific_info['approach'] == "post_training_dynamic_quant":
                if self.version < PT17_VERSION:
                    self.q_mapping = \
                        tq.default_mappings.DEFAULT_DYNAMIC_MODULE_MAPPING
                elif self.version < PT18_VERSION:
                    self.q_mapping = \
                        tq.quantization_mappings.get_dynamic_quant_module_mappings()
                else:
                    self.q_mapping = \
                        tq.quantization_mappings.get_default_dynamic_quant_module_mappings()
            else:
                assert False, "Unsupport approach: {}".format(self.approach)

        self.fp32_results = []
        self.fp32_preds_as_label = False


    def _get_quantizable_ops_recursively(self, model, prefix, quantizable_ops):
        """This is a helper function for `query_fw_capability`,
           and it will get all quantizable ops from model.

        Args:
            model (object): input model
            prefix (string): prefix of op name
            quantizable_ops (list): list of quantizable ops from model include op name and type.

        Returns:
            None
        """

        raise NotImplementedError

    @dump_elapsed_time("Pass query framework capability")
    def query_fw_capability(self, model):
        """This is a helper function to get all quantizable ops from model.

        Args:
            model (object): input model which is LPOT model

        Returns:
            q_capability (dictionary): tuning capability for each op from model.
        """
        self.pre_optimized_model = model
        quantizable_ops = []
        self._get_quantizable_ops_recursively(model.model, '', quantizable_ops)
        capability = self.query_handler.get_quantization_capability()['dynamic'] \
            if self.approach == "post_training_dynamic_quant" else \
            self.query_handler.get_quantization_capability()['int8']

        q_capability = {}
        q_capability['optypewise'] = OrderedDict()
        q_capability['opwise'] = OrderedDict()

        for q_op in quantizable_ops:
            q_capability['opwise'][q_op] = copy.deepcopy(capability[q_op[1]]) \
                if q_op[1] in capability.keys() else copy.deepcopy(capability['default'])
            if q_op[1] not in q_capability['optypewise'].keys():
                q_capability['optypewise'][q_op[1]] = copy.deepcopy(capability[q_op[1]]) \
                    if q_op[1] in capability.keys() else copy.deepcopy(capability['default'])

        return q_capability


unify_op_type_mapping = {
    "ConvReLU2d": "Conv2d",
    "ConvReLU3d": "Conv3d",
    "LinearReLU": "Linear",
    "ConvBn2d": "Conv2d",
    "ConvBnReLU2d": "Conv2d"
}


@adaptor_registry
class PyTorchAdaptor(TemplateAdaptor):
    """Adaptor of PyTorch framework, all PyTorch API is in this class.

    Args:
        framework_specific_info (dict): dictionary of tuning configure from yaml file.
    """
    def __init__(self, framework_specific_info):
        super(PyTorchAdaptor, self).__init__(framework_specific_info)
        import torch.quantization as tq
        """
        # Map for swapping float module to quantized ones,
        # and this dictionary will change with different PoTorch versions
        DEFAULT_MODULE_MAPPING = {
            nn.Linear: nnq.Linear,
            nn.ReLU: nnq.ReLU,
            nn.ReLU6: nnq.ReLU6,
            nn.Conv2d: nnq.Conv2d,
            nn.Conv3d: nnq.Conv3d,
            QuantStub: nnq.Quantize,
            DeQuantStub: nnq.DeQuantize,
            # Wrapper Modules:
            nnq.FloatFunctional: nnq.QFunctional,
            # Intrinsic modules:
            nni.ConvReLU2d: nniq.ConvReLU2d,
            nni.ConvReLU3d: nniq.ConvReLU3d,
            nni.LinearReLU: nniq.LinearReLU,
            nniqat.ConvReLU2d: nniq.ConvReLU2d,
            nniqat.LinearReLU: nniq.LinearReLU,
            nniqat.ConvBn2d: nnq.Conv2d,
            nniqat.ConvBnReLU2d: nniq.ConvReLU2d,
            # QAT modules:
            nnqat.Linear: nnq.Linear,
            nnqat.Conv2d: nnq.Conv2d,
        }
        """

        self.tune_cfg = None
        if self.device == "cpu":
            query_config_file = "pytorch_cpu.yaml"
        elif self.device == "gpu":
            query_config_file = "pytorch_gpu.yaml"
        else:
            assert False, "Unsupport this device {}".format(self.device)
        self.query_handler = PyTorchQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), query_config_file))

        # there is accuracy issue in quantized LayerNorm op and embedding op in pytorch <1.8.1,
        # so remove it here
        if self.version < PT17_VERSION:
            self.white_list = \
                tq.default_mappings.DEFAULT_DYNAMIC_MODULE_MAPPING \
                if self.approach == 'post_training_dynamic_quant' else \
                tq.default_mappings.DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST - \
                    {torch.nn.LayerNorm, torch.nn.InstanceNorm3d, torch.nn.Embedding}
        elif self.version < PT18_VERSION:
            self.white_list = \
                tq.quantization_mappings.get_dynamic_quant_module_mappings() \
                if self.approach == 'post_training_dynamic_quant' else \
                tq.quantization_mappings.get_qconfig_propagation_list() - \
                    {torch.nn.LayerNorm, torch.nn.InstanceNorm3d, torch.nn.Embedding}
        else:
            self.white_list = \
                tq.quantization_mappings.get_default_dynamic_quant_module_mappings() \
                if self.approach == 'post_training_dynamic_quant' else \
                tq.quantization_mappings.get_default_qconfig_propagation_list() - \
                    {torch.nn.LayerNorm, torch.nn.InstanceNorm3d, torch.nn.Embedding}

        # for tensorboard
        self.dump_times = 0
        self.fused_op = ['nni.ConvReLU1d',
                         'nni.ConvReLU2d',
                         'nni.ConvReLU3d',
                         'nni.LinearReLU',
                         'nni.BNReLU2d',
                         'nni.BNReLU3d',
                         'nniqat.ConvReLU2d',
                         'nniqat.ConvBn2d',
                         'nniqat.ConvBnReLU2d',
                         'nni.LinearReLU']
        self.fused_dict = {}

    def model_calibration(self, q_model, dataloader, iterations=1):
        assert iterations > 0
        with torch.no_grad():
            for idx, (input, label) in enumerate(dataloader):
                if isinstance(input, dict):
                    if self.device == "gpu":
                        for inp in input.keys():
                            input[inp] = input[inp].to("dpcpp") \
                                if isinstance(input[inp], torch.Tensor) else input[inp]
                    output = q_model(**input)
                elif isinstance(input, list) or isinstance(input, tuple):
                    if self.device == "gpu":
                        input = [inp.to("dpcpp")
                                 if isinstance(inp, torch.Tensor) else inp
                                 for inp in input]
                    output = q_model(*input)
                else:
                    if self.device == "gpu" and isinstance(input, torch.Tensor):
                        input = input.to("dpcpp")
                    output = q_model(input)
                if idx >= iterations - 1:
                    break

    @dump_elapsed_time("Pass quantize model")
    def quantize(self, tune_cfg, model, dataloader, q_func=None):
        """Execute the quantize process on the specified model.

        Args:
            tune_cfg (dict): quantization config.
            model (object): model need to do quantization.
            dataloader (object): calibration dataset.
            q_func (objext, optional): training function for quantization aware training mode.

        Returns:
            (object): quantized model
        """

        assert isinstance(model.model, torch.nn.Module), \
               "The model passed in is not the instance of torch.nn.Module"

        # For tensorboard display
        self.tune_cfg = tune_cfg
        self.tune_cfg["approach"] = self.approach
        self.tune_cfg["framework"] = "pytorch"
        op_cfgs = _cfg_to_qconfig(tune_cfg, self.approach)

        try:
            q_model = copy.deepcopy(model)
        except Exception as e:                              # pragma: no cover
            logger.warning("Deepcopy failed: {}, inplace=True now!".format(repr(e)))
            q_model = model
        if self.approach == 'quant_aware_training':
            q_model.model.train()
        else:
            q_model.model.eval()
        if self.version < PT17_VERSION or self.approach != 'quant_aware_training':
            _propagate_qconfig(q_model.model, op_cfgs, white_list=self.white_list,
                               approach=self.approach)
            # sanity check common API misusage
            if not any(hasattr(m, 'qconfig') and m.qconfig for m in q_model.model.modules()):
                logger.warn("None of the submodule got qconfig applied. Make sure you "
                            "passed correct configuration through `qconfig_dict` or "
                            "by assigning the `.qconfig` attribute directly on submodules")

        if self.approach == 'post_training_static_quant':
            torch.quantization.add_observer_(q_model.model)
            iterations = tune_cfg.get('calib_iteration', 1)
            self.model_calibration(q_model.model, dataloader, iterations)
        elif self.approach == 'quant_aware_training':
            if self.version >= PT17_VERSION:       # pragma: no cover
                _propagate_qconfig(q_model.model, op_cfgs, is_qat_convert=True,
                                   white_list=self.white_list)
                torch.quantization.convert(q_model.model, mapping=self.q_mapping,
                                           inplace=True, remove_qconfig=False)
                _propagate_qconfig(q_model.model, op_cfgs, white_list=self.white_list)
                torch.quantization.add_observer_(q_model.model, self.white_list,
                                                 set(self.q_mapping.values()))
            else:
                torch.quantization.add_observer_(q_model.model)
                torch.quantization.convert(q_model.model, self.q_mapping, inplace=True)
            if q_func is None:
                assert False, "quantization aware training mode requires q_function to train"
            else:
                q_func(q_model.model)
            q_model.model.eval()

        if self.approach == 'quant_aware_training':
            torch.quantization.convert(q_model.model, inplace=True)
        else:
            torch.quantization.convert(q_model.model, mapping=self.q_mapping, inplace=True)
        q_model.tune_cfg = copy.deepcopy(self.tune_cfg)
        q_model.is_quantized = True

        if self.is_baseline:
            self.is_baseline = False

        return q_model

    def evaluate(self, model, dataloader, postprocess=None,
                 metric=None, measurer=None, iteration=-1,
                 tensorboard=False, fp32_baseline=False):
        """Execute the evaluate process on the specified model.

        Args:
            model (object): model to run evaluation.
            dataloader (object): evaluation dataset.
            postprocess (object, optional): process function after evaluation.
            metric (object, optional): metric function.
            measurer (object, optional): measurer function.
            iteration (int, optional): number of iterations to evaluate.
            tensorboard (bool, optional): dump output tensor to tensorboard summary files.
            fp32_baseline (boolen, optional): only for compare_label=False pipeline

        Returns:
            (object): accuracy
        """
        if tensorboard:
            model = self._pre_eval_hook(model)

        model_ = model.model
        assert isinstance(
            model_, torch.nn.Module), "The model passed in is not the instance of torch.nn.Module"
        model_.eval()
        if self.device == "cpu":
            model_.to("cpu")
        elif self.device == "gpu":
            if self.is_baseline:
                model_.to("dpcpp")

        if metric and hasattr(metric, "compare_label") and not metric.compare_label:
            self.fp32_preds_as_label = True
            results = []

        with torch.no_grad():
            if metric:
                metric.reset()
            for idx, (input, label) in enumerate(dataloader):
                if measurer is not None:
                    measurer.start()

                if isinstance(input, dict):
                    if self.device == "gpu":
                        for inp in input.keys():
                            input[inp] = input[inp].to("dpcpp") \
                                if isinstance(input[inp], torch.Tensor) else input[inp]
                    output = model_(**input)
                elif isinstance(input, list) or isinstance(input, tuple):
                    if self.device == "gpu":
                        input = [inp.to("dpcpp")
                                 if isinstance(inp, torch.Tensor) else inp for inp in input]
                    output = model_(*input)
                else:
                    if self.device == "gpu" and isinstance(input, torch.Tensor):
                        input = input.to("dpcpp")
                    output = model_(input)
                if self.device == "gpu":
                    output = output.to("cpu")
                if measurer is not None:
                    measurer.end()
                if postprocess is not None:
                    output, label = postprocess((output, label))
                if metric is not None and not self.fp32_preds_as_label:
                    metric.update(output, label)
                if self.fp32_preds_as_label:
                    self.fp32_results.append(output) if fp32_baseline else \
                        results.append(output)
                if idx + 1 == iteration:
                    break

        if self.fp32_preds_as_label:
            from .torch_utils.util import collate_torch_preds
            if fp32_baseline:
                results = collate_torch_preds(self.fp32_results)
                metric.update(results, results)
            else:
                reference = collate_torch_preds(self.fp32_results)
                results = collate_torch_preds(results)
                metric.update(results, reference)

        acc = metric.result() if metric is not None else 0

        if tensorboard:
            self._post_eval_hook(model, accuracy=acc)
        return acc

    def train(self, model, dataloader, optimizer_tuple, criterion_tuple, hooks, **kwargs):
        """Execute the train process on the specified model.

        Args:
            model (object): model to run evaluation.
            dataloader (object): training dataset.
            optimizer (tuple): It is a tuple of (cls, parameters) for optimizer.
            criterion (tuple): It is a tuple of (cls, parameters) for criterion.
            kwargs (dict, optional): other parameters.

        Returns:
            None
        """
        model_ = model.model
        optimizer = optimizer_tuple[0](model_.parameters(), **optimizer_tuple[1])
        criterion = criterion_tuple[0](**criterion_tuple[1])
        start_epochs = kwargs['kwargs']['start_epoch']
        end_epochs = kwargs['kwargs']['end_epoch']
        iters = kwargs['kwargs']['iteration']
        if hooks is not None:
            on_epoch_start = hooks['on_epoch_start']
            on_epoch_end = hooks['on_epoch_end']
            on_batch_start = hooks['on_batch_start']
            on_batch_end = hooks['on_batch_end']
            on_post_grad = hooks['on_post_grad']
        for nepoch in range(start_epochs, end_epochs):
            model_.train()
            cnt = 0
            if hooks is not None:
                on_epoch_start(nepoch)
            for image, target in dataloader:
                if hooks is not None:
                    on_batch_start(cnt)
                print('.', end='', flush=True)
                cnt += 1
                output = model_(image)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                if hooks is not None:
                    on_post_grad()
                optimizer.step()
                if hooks is not None:
                    on_batch_end()
                if cnt >= iters:
                    break
            if hooks is not None:
                on_epoch_end()

    def is_fused_module(self, module):
        """This is a helper function for `_propagate_qconfig_helper` to detecte
           if this module is fused.

        Args:
            module (object): input module

        Returns:
            (bool): is fused or not
        """
        op_type = str(type(module))
        op_type = op_type[op_type.rfind('.')+1:].strip('>').strip('\'')
        op_type = 'nni.' + op_type
        if op_type in self.fused_op:
            return True
        else:
            return False

    def _get_quantizable_ops_recursively(self, model, prefix, quantizable_ops):
        """This is a helper function for `query_fw_capability`,
           and it will get all quantizable ops from model.

        Args:
            model (object): input model
            prefix (string): prefix of op name
            quantizable_ops (list): list of quantizable ops from model include op name and type.

        Returns:
            None
        """

        for name, child in model.named_children():
            op_name = prefix + '.' + name if prefix != '' else name
            # there is accuracy issue in quantized LayerNorm op in pytorch <1.8.1,
            # so remove it here
            if type(child) in self.white_list and type(child) != torch.nn.Sequential and \
                    type(child) != torch.quantization.stubs.DeQuantStub and not \
                        isinstance(child, torch.nn.LayerNorm) and not \
                        isinstance(child, torch.nn.InstanceNorm3d) and not \
                        isinstance(child, torch.nn.Embedding):
                quantizable_ops.append((
                    op_name, unify_op_type_mapping[str(child.__class__.__name__)]
                    if str(child.__class__.__name__) in unify_op_type_mapping else
                    str(child.__class__.__name__)))
                if self.is_fused_module(child):
                    for name, _ in child.named_children():
                        module_prefix = op_name + '.' + name
                        if op_name in self.fused_dict:
                            self.fused_dict[op_name] = [self.fused_dict[op_name], module_prefix]
                        else:
                            self.fused_dict[op_name] = module_prefix
            else:
                self._get_quantizable_ops_recursively(child, op_name, quantizable_ops)

    def _pre_eval_hook(self, model, op_list=None, iteration_list=None):
        """The function is used to do some preprocession before evaluation phase.
           Here, it used to add hook for dump output tensor for quantizable ops.

        Args:
             model (object): input model

        Returns:
              model (object): model with hook
        """
        from abc import ABCMeta

        def _with_args(cls_or_self, **kwargs):
            r"""Wrapper that allows creation of class factories.

            This can be useful when there is a need to create classes with the same
            constructor arguments, but different instances.

            Example::

                >>> Foo.with_args = classmethod(_with_args)
                >>> foo_builder = Foo.with_args(a=3, b=4).with_args(answer=42)
                >>> foo_instance1 = foo_builder()
                >>> foo_instance2 = foo_builder()
                >>> id(foo_instance1) == id(foo_instance2)
                False
            """
            class _PartialWrapper(object):
                def __init__(self, p):
                    self.p = p

                def __call__(self, *args, **keywords):
                    return self.p(*args, **keywords)

                def __repr__(self):
                    return self.p.__repr__()

                with_args = _with_args
            r = _PartialWrapper(partial(cls_or_self, **kwargs))
            return r

        ABC = ABCMeta(str("ABC"), (object, ),
                      {})  # compatible with Python 2 *and* 3:

        class _RecordingObserver(ABC, torch.nn.Module):
            """The module is mainly for debug and records the tensor values during runtime.

            Args:
                iteration_list (list, optional): indexs of iteration which to dump tensor.
            """

            def __init__(self, iteration_list=None, **kwargs):
                super(_RecordingObserver, self).__init__(**kwargs)
                self.output_tensors_dict = OrderedDict()
                self.current_iter = 1
                self.iteration_list = iteration_list

            def forward(self, x):
                if (self.iteration_list is None and self.current_iter == 1) or \
                    (self.iteration_list is not None and
                     self.current_iter in self.iteration_list):
                    if type(x) is tuple or type(x) is list:
                        self.output_tensors_dict[self.current_iter] = \
                            [i.to("cpu") if i.device != 'cpu' else i.clone() for i in x]
                    else:
                        self.output_tensors_dict[self.current_iter] = \
                            x.to("cpu") if x.device != "cpu" else x.clone()
                self.current_iter += 1
                return x

            @torch.jit.export
            def get_tensor_value(self):
                return self.output_tensors_dict

            with_args = classmethod(_with_args)

        def _observer_forward_hook(module, input, output):
            """Forward hook that calls observer on the output

            Args:
                module (object): input module
                input (object): module input
                output (object): module output

            Returns:
                module output tensor (object)
            """
            return module.activation_post_process(output)

        def _add_observer_(module, op_list=None, prefix=""):
            """Add observer for the leaf child of the module.

               This function insert observer module to all leaf child module that
               has a valid qconfig attribute.

            Args:
                module (object): input module with qconfig attributes for all the leaf modules that
                                 we want to dump tensor
                op_list (list, optional): list of ops which to be dumped in module
                prefix (string): name of module

            Returns:
                None, module is modified inplace with added observer modules and forward_hooks
            """
            for name, child in module.named_children():
                op_name = name if prefix == "" else prefix + "." + name
                if isinstance(child, torch.nn.quantized.FloatFunctional) and \
                             (op_list is None or op_name in op_list):
                    if hasattr(child, 'qconfig') and child.qconfig is not None and (
                            op_list is None or op_name in op_list):
                        child.activation_post_process = \
                            child.qconfig.activation()
                elif hasattr(child, 'qconfig') and child.qconfig is not None and \
                        (op_list is None or op_name in op_list):
                    # observer and hook will be gone after we swap the module
                    child.add_module(
                        'activation_post_process',
                        child.qconfig.activation())
                    child.register_forward_hook(_observer_forward_hook)
                else:
                    _add_observer_(child, op_list, op_name)

        def _propagate_qconfig_helper(module,
                                      qconfig_dict,
                                      white_list=None,
                                      qconfig_parent=None,
                                      prefix='',
                                      fused=False):
            """This is a helper function for `propagate_qconfig_`

            Args:
                module (object): input module
                qconfig_dict (dictionary): dictionary that maps from name of submodule to
                                           quantization configuration
                white_list (list, optional): list of quantizable modules
                qconfig_parent (object, optional): config of parent module, we will fallback to
                                                   this config when there is no specified config
                                                   for current module
                prefix (string, optional): corresponding prefix of the current module,
                                           used as key in qconfig_dict
                fused (bool, optional): Indicates whether the module is fused or not

            Return:
                None, module is modified inplace with qconfig attached
            """
            # TODO: Add test
            if white_list is None:
                white_list = \
                   torch.quantization.default_mappings.DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST \
                   if self.version < PT17_VERSION else \
                   torch.quantization.quantization_mappings.get_qconfig_propagation_list()

            if type(module) in white_list and type(module) != torch.nn.Sequential:
                module.qconfig = qconfig_parent
            else:
                module.qconfig = None
            if hasattr(module, '_modules'):
                for name, child in module.named_children():
                    module_prefix = prefix + '.' + name if prefix else name
                    _propagate_qconfig_helper(child, qconfig_dict, white_list,
                                              qconfig_parent, module_prefix)

        def _prepare(model, inplace=True, op_list=[], white_list=None):
            """The model will be attached with observer or fake quant modules, and qconfig
               will be propagated.

            Args:
                model (object): input model to be modified in-place
                inplace (bool, optional): carry out model transformations in-place,
                                          the original module is mutated
                op_list (list, optional): list of ops which to be dumped in module
                white_list (list, optional): list of quantizable modules

            Returns:
                model (object): model with qconfig
            """
            if not inplace:
                model = copy.deepcopy(model)
            _propagate_qconfig_helper(model,
                                      qconfig_dict={},
                                      white_list=white_list,
                                      qconfig_parent=model.qconfig)
            # sanity check common API misusage
            if not any(
                    hasattr(m, 'qconfig') and m.qconfig
                    for m in model.modules()):
                logger.warn(
                    "None of the submodule got qconfig applied. Make sure you "
                    "passed correct configuration through `qconfig_dict` or "
                    "by assigning the `.qconfig` attribute directly on submodules"
                )
            _add_observer_(model, op_list=op_list)
            return model

        # create properties
        if self.version < PT17_VERSION:
            white_list = self.white_list | \
                (set(torch.quantization.default_mappings.DEFAULT_MODULE_MAPPING.values()) |
                 set(torch.quantization.default_mappings.DEFAULT_QAT_MODULE_MAPPING.values()) |
                 set(torch.quantization.default_mappings.DEFAULT_DYNAMIC_MODULE_MAPPING.values()))
        elif self.version < PT18_VERSION:
            white_list = torch.quantization.get_compare_output_module_list()
        else:
            white_list = torch.quantization.get_default_compare_output_module_list()

        model = copy.deepcopy(model) if self.is_baseline else model
        model.model.qconfig = torch.quantization.QConfig(
            weight=torch.quantization.default_debug_observer,
            activation=_RecordingObserver.with_args(iteration_list=iteration_list))
        _prepare(model.model, op_list=op_list, white_list=white_list)

        return model

    def is_fused_child(self, op_name):
        """This is a helper function for `_post_eval_hook`

        Args:
            op_name (string): op name

        Returns:
            (bool): if this op is fused

        """
        op = op_name[:op_name.rfind('.')]
        if op in self.fused_dict and op_name[op_name.rfind('.')+1:].isdigit():
            return True
        else:
            return False

    def is_fused_op(self, op_name):
        """This is a helper function for `_post_eval_hook`

        Args:
            op_name (string): op name

        Returns:
            (bool): if this op is fused

        """
        op = op_name[:op_name.rfind('.')]
        if op in self.fused_dict:
            return True
        else:
            return False

    def is_last_fused_child(self, op_name):
        """This is a helper function for `_post_eval_hook`

        Args:
            op_name (string): op name

        Returns:
            (bool): if this op is last fused op

        """
        op = op_name[:op_name.rfind('.')]
        if op_name in self.fused_dict[op][-1]:
            return True
        else:
            return False

    def _post_eval_hook(self, model, **args):
        """The function is used to do some post process after complete evaluation.
           Here, it used to dump quantizable op's output tensor.

        Args:
            model (object): input model

        Returns:
            None
        """
        from torch.utils.tensorboard import SummaryWriter
        from torch.quantization import get_observer_dict

        model = model.model

        if args is not None and 'accuracy' in args:
            accuracy = args['accuracy']
        else:
            accuracy = ''

        if self.dump_times == 0:
            writer = SummaryWriter('runs/eval/baseline' +
                                   '_acc' + str(accuracy), model)
        else:
            writer = SummaryWriter('runs/eval/tune_' +
                                   str(self.dump_times) +
                                   '_acc' + str(accuracy), model)

        if self.dump_times == 0:
            for (input, _) in self.q_dataloader:
                if isinstance(input, dict):
                    if self.device == "gpu":
                        for inp in input.keys():
                            input[inp] = input[inp].to("dpcpp")
                elif isinstance(input, list) or isinstance(input, tuple):
                    if self.device == "gpu":
                        input = [inp.to("dpcpp") for inp in input]
                else:
                    if self.device == "gpu":
                        input = input.to("dpcpp")
                writer.add_graph(model, input)
                break

        summary = OrderedDict()
        observer_dict = {}
        get_observer_dict(model, observer_dict)
        for key in observer_dict:
            if isinstance(observer_dict[key],
                          torch.nn.modules.linear.Identity):
                continue
            op_name = key.strip(".activation_post_process")
            summary[op_name + ".output"] = observer_dict[key].get_tensor_value()
            for iter in summary[op_name + ".output"]:
                # Only collect last fused child output
                op = op_name
                if self.is_fused_child(op_name) == True and \
                   self.is_last_fused_child(op_name) == True:
                    op = op_name[:op_name.rfind('.')]
                else:
                    if self.is_fused_child(op_name) == True and \
                       self.is_last_fused_child(op_name) == False:
                        continue
                    else:
                        op = op_name

                if summary[op_name + ".output"][iter].is_quantized:
                    writer.add_histogram(
                        op + "/Output/int8",
                        torch.dequantize(summary[op_name +
                                                 ".output"][iter]))
                else:
                    writer.add_histogram(
                        op + "/Output/fp32",
                        summary[op_name + ".output"][iter])

        state_dict = model.state_dict()
        for key in state_dict:
            if not isinstance(state_dict[key], torch.Tensor):
                continue

            op = key[:key.rfind('.')]
            if self.is_fused_child(op) is True:
                # fused child tensorboard tag will be merge
                weight = key[key.rfind('.')+1:]
                op = op[:op.rfind('.')] + '/' + weight
            else:
                weight = key[key.rfind('.')+1:]
                op = key[:key.rfind('.')] + '/' + weight

            # To merge ._packed_params
            op = op.replace('._packed_params', '')

            if state_dict[key].is_quantized:
                writer.add_histogram(op + "/int8",
                                     torch.dequantize(state_dict[key]))
            else:
                writer.add_histogram(op + "/fp32", state_dict[key])

        writer.close()
        self.dump_times = self.dump_times + 1

        return summary

    @dump_elapsed_time("Pass save quantized model")
    def save(self, model, path=None):
        pass

    def inspect_tensor(self, model, dataloader, op_list=None, iteration_list=None,
                       inspect_type='activation', save_to_disk=False):
        if self.version > PT17_VERSION:
            from torch.fx import GraphModule
            if type(model.model) == GraphModule:
                assert False, "Inspect_tensor didn't support fx graph model now!"
        from torch import dequantize
        import numpy as np
        is_quantized = model.is_quantized
        op_list_ = []
        fp32_int8_map = {}
        for op_name in op_list:
            op_list_.append(op_name)
            for key in self.fused_dict:
                if op_name in self.fused_dict[key]:
                    fp32_int8_map[op_name] = \
                        {'activation': self.fused_dict[key][-1], 'weight': key}
                    if is_quantized:
                        op_list_.append(key)
                        op_list_.remove(op_name)
                    else:
                        op_list_.append(self.fused_dict[key][-1])

        new_model = model if is_quantized else copy.deepcopy(model)

        assert min(iteration_list) > 0, \
            "Iteration number should great zero, 1 means first iteration."
        iterations = max(iteration_list) if iteration_list is not None else -1
        new_model = self._pre_eval_hook(new_model, op_list=op_list_, iteration_list=iteration_list)
        self.evaluate(new_model, dataloader, iteration=iterations)
        observer_dict = {}
        ret = {}
        if inspect_type == 'activation' or inspect_type == 'all':
            from torch.quantization import get_observer_dict
            ret['activation'] = []
            get_observer_dict(new_model.model, observer_dict)
            if iteration_list is None:
                iteration_list = [1]
            for i in iteration_list:
                summary = OrderedDict()
                for key in observer_dict:
                    if isinstance(observer_dict[key],
                                  torch.nn.modules.linear.Identity):
                        continue
                    op_name = key.replace(".activation_post_process", "")
                    value = observer_dict[key].get_tensor_value()[i]
                    if op_name in op_list:
                        if type(value) is list:
                            summary[op_name] = {}
                            for index in range(len(value)):
                                summary[op_name].update(
                                    {op_name + ".output" +
                                     str(index): dequantize(value[index]).numpy()
                                     if value[index].is_quantized else value[index].numpy()})
                        else:
                            summary[op_name] = {op_name + ".output0":
                                                dequantize(value).numpy()
                                                if value.is_quantized else value.numpy()}
                    else:
                        if bool(self.fused_dict):
                            if is_quantized:
                                for a in fp32_int8_map:
                                    if op_name == fp32_int8_map[a]['weight']:
                                        if type(value) is list:
                                            summary[a] = {}
                                            for index in range(len(value)):
                                                summary[a].update(
                                                    {op_name + ".output" +
                                                     str(index): dequantize(value[index]).numpy()
                                                     if value[index].is_quantized else
                                                     value[index].numpy()})
                                        else:
                                            summary[a] = {op_name + ".output0":
                                                          dequantize(value).numpy()
                                                          if value.is_quantized else
                                                          value.numpy()}
                            else:
                                for a in fp32_int8_map:
                                    if op_name == fp32_int8_map[a]['activation']:
                                        if type(value) is list:
                                            summary[a] = {}
                                            for index in range(len(value)):
                                                summary[a].update(
                                                    {op_name + ".output" +
                                                     str(index): dequantize(value[index]).numpy()
                                                     if value[index].is_quantized else
                                                     value[index].numpy()})
                                        else:
                                            summary[a] = {op_name + ".output0":
                                                          dequantize(value).numpy()
                                                          if value.is_quantized else
                                                          value.numpy()}

                if save_to_disk:
                    dump_dir = os.path.join(self.workspace_path, 'dump_tensor')
                    os.makedirs(dump_dir, exist_ok=True)
                    np.savez(os.path.join(dump_dir, 'activation_iter{}.npz'.format(i)), **summary)

                ret['activation'].append(summary)

        if inspect_type == 'weight' or inspect_type == 'all':
            ret['weight'] = {}
            state_dict = new_model.model.state_dict()

            for key in state_dict:
                if not isinstance(state_dict[key], torch.Tensor):
                    continue
                if 'weight' not in key and 'bias' not in key:
                    continue

                op = key[:key.rfind('.')]
                op = op.replace('._packed_params', '')

                if op in op_list:
                    if op in ret['weight']:
                        ret['weight'][op].update({key: dequantize(state_dict[key]).numpy()
                            if state_dict[key].is_quantized else
                            state_dict[key].detach().numpy()})
                    else:
                        ret['weight'][op] = {key: dequantize(state_dict[key]).numpy()
                            if state_dict[key].is_quantized else
                            state_dict[key].detach().numpy()}
                else:
                    if bool(self.fused_dict):
                        if is_quantized:
                            for a in fp32_int8_map:
                                if op == fp32_int8_map[a]['weight']:
                                    if a in ret['weight']:
                                        ret['weight'][a].update(
                                            {key: dequantize(state_dict[key]).numpy()
                                                if state_dict[key].is_quantized else
                                                    state_dict[key].detach().numpy()})
                                    else:
                                        ret['weight'][a] = \
                                            {key: dequantize(state_dict[key]).numpy()
                                                if state_dict[key].is_quantized else
                                                    state_dict[key].detach().numpy()}
                                    break

            if save_to_disk:
                np.savez(os.path.join(dump_dir, 'weight.npz'), **ret['weight'])
        else:
            ret['weight'] = None

        return ret

    def set_tensor(self, model, tensor_dict):
        state_dict = model.model.state_dict()
        tensor_name = None
        for key in tensor_dict.keys():
            end = key.rfind('.')
            op_name = key[:end]
            state_op_name = None
            weight_bias = key[end+1:]
            for op in self.fused_dict:
                if op_name in self.fused_dict[op]:
                    state_op_name = op
            if state_op_name is None:
                state_op_name = op_name
            for state_dict_key in state_dict.keys():
                state_key_end = state_dict_key.rfind('.')
                state_key = state_dict_key[:state_key_end].replace('._packed_params', '')
                if weight_bias in state_dict_key and state_op_name == state_key:
                    tensor_name = state_dict_key
            assert tensor_name is not None, key + " is not in the state dict"
            tensor = torch.from_numpy(tensor_dict[key])
            dtype = state_dict[tensor_name].dtype
            if state_dict[tensor_name].is_quantized:
                if 'channel' in str(state_dict[tensor_name].qscheme()):
                    scales = state_dict[tensor_name].q_per_channel_scales()
                    zero_points = state_dict[tensor_name].q_per_channel_zero_points()
                    axis = state_dict[tensor_name].q_per_channel_axis()
                    state_dict[tensor_name] = torch.quantize_per_channel(tensor, scales,
                                                                         zero_points,
                                                                         axis, dtype=dtype)
                elif 'tensor' in str(state_dict[tensor_name].qscheme()):
                    scales = state_dict[tensor_name].q_scale()
                    zero_points = state_dict[tensor_name].q_zero_point()
                    state_dict[tensor_name] = torch.quantize_per_tensor(tensor, scales,
                                                                        zero_points, dtype)
            else:
                state_dict[tensor_name] = tensor
        model.model.load_state_dict(state_dict)


unify_op_type_mapping_ipex = {
    "Convolution_Relu": "Conv2d",
    "Convolution_Sum_Relu": "Conv2d",
    "Convolution_BatchNorm": "Conv2d",
    "Linear_Relu": "Linear"
}


@adaptor_registry
class PyTorch_IPEXAdaptor(TemplateAdaptor): # pragma: no cover
    """Adaptor of PyTorch framework with Intel PyTorch Extension,
       all PyTorch IPEX API is in this class.

    Args:
        framework_specific_info (dict): dictionary of tuning configure from yaml file.
    """
    def __init__(self, framework_specific_info):
        super(PyTorch_IPEXAdaptor, self).__init__(framework_specific_info)

        query_config_file = "pytorch_ipex.yaml"
        self.query_handler = PyTorchQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), query_config_file))
        self.cfgs = None

        self.ipex_config_path = \
            os.path.join(self.workspace_path, 'ipex_config_tmp.json')

        try:
            os.remove(self.ipex_config_path)
        except:
            logger.warning('removing {} fails'.format(self.ipex_config_path))

    def model_calibration(self, q_model, dataloader, iterations=1, conf=None):
        assert iterations > 0
        with torch.no_grad():
            for idx, (input, label) in enumerate(dataloader):
                if isinstance(input, dict):
                    for inp in input.keys():
                        input[inp] = input[inp].to(ipex.DEVICE) \
                            if isinstance(input[inp], torch.Tensor) else input[inp]
                    with ipex.AutoMixPrecision(conf, running_mode='calibration'):
                        output = q_model(**input)
                elif isinstance(input, list) or isinstance(input, tuple):
                    input = [inp.to(ipex.DEVICE)
                             if isinstance(inp, torch.Tensor) else inp for inp in input]
                    with ipex.AutoMixPrecision(conf, running_mode='calibration'):
                        output = q_model(*input)
                else:
                    if isinstance(input, torch.Tensor):
                        input = input.to(ipex.DEVICE)  # pylint: disable=no-member
                    with ipex.AutoMixPrecision(conf, running_mode='calibration'):
                        output = q_model(input)
                if idx >= iterations - 1:
                    break

    @dump_elapsed_time("Pass quantize model")
    def quantize(self, tune_cfg, model, dataloader, q_func=None):
        """Execute the quantize process on the specified model.

        Args:
            tune_cfg (dict): quantization config.
            model (object): model need to do quantization, it is LPOT model.
            dataloader (object): calibration dataset.
            q_func (objext, optional): training function for quantization aware training mode.

        Returns:
            (dict): quantized model
        """

        try:
            model_ = copy.deepcopy(model)
        except Exception as e:                              # pragma: no cover
            logger.warning("Deepcopy failed: {}, inplace=True now!".format(repr(e)))
            model_ = model
        try:
            q_model = torch.jit.script(model_.model.eval().to(ipex.DEVICE))
        except:
            try:
                for input, _ in dataloader:
                    q_model = torch.jit.trace(model_.model.eval().to(ipex.DEVICE),
                                              input.to(ipex.DEVICE)).to(ipex.DEVICE)
                    break
            except:
                logger.info("This model can't convert to Script model")
                q_model = model_.model.eval().to(ipex.DEVICE)
        self._cfg_to_qconfig(tune_cfg)

        if self.approach == 'post_training_static_quant':
            iterations = tune_cfg.get('calib_iteration', 1)
            ipex_conf = ipex.AmpConf(torch.int8, configure_file=self.ipex_config_path)
            self.model_calibration(q_model, dataloader, iterations, conf=ipex_conf)
            ipex_conf.save(self.ipex_config_path)

        assert self.approach != 'quant_aware_training', "Intel PyTorch Extension didn't support \
                               quantization aware training mode"
        model_.model = q_model
        model_.tune_cfg = copy.deepcopy(self.cfgs)

        if self.is_baseline:
            self.is_baseline = False
        return model_

    def _cfg_to_qconfig(self, tune_cfg):
        """Convert tune configure to quantization config for each op.

            Args:
                tune_cfg (dict): dictionary of tune configure for each op
                ipex_config_path: configure file of Intel PyTorch Extension

            tune_cfg should be a format like below:
            {
              'calib_iteration': 10,
              'op': {
                 ('op1', 'CONV2D'): {
                   'activation':  {'dtype': 'uint8',
                                   'algorithm': 'minmax',
                                   'scheme':'sym',
                                   'granularity': 'per_tensor'},
                   'weight': {'dtype': 'int8',
                              'algorithm': 'kl',
                              'scheme':'asym',
                              'granularity': 'per_channel'}
                 },
                 ('op2', 'RELU): {
                   'activation': {'dtype': 'int8',
                   'scheme': 'asym',
                   'granularity': 'per_tensor',
                   'algorithm': 'minmax'}
                 },
                 ('op3', 'CONV2D'): {
                   'activation':  {'dtype': 'fp32'},
                   'weight': {'dtype': 'fp32'}
                 },
                 ...
              }
            }
        """
        assert self.cfgs is not None, "No configure for IPEX int8 model..."
        for key in tune_cfg['op']:
            value = tune_cfg['op'][key]
            assert isinstance(value, dict)
            assert 'activation' in value
            if value['activation']['dtype'] == 'fp32':
                if 'weight' in value:
                    assert value['weight']['dtype'] == 'fp32'
                for op_cfg in self.cfgs:
                    if op_cfg["id"] == key[0]:
                        op_cfg["quantized"] = False
            else:
                for op_cfg in self.cfgs:
                    if op_cfg["id"] == key[0]:
                        op_cfg["quantized"] = True
        with open(self.ipex_config_path, 'w') as write_f:
            json.dump(self.cfgs, write_f)

    def evaluate(self, model, dataloader, postprocess=None,
                 metric=None, measurer=None, iteration=-1,
                 tensorboard=False, fp32_baseline=False):
        """Execute the evaluate process on the specified model.

        Args:
            model (object): LPOT model to run evaluation.
            dataloader (object): evaluation dataset.
            postprocess (object, optional): process function after evaluation.
            metric (object, optional): metric function.
            measurer (object, optional): measurer function.
            iteration (int, optional): number of iterations to evaluate.
            tensorboard (bool, optional): dump output tensor to tensorboard summary
                                          files(IPEX unspport).
            fp32_baseline (boolen, optional): only for compare_label=False pipeline

        Returns:
            (dict): quantized model
        """
        assert not tensorboard, "Intel PyTorch Extension didn't tensor dump"

        model_ = model.model
        model_.eval()
        if self.is_baseline:
            model_.to(ipex.DEVICE)

        ipex_config = self.ipex_config_path if not self.benchmark else \
                      os.path.join(self.workspace_path, 'best_configure.json')
        conf = ipex.AmpConf(torch.int8, configure_file=ipex_config) \
            if not self.is_baseline else ipex.AmpConf(None)

        with torch.no_grad():
            for idx, (input, label) in enumerate(dataloader):
                if measurer is not None:
                    measurer.start()

                if isinstance(input, dict):
                    for inp in input.keys():
                        input[inp] = input[inp].to(ipex.DEVICE) \
                            if isinstance(input[inp], torch.Tensor) else input[inp]
                    with ipex.AutoMixPrecision(conf, running_mode='inference'):
                        output = model_(**input)
                elif isinstance(input, list) or isinstance(input, tuple):
                    input = [inp.to(ipex.DEVICE)
                             if isinstance(inp, torch.Tensor) else inp for inp in input]
                    with ipex.AutoMixPrecision(conf, running_mode='inference'):
                        output = model_(*input)
                else:
                    if isinstance(input, torch.Tensor):
                        input = input.to(ipex.DEVICE)  # pylint: disable=no-member
                    with ipex.AutoMixPrecision(conf, running_mode='inference'):
                        output = model_(input)
                label = label.to("cpu")
                output = output.to("cpu")
                if measurer is not None:
                    measurer.end()
                if postprocess is not None:
                    output, label = postprocess((output, label))
                if metric is not None:
                    metric.update(output, label)
                if idx + 1 == iteration:
                    break
        acc = metric.result() if metric is not None else 0

        return acc

    def _get_quantizable_ops_recursively(self, model, prefix, quantizable_ops):
        """This is a helper function for `query_fw_capability`,
           and it will get all quantizable ops from model.

        Args:
            model (object): input model
            prefix (string): prefix of op name
            quantizable_ops (list): list of quantizable ops from model include op name and type.

        Returns:
            None
        """

        if not os.path.exists(self.ipex_config_path):
            assert isinstance(model, torch.nn.Module), \
                    "The model passed in is not the instance of torch.nn.Module"

            model_ = copy.deepcopy(model)
            model_.eval().to(ipex.DEVICE)
            try:
                init_model = torch.jit.script(model_)
            except:
                try:
                    for input, _ in self.q_dataloader:
                        init_model = torch.jit.trace(model_, input.to(ipex.DEVICE))
                        break
                except:
                    logger.info("This model can't convert to Script model")
                    init_model = model_

            # create a quantization config file for intel pytorch extension model
            os.makedirs(os.path.dirname(self.ipex_config_path), exist_ok=True)
            ipex_conf = ipex.AmpConf(torch.int8)
            self.model_calibration(init_model, self.q_dataloader, conf=ipex_conf)
            ipex_conf.save(self.ipex_config_path)

        with open(self.ipex_config_path, 'r') as f:
            self.cfgs = json.load(f)
            for op_cfg in self.cfgs:
                quantizable_ops.append((op_cfg["id"],
                                       unify_op_type_mapping_ipex[op_cfg["name"]]
                                       if op_cfg["name"] in unify_op_type_mapping_ipex else
                                       op_cfg["name"]))
        os.remove(self.ipex_config_path)

    @dump_elapsed_time("Pass save quantized model")
    def save(self, model, path=None):
        """The function is used by tune strategy class for set best configure in LPOT model.

           Args:
               model (object): The LPOT model which is best results.
               path (string): No used.

        Returns:
            None
        """

        pass

    def inspect_tensor(self, model, dataloader, op_list=None, iteration_list=None,
                       inspect_type='activation', save_to_disk=False):
        assert False, "Inspect_tensor didn't support IPEX backend now!"


@adaptor_registry
class PyTorch_FXAdaptor(TemplateAdaptor):                           # pragma: no cover
    """Adaptor of PyTorch framework with FX graph mode, all PyTorch API is in this class.

    Args:
        framework_specific_info (dict): dictionary of tuning configure from yaml file.
    """
    def __init__(self, framework_specific_info):
        super(PyTorch_FXAdaptor, self).__init__(framework_specific_info)
        assert self.version >= PT18_VERSION, \
                      "Please use PyTroch 1.8 or higher version with pytorch_fx backend"
        import torch.quantization as tq
        """
        # Map for swapping float module to quantized ones,
        # and this dictionary will change with different PoTorch versions
        DEFAULT_MODULE_MAPPING = {
            nn.Linear: nnq.Linear,
            nn.ReLU: nnq.ReLU,
            nn.ReLU6: nnq.ReLU6,
            nn.Conv2d: nnq.Conv2d,
            nn.Conv3d: nnq.Conv3d,
            QuantStub: nnq.Quantize,
            DeQuantStub: nnq.DeQuantize,
            # Wrapper Modules:
            nnq.FloatFunctional: nnq.QFunctional,
            # Intrinsic modules:
            nni.ConvReLU2d: nniq.ConvReLU2d,
            nni.ConvReLU3d: nniq.ConvReLU3d,
            nni.LinearReLU: nniq.LinearReLU,
            nniqat.ConvReLU2d: nniq.ConvReLU2d,
            nniqat.LinearReLU: nniq.LinearReLU,
            nniqat.ConvBn2d: nnq.Conv2d,
            nniqat.ConvBnReLU2d: nniq.ConvReLU2d,
            # QAT modules:
            nnqat.Linear: nnq.Linear,
            nnqat.Conv2d: nnq.Conv2d,
        }
        """

        self.tune_cfg = None
        if self.device == "cpu":
            query_config_file = "pytorch_cpu.yaml"
        else:
            assert False, "Unsupport this device {}".format(self.device)
        self.query_handler = PyTorchQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), query_config_file))

        self.white_list = \
            tq.quantization_mappings.get_default_dynamic_quant_module_mappings() \
            if self.approach == 'post_training_dynamic_quant' else \
            tq.quantization_mappings.get_default_qconfig_propagation_list()

    def model_calibration(self, q_model, dataloader, iterations=1):
        assert iterations > 0
        with torch.no_grad():
            for idx, (input, label) in enumerate(dataloader):
                if isinstance(input, dict):
                    output = q_model(**input)
                elif isinstance(input, list) or isinstance(input, tuple):
                    output = q_model(*input)
                else:
                    output = q_model(input)
                if idx >= iterations - 1:
                    break

    @dump_elapsed_time("Pass quantize model")
    def quantize(self, tune_cfg, model, dataloader, q_func=None):
        """Execute the quantize process on the specified model.

        Args:
            tune_cfg (dict): quantization config.
            model (object): model need to do quantization.
            dataloader (object): calibration dataset.
            q_func (objext, optional): training function for quantization aware training mode.

        Returns:
            (object): quantized model
        """

        assert isinstance(model.model, torch.nn.Module), \
               "The model passed in is not the instance of torch.nn.Module"

        self.tune_cfg = tune_cfg
        self.tune_cfg["approach"] = self.approach
        self.tune_cfg["framework"] = "pytorch_fx"
        op_cfgs = _cfg_to_qconfig(tune_cfg, self.approach)

        from torch.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx
        try:
            q_model = copy.deepcopy(model)
        except Exception as e:                              # pragma: no cover
            logger.warning("Deepcopy failed: {}, inplace=True now!".format(repr(e)))
            q_model = model
        q_model.model.eval()
        fx_op_cfgs = _cfgs_to_fx_cfgs(op_cfgs, self.approach)
        if self.approach == 'quant_aware_training':
            q_model.model.train()
            q_model.model = prepare_qat_fx(q_model.model, fx_op_cfgs,
              prepare_custom_config_dict=q_model.kwargs
              if q_model.kwargs is not None else None)
            if q_func is None:
                assert False, \
                    "quantization aware training mode requires q_function to train"
            else:
                q_func(q_model.model)
            q_model.model.eval()
        else:
            q_model.model = prepare_fx(q_model.model, fx_op_cfgs,
                                       prepare_custom_config_dict=q_model.kwargs
                                       if q_model.kwargs is not None else None)
            if self.approach == 'post_training_static_quant':
                iterations = tune_cfg.get('calib_iteration', 1)
                self.model_calibration(q_model.model, dataloader, iterations)
        q_model.model = convert_fx(q_model.model)
        q_model.tune_cfg = copy.deepcopy(self.tune_cfg)
        if self.is_baseline:
            self.is_baseline = False
        return q_model


    def evaluate(self, model, dataloader, postprocess=None,
                 metric=None, measurer=None, iteration=-1,
                 tensorboard=False, fp32_baseline=False):
        """Execute the evaluate process on the specified model.

        Args:
            model (object): model to run evaluation.
            dataloader (object): evaluation dataset.
            postprocess (object, optional): process function after evaluation.
            metric (object, optional): metric function.
            measurer (object, optional): measurer function.
            iteration (int, optional): number of iterations to evaluate.
            tensorboard (bool, optional): dump output tensor to tensorboard summary files.
            fp32_baseline (boolen, optional): only for compare_label=False pipeline

        Returns:
            (object): accuracy
        """
        if tensorboard:
            assert False, "PyTorch FX mode didn't support tensorboard flag now!"

        model_ = model.model
        assert isinstance(
            model_, torch.nn.Module), "The model passed in is not the instance of torch.nn.Module"
        model_.eval()
        model_.to(self.device)

        if metric and hasattr(metric, "compare_label") and not metric.compare_label:
            self.fp32_preds_as_label = True
            results = []

        with torch.no_grad():
            if metric:
                metric.reset()
            for idx, (input, label) in enumerate(dataloader):
                if measurer is not None:
                    measurer.start()

                if isinstance(input, dict):
                    output = model_(**input)
                elif isinstance(input, list) or isinstance(input, tuple):
                    output = model_(*input)
                else:
                    output = model_(input)
                if measurer is not None:
                    measurer.end()
                if postprocess is not None:
                    output, label = postprocess((output, label))
                if metric is not None and not self.fp32_preds_as_label:
                    metric.update(output, label)
                if self.fp32_preds_as_label:
                    self.fp32_results.append(output) if fp32_baseline else \
                        results.append(output)
                if idx + 1 == iteration:
                    break

        if self.fp32_preds_as_label:
            from .torch_utils.util import collate_torch_preds
            if fp32_baseline:
                results = collate_torch_preds(self.fp32_results)
                metric.update(results, results)
            else:
                reference = collate_torch_preds(self.fp32_results)
                results = collate_torch_preds(results)
                metric.update(results, reference)

        acc = metric.result() if metric is not None else 0

        return acc

    def train(self, model, dataloader, optimizer_tuple, criterion_tuple, hooks, **kwargs):
        """Execute the train process on the specified model.

        Args:
            model (object): model to run evaluation.
            dataloader (object): training dataset.
            optimizer (tuple): It is a tuple of (cls, parameters) for optimizer.
            criterion (tuple): It is a tuple of (cls, parameters) for criterion.
            kwargs (dict, optional): other parameters.

        Returns:
            None
        """
        model_ = model.model
        optimizer = optimizer_tuple[0](model_.parameters(), **optimizer_tuple[1])
        criterion = criterion_tuple[0](**criterion_tuple[1])
        start_epochs = kwargs['kwargs']['start_epoch']
        end_epochs = kwargs['kwargs']['end_epoch']
        iters = kwargs['kwargs']['iteration']
        if hooks is not None:
            on_epoch_start = hooks['on_epoch_start']
            on_epoch_end = hooks['on_epoch_end']
            on_batch_start = hooks['on_batch_start']
            on_batch_end = hooks['on_batch_end']
        for nepoch in range(start_epochs, end_epochs):
            model_.train()
            cnt = 0
            if hooks is not None:
                on_epoch_start(nepoch)
            for image, target in dataloader:
                if hooks is not None:
                    on_batch_start(cnt)
                print('.', end='', flush=True)
                cnt += 1
                output = model_(image)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if hooks is not None:
                    on_batch_end()
                if cnt >= iters:
                    break
            if hooks is not None:
                on_epoch_end()

    def _get_quantizable_ops_recursively(self, model, prefix, quantizable_ops):
        """This is a helper function for `query_fw_capability`,
           and it will get all quantizable ops from model.

        Args:
            model (object): input model
            prefix (string): prefix of op name
            quantizable_ops (list): list of quantizable ops from model include op name and type.

        Returns:
            None
        """

        for name, child in model.named_children():
            op_name = prefix + '.' + name if prefix != '' else name
            if type(child) in self.white_list and type(child) != torch.nn.Sequential and \
                    type(child) != torch.quantization.stubs.DeQuantStub:
                quantizable_ops.append((
                    op_name, unify_op_type_mapping[str(child.__class__.__name__)]
                    if str(child.__class__.__name__) in unify_op_type_mapping else
                    str(child.__class__.__name__)))
            else:
                self._get_quantizable_ops_recursively(child, op_name, quantizable_ops)


class PyTorchQuery(QueryBackendCapability):

    def __init__(self, local_config_file=None):
        import torch

        super().__init__()
        self.version = get_torch_version()
        self.cfg = local_config_file
        self.cur_config = None
        self._one_shot_query()

    def _get_specified_version_cfg(self, data):
        """Get the configuration for the current runtime。
        If there's no matched configuration in the input yaml, we'll
        use the `default` field of yaml.

        Args:
            data (Yaml content): input yaml file.

        Returns:
            [dictionary]: the content for specific version.
        """
        # default_config = None
        for sub_data in data:
            if sub_data['version']['name'] == 'default':
                return sub_data
            sub_data_version = StrictVersion(sub_data['version']['name'])
            if self.version >= sub_data_version:
                return sub_data

    def _one_shot_query(self):
        with open(self.cfg) as f:
            content = yaml.safe_load(f)
            try:
                self.cur_config = self._get_specified_version_cfg(content)
            except Exception as e:
                self.logger.info("Failed to parse {} due to {}".format(self.cfg, str(e)))
                self.cur_config = None
                raise ValueError("Please check the {} format.".format(self.cfg))

    def get_quantization_capability(self):
        """Get the supported op types' quantization capability.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is a dict that describes all op types' quantization capability.
        """
        return self.cur_config['capabilities']

