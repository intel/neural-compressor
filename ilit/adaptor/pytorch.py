import pandas as pd
from .adaptor import adaptor_registry, Adaptor
from ..utils.utility import LazyImport, AverageMeter, compute_sparsity, CPUINFO_FLAGS
import copy
from collections import OrderedDict
from ..utils import logger
import random
import numpy as np
import os
import yaml

torch = LazyImport('torch')

REDUCE_RANGE = False if "avx512vnni" in CPUINFO_FLAGS else True
logger.debug("reduce range:")
logger.debug(REDUCE_RANGE)


def _cfg_to_qconfig(tune_cfg, is_insert_fakequant=False):
    '''tune_cfg should be a format like below:
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
    '''
    op_qcfgs = OrderedDict()
    for key in tune_cfg['op']:
        value = tune_cfg['op'][key]
        assert isinstance(value, dict)
        assert 'weight' in value
        assert 'activation' in value
        if value['activation']['dtype'] == 'fp32':
            assert value['weight']['dtype'] == 'fp32'
            op_qcfgs[key] = None
        else:
            weight = value['weight']
            activation = value['activation']

            scheme = weight['scheme']
            granularity = weight['granularity']
            algorithm = weight['algorithm']
            dtype = weight['dtype']
            if is_insert_fakequant:
                weights_fake_quantize = _fake_quantize(algorithm, scheme, granularity, dtype)
            else:
                weights_observer = _observer(algorithm, scheme, granularity, dtype)

            scheme = activation['scheme']
            granularity = activation['granularity']
            algorithm = activation['algorithm']
            dtype = activation['dtype']
            if is_insert_fakequant:
                activation_fake_quantize = _fake_quantize(algorithm, scheme, granularity, dtype)
            else:
                activation_observer = _observer(algorithm, scheme, granularity, dtype)

            if is_insert_fakequant:
                qconfig = torch.quantization.QConfig(
                    activation=activation_fake_quantize, weight=weights_fake_quantize)
            else:
                qconfig = torch.quantization.QConfig(
                    activation=activation_observer, weight=weights_observer)

            op_qcfgs[key] = qconfig

    return op_qcfgs


def _observer(algorithm, scheme, granularity, dtype):
    if algorithm == 'minmax':
        if granularity == 'per_channel':
            observer = torch.quantization.PerChannelMinMaxObserver
            if scheme == 'sym':
                qscheme = torch.per_channel_symmetric
            else:
                assert scheme == 'asym'
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
        if granularity == 'per_channel':
            if scheme == 'sym':
                qscheme = torch.per_channel_symmetric
            else:
                assert scheme == 'asym'
                qscheme = torch.per_channel_affine
        else:
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
        if granularity == 'per_channel':
            if scheme == 'sym':
                qscheme = torch.per_channel_symmetric
            else:
                assert scheme == 'asym'
                qscheme = torch.per_channel_affine
        else:
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


def _propagate_qconfig(model, op_qcfgs):
    fallback_ops = []
    WHITE_LIST = torch.quantization.default_mappings.DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST \
        - torch.quantization.default_mappings._INCLUDE_QCONFIG_PROPAGATE_LIST
    for k, v in op_qcfgs.items():
        if v is None and k[1] != str(torch.quantization.QuantStub) \
                and k[1] != str(torch.quantization.DeQuantStub):
            fallback_ops.append(k[0])
        else:
            if v is None:
                weights_observer = _observer('minmax', 'asym',
                                             'per_channel', 'int8')
                activation_observer = _observer('minmax', 'sym',
                                                'per_tensor', 'uint8')
                v = torch.quantization.QConfig(
                    activation=activation_observer, weight=weights_observer)
            op_qcfg = {k[0]: v}
            _propagate_qconfig_recursively(model, '', op_qcfg, white_list=WHITE_LIST)

    if fallback_ops:
        _fallback_quantizable_ops_recursively(model, '', fallback_ops)


def _propagate_qconfig_recursively(model, prefix, op_qcfg, white_list, qconfig_parent=None):
    for name, child in model.named_children():
        model_qconfig = qconfig_parent
        op_name = prefix + name
        if op_name in op_qcfg:
            child.qconfig = op_qcfg[op_name]
            model_qconfig = op_qcfg[op_name]
        elif model_qconfig is not None and type(child) in white_list:
            child.qconfig = model_qconfig
        _propagate_qconfig_recursively(
            child, op_name + '.', op_qcfg, white_list, model_qconfig)


def _find_quantized_op_num(model, white_list, op_count=0):
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


def _fallback_quantizable_ops_recursively(model, prefix, fallback_ops):
    class DequantQuantWrapper(torch.nn.Module):
        r"""A wrapper class that wraps the input module, adds DeQuantStub and
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

    WHITE_LIST = torch.quantization.default_mappings.DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST \
        - torch.quantization.default_mappings._INCLUDE_QCONFIG_PROPAGATE_LIST
    for name, child in model.named_children():
        op_name = prefix + name
        if op_name in fallback_ops:
            child.qconfig = None
            quantize_op_num = _find_quantized_op_num(model, white_list=WHITE_LIST)
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
                child, op_name + '.', fallback_ops)


@adaptor_registry
class PyTorchAdaptor(Adaptor):
    def __init__(self, framework_specific_info):
        super(PyTorchAdaptor, self).__init__(framework_specific_info)
        """
        # Map for swapping float module to quantized ones
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

        # set torch random seed
        random_seed = framework_specific_info['random_seed']
        random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.approach = framework_specific_info['approach']
        self.device = framework_specific_info['device']
        self.is_baseline = True
        self.tune_cfg = None

        self.white_list = \
            torch.quantization.default_mappings.DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST \
            - torch.quantization.default_mappings._INCLUDE_QCONFIG_PROPAGATE_LIST

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

        if framework_specific_info['approach'] == "post_training_static_quant":
            self.q_mapping = torch.quantization.default_mappings.DEFAULT_MODULE_MAPPING
        elif framework_specific_info['approach'] == "quant_aware_training":
            self.q_mapping = torch.quantization.default_mappings.DEFAULT_QAT_MODULE_MAPPING
        else:
            assert False, "Unsupport quantization approach: {}".format(self.approach)

        if self.device == "cpu":
            self.capability = \
                {
                    'activation':
                    {
                        'granularity': ['per_tensor'],
                        'scheme': ['asym', 'sym'],
                        'dtype': ['uint8', 'fp32'],
                        'algorithm': ['kl', 'minmax'],
                    },
                    'weight':
                    {
                        'granularity': ['per_channel'],
                        'scheme': ['asym', 'sym'],
                        'dtype': ['int8', 'fp32'],
                        'algorithm': ['minmax'],
                    }
                }
        elif self.device == "gpu":
            self.capability = \
                {
                    'activation':
                    {
                        'granularity': ['per_tensor'],
                        'scheme': ['sym'],
                        'dtype': ['uint8', 'fp32', 'int8'],
                        'algorithm': ['minmax'],
                    },
                    'weight':
                    {
                        'granularity': ['per_channel'],
                        'scheme': ['sym'],
                        'dtype': ['int8', 'fp32'],
                        'algorithm': ['minmax'],
                    }
                }
        else:
            assert False, "Unsupport this device {}".format(self.device)

    def quantize(self, tune_cfg, model, dataloader, q_func=None):
        """Execute the quantize process on the specified model.

        Args:
            tune_cfg (dict): quantization config.
            model (object): model need to do quantization.
            dataloader (object): calibration dataset.
            q_func (optional): training function for quantization aware training mode.

        Returns:
            (dict): quantized model
        """
        assert isinstance(
            model, torch.nn.Module), "The model passed in is not the instance of torch.nn.Module"

        q_model = copy.deepcopy(model.eval())
        if self.approach == 'quant_aware_training':
            q_model.train()
        elif self.approach == 'post_training_static_quant':
            q_model.eval()

        # For tensorboard display
        self.tune_cfg = tune_cfg

        op_cfgs = _cfg_to_qconfig(tune_cfg, (self.approach == 'quant_aware_training'))
        _propagate_qconfig(q_model, op_cfgs)
        # sanity check common API misusage
        if not any(hasattr(m, 'qconfig') and m.qconfig for m in q_model.modules()):
            logger.warn("None of the submodule got qconfig applied. Make sure you "
                        "passed correct configuration through `qconfig_dict` or "
                        "by assigning the `.qconfig` attribute directly on submodules")
        torch.quantization.add_observer_(q_model)

        if self.approach == 'post_training_static_quant':
            iterations = tune_cfg.get('calib_iteration', 1)
            assert iterations >= 1
            with torch.no_grad():
                for _, (input, label) in enumerate(dataloader):
                    if isinstance(input, dict):
                        if self.device == "gpu":
                            for inp in input.keys():
                                input[inp] = input[inp].to("dpcpp")
                        output = q_model(**input)
                    elif isinstance(input, list) or isinstance(input, tuple):
                        if self.device == "gpu":
                            input = [inp.to("dpcpp") for inp in input]
                        output = q_model(*input)
                    else:
                        if self.device == "gpu":
                            input = input.to("dpcpp")
                        output = q_model(input)

                    iterations -= 1
                    if iterations == 0:
                        break
        elif self.approach == 'quant_aware_training':
            torch.quantization.convert(q_model, self.q_mapping, inplace=True)
            if q_func is None:
                assert False, "quantization aware training mode requires q_function to train"
            else:
                q_func(q_model)
            q_model.eval()

        q_model = torch.quantization.convert(q_model, inplace=True)

        return q_model

    def evaluate(self, model, dataloader, postprocess=None, \
                 metric=None, measurer=None, iteration=-1, tensorboard=False):
        assert isinstance(
            model, torch.nn.Module), "The model passed in is not the instance of torch.nn.Module"
        model.eval()
        if self.device == "cpu":
            model.to("cpu")
        elif self.device == "gpu":
            if self.is_baseline:
                model.to("dpcpp")

        if tensorboard:
            model = self._pre_eval_hook(model)

        if self.is_baseline:
            self.is_baseline = False

        with torch.no_grad():
            for idx, (input, label) in enumerate(dataloader):
                if measurer is not None:
                    measurer.start()

                if isinstance(input, dict):
                    if self.device == "gpu":
                        for inp in input.keys():
                            input[inp] = input[inp].to("dpcpp")
                    output = model(**input)
                elif isinstance(input, list) or isinstance(input, tuple):
                    if self.device == "gpu":
                        input = [inp.to("dpcpp") for inp in input]
                    output = model(*input)
                else:
                    if self.device == "gpu":
                        input = input.to("dpcpp")
                    output = model(input)
                if self.device == "gpu":
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

        if tensorboard:

            for idx, (input, label) in enumerate(dataloader):
                if isinstance(input, dict):
                    if self.device == "gpu":
                        for inp in input.keys():
                            input[inp] = input[inp].to("dpcpp")
                    self._post_eval_hook(model, accuracy=acc, input=input)
                elif isinstance(input, list) or isinstance(input, tuple):
                    if self.device == "gpu":
                        input = [inp.to("dpcpp") for inp in input]
                    self._post_eval_hook(model, accuracy=acc, input=input)
                else:
                    if self.device == "gpu":
                        input = input.to("dpcpp")
                    self._post_eval_hook(model, accuracy=acc, input=input)
                break
        return acc


    def _get_quantizable_ops_recursively(self, model, prefix, quantizable_ops):
        for name, child in model.named_children():
            op_name = prefix + name
            if type(child) in self.white_list:
                quantizable_ops.append((op_name, str(type(child))))
            else:
                self._get_quantizable_ops_recursively(
                    child, op_name + '.', quantizable_ops)

    def query_fused_patterns(self, model):
        pass

    def query_fw_capability(self, model):
        quantizable_ops = []
        self._get_quantizable_ops_recursively(model, '', quantizable_ops)

        q_capability = {}
        q_capability['modelwise'] = self.capability
        q_capability['opwise'] = OrderedDict()

        for q_op in quantizable_ops:
            q_capability['opwise'][q_op] = copy.deepcopy(self.capability)

        return q_capability

    def inspect_tensor(self, model, dataloader, op_list=[], iteration_list=[]):
        pass

    def _pre_eval_hook(self, model):
        '''The function is used to do some preprocession before evaluation phase.

        Return:
              model
        '''
        from abc import ABCMeta

        ABC = ABCMeta(str("ABC"), (object, ),
                      {})  # compatible with Python 2 *and* 3:

        class _RecordingObserver(ABC, torch.nn.Module):
            r"""
            The module is mainly for debug and records the tensor values during runtime.

            """

            def __init__(self, iteration_list=None, **kwargs):
                super(_RecordingObserver, self).__init__(**kwargs)
                self.output_tensors_dict = OrderedDict()
                self.current_iter = 0
                self.iteration_list = iteration_list

            def forward(self, x):
                if (self.iteration_list is None and self.current_iter == 0) or \
                    (self.iteration_list is not None and
                     self.current_iter in self.iteration_list):
                    self.output_tensors_dict[self.current_iter] = x.to("cpu") \
                        if x.device != "cpu" else x.clone()
                self.current_iter += 1
                return x

            @torch.jit.export
            def calculate_qparams(self):
                raise Exception(
                    "calculate_qparams should not be called for RecordingObserver"
                )

            @torch.jit.export
            def get_tensor_value(self):
                return self.output_tensors_dict

        def _observer_forward_hook(module, input, output):
            r"""Forward hook that calls observer on the output
            """
            return module.activation_post_process(output)

        def _add_observer_(module, op_list=None, prefix=""):
            r"""Add observer for the leaf child of the module.

            This function insert observer module to all leaf child module that
            has a valid qconfig attribute.

            Args:
                module: input module with qconfig attributes for all the leaf modules that
                we want to dump tensor

            Return:
                None, module is modified inplace with added observer modules and forward_hooks
            """
            for name, child in module.named_children():
                op_name = name if prefix == "" else prefix + "." + name
                if isinstance(child, torch.nn.quantized.FloatFunctional):
                    if hasattr(child,
                               'qconfig') and child.qconfig is not None and (
                                   op_list is None or op_name in op_list):
                        child.activation_post_process = \
                            child.qconfig.activation()
                else:
                    _add_observer_(child, op_list, op_name)

            # Insert observers only for leaf nodes
            if hasattr(module, 'qconfig') and module.qconfig is not None and \
                    len(module._modules) == 0 and not isinstance(module, torch.nn.Sequential) and \
                    (op_list is None or prefix in op_list):
                # observer and hook will be gone after we swap the module
                module.add_module(
                    'activation_post_process',
                    module.qconfig.activation())
                module.register_forward_hook(_observer_forward_hook)

        def is_fused_module(module):
            op_type = str(type(module))
            op_type = op_type[op_type.rfind('.')+1:].strip('>').strip('\'') 
            op_type = 'nni.' + op_type
            if op_type in self.fused_op:
                return True
            else:
                return False


        def _propagate_qconfig_helper(module,
                                      qconfig_dict,
                                      white_list=None,
                                      qconfig_parent=None,
                                      prefix='',
                                      fused=False):
            r"""This is a helper function for `propagate_qconfig_`

            Args:
                module: input module
                qconfig_dict: dictionary that maps from name of submodule to quantization
                             configuration
                white_list: list of quantizable modules
                qconfig_parent: config of parent module, we will fallback to
                               this config when there is no specified config for current
                               module
                prefix: corresponding prefix of the current module, used as key in
                        qconfig_dict

            Return:
                None, module is modified inplace with qconfig attached
            """
            # TODO: Add test
            if white_list is None:
                white_list = \
                    torch.quantization.default_mappings.DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST

            module_qconfig = qconfig_dict.get(type(module), qconfig_parent)
            module_qconfig = qconfig_dict.get(prefix, module_qconfig)
            module_qconfig = getattr(module, 'qconfig', module_qconfig)

            if type(module) in white_list:
                module.qconfig = module_qconfig
            for name, child in module.named_children():
                module_prefix = prefix + '.' + name if prefix else name
                if is_fused_module(module):
                   if prefix in self.fused_dict:
                      self.fused_dict[prefix] = [self.fused_dict[prefix], module_prefix]
                   else:
                      self.fused_dict[prefix] = module_prefix 
                   _fused = True
                else:
                   _fused = False 

                _propagate_qconfig_helper(child, qconfig_dict, white_list,
                                          module_qconfig, module_prefix, fused=_fused)

        def _prepare(model, inplace=True, op_list=[], white_list=None):
            r"""
            The model will be attached with observer or fake quant modules, and qconfig
            will be propagated.

            Args:
                model: input model to be modified in-place
                inplace: carry out model transformations in-place, the original module is mutated
            """
            if not inplace:
                model = copy.deepcopy(model)
            _propagate_qconfig_helper(model,
                                      qconfig_dict={},
                                      white_list=white_list)
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
        white_list = self.white_list | \
            (set(torch.quantization.default_mappings.DEFAULT_MODULE_MAPPING.values()) |
             set(torch.quantization.default_mappings.DEFAULT_QAT_MODULE_MAPPING.values()) |
             set(torch.quantization.default_mappings.DEFAULT_DYNAMIC_MODULE_MAPPING.values()))

        model = model if not self.is_baseline else copy.deepcopy(model)
        model.qconfig = torch.quantization.QConfig(
            weight=torch.quantization.default_weight_observer,
            activation=_RecordingObserver)
        model = _prepare(model, op_list=None, white_list=white_list)

        if self.is_baseline:
            self.is_baseline = False

        return model

    def is_fused_child(self, op_name):
        op = op_name[:op_name.rfind('.')]
        if op in self.fused_dict and op_name[op_name.rfind('.')+1:].isdigit():
           return True
        else:
           return False

    def is_fused_op(self, op_name):
        op = op_name[:op_name.rfind('.')]
        if op in self.fused_dict:
           return True
        else:
           return False

    def is_last_fused_child(self, op_name):
        op = op_name[:op_name.rfind('.')]
        if op_name in self.fused_dict[op][-1]:
           return True
        else:
           return False

    def _post_eval_hook(self, model, **args):
        '''The function is used to do some post process after complete evaluation.
        '''
        from torch.utils.tensorboard import SummaryWriter
        from torch.quantization import get_observer_dict

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

        if args is not None and 'input' in args and self.dump_times == 0:
           writer.add_graph(model, args['input'])

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
            if self.is_fused_child(op) == True:
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

    def get_all_weight_names(self, model):
        names = []
        for name, param in model.named_parameters():
            names.append(name)
        return names

    def get_weight(self, model, tensor_name):
        for name, param in model.named_parameters():
            if tensor_name == name:
                return param.data

    def update_weights(self, model, tensor_name, new_tensor):
        new_tensor = torch.Tensor(new_tensor)
        for name, param in model.named_parameters():
            if name == tensor_name:
                param.data.copy_(new_tensor.data)
        return model

    def report_sparsity(self, model):
        df = pd.DataFrame(columns=['Name', 'Shape', 'NNZ (dense)', 'NNZ (sparse)', "Sparsity(%)",
                                   'Std', 'Mean', 'Abs-Mean'])
        pd.set_option('precision', 2)
        param_dims = [2, 4]
        params_size = 0
        sparse_params_size = 0
        for name, param in model.named_parameters():
            # Extract just the actual parameter's name, which in this context we treat
            # as its "type"
            if param.dim() in param_dims and any(type in name for type in ['weight', 'bias']):
                param_size, sparse_param_size, dense_param_size = compute_sparsity(
                    param.detach().numpy())
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

    def save(self, model, path):
        '''The function is used by tune strategy class for saving model.

           Args:
               model (object): The model to saved.
               path (string): The path where to save.
        '''

        path = os.path.expanduser(path)
        os.makedirs(path, exist_ok=True)
        try:
            with open(os.path.join(path, "best_configure.yaml"), 'w') as f:
                yaml.dump(self.tune_cfg, f, default_flow_style=False)
        except IOError as e:
            logger.error("Unable to save configure file. %s" % e)

        torch.save(model.state_dict(), os.path.join(path, "best_model_weights.pt"))
