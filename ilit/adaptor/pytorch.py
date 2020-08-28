from .adaptor import adaptor_registry, Adaptor
from ..utils.utility import LazyImport, AverageMeter
import copy
from collections import OrderedDict
from ..utils import logger
import random

torch = LazyImport('torch')

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

        if framework_specific_info['approach'] == "post_training_static_quant":
            self.q_mapping = torch.quantization.default_mappings.DEFAULT_MODULE_MAPPING
        elif framework_specific_info['approach'] == "quant_aware_training":
            self.q_mapping = torch.quantization.default_mappings.DEFAULT_QAT_MODULE_MAPPING
        else:
            assert False, "Unsupport quantization approach: {}".format(self.approach)

        self.white_list = torch.quantization.default_mappings.DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST\
            - torch.quantization.default_mappings._INCLUDE_QCONFIG_PROPAGATE_LIST

        self.approach = framework_specific_info['approach']
        self.device = framework_specific_info['device']
        self.is_baseline = True

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
                        'dtype': ['uint8', 'fp32'],
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

        op_cfgs = self._cfg_to_qconfig(tune_cfg)
        self._propagate_qconfig(q_model, op_cfgs)
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

    def evaluate(self, model, dataloader, postprocess=None, metric=None):
        assert isinstance(
            model, torch.nn.Module), "The model passed in is not the instance of torch.nn.Module"
        model.eval()
        if self.device == "cpu":
            model.to("cpu")
        elif self.device == "gpu":
            if self.is_baseline:
                model.to("dpcpp")
                self.is_baseline = False
        with torch.no_grad():
            for _, (input, label) in enumerate(dataloader):
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
                if postprocess is not None:
                    output = postprocess(output)
                if metric is not None:
                    metric.update(output, label)
        acc = metric.result() if metric is not None else 0
        return acc

    def _cfg_to_qconfig(self, tune_cfg):
        '''tune_cfg should be a format like below:
          {
            'fuse': {'int8': [['CONV2D', 'RELU', 'BN'], ['CONV2D', 'RELU']], 'fp32': [['CONV2D', 'RELU', 'BN']]},
            'calib_iteration': 10,
            'op': {
               ('op1', 'CONV2D'): {
                 'activation':  {'dtype': 'uint8', 'algorithm': 'minmax', 'scheme':'sym', 'granularity': 'per_tensor'},
                 'weight': {'dtype': 'int8', 'algorithm': 'kl', 'scheme':'asym', 'granularity': 'per_channel'}
               },
               ('op2', 'RELU): {
                 'activation': {'dtype': 'int8', 'scheme': 'asym', 'granularity': 'per_tensor', 'algorithm': 'minmax'}
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
                if self.approach == 'post_training_static_quant':
                    weights_observer = self._observer(algorithm, scheme, granularity, dtype)
                elif self.approach == 'quant_aware_training':
                    weights_fake_quantize = self._fake_quantize(
                        algorithm, scheme, granularity, dtype)
                else:
                    assert False, "Unsupport quantization approach: {}".format(self.approach)

                scheme = activation['scheme']
                granularity = activation['granularity']
                algorithm = activation['algorithm']
                dtype = activation['dtype']
                if self.approach == 'post_training_static_quant':
                    activation_observer = self._observer(algorithm, scheme, granularity, dtype)
                elif self.approach == 'quant_aware_training':
                    activation_fake_quantize = self._fake_quantize(
                        algorithm, scheme, granularity, dtype)
                else:
                    assert False, "Unsupport quantization approach: {}".format(self.approach)

                if self.approach == 'post_training_static_quant':
                    qconfig = torch.quantization.QConfig(
                        activation=activation_observer, weight=weights_observer)
                elif self.approach == 'quant_aware_training':
                    qconfig = torch.quantization.QConfig(
                        activation=activation_fake_quantize, weight=weights_fake_quantize)
                else:
                    assert False, "Unsupport quantization approach: {}".format(self.approach)

                op_qcfgs[key] = qconfig

        return op_qcfgs

    def _observer(self, algorithm, scheme, granularity, dtype):
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

        return observer.with_args(qscheme=qscheme, dtype=dtype)

    def _fake_quantize(self, algorithm, scheme, granularity, dtype):
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
                                    dtype=dtype, qscheme=qscheme)

    def _propagate_qconfig(self, model, op_qcfgs):
        fallback_ops = []
        for k, v in op_qcfgs.items():
            if v is None and k[1] != torch.quantization.QuantStub \
                    and k[1] != torch.quantization.DeQuantStub:
                fallback_ops.append(k[0])
            else:
                if v is None:
                    weights_observer = self._observer('minmax', 'asym', 'per_channel', 'int8')
                    activation_observer = self._observer('minmax', 'sym', 'per_tensor', 'uint8')
                    v = torch.quantization.QConfig(
                        activation=activation_observer, weight=weights_observer)
                op_qcfg = {k[0]: v}
                self._propagate_qconfig_recursively(model, '', op_qcfg)

        if fallback_ops:
            self._fallback_quantizable_ops_recursively(model, '', fallback_ops)

    def _propagate_qconfig_recursively(
            self, model, prefix, op_qcfg, qconfig_parent=None):
        for name, child in model.named_children():
            model_qconfig = qconfig_parent
            op_name = prefix + name
            if op_name in op_qcfg:
                child.qconfig = op_qcfg[op_name]
                model_qconfig = op_qcfg[op_name]
            elif model_qconfig is not None and type(child) in self.white_list:
                child.qconfig = model_qconfig
            self._propagate_qconfig_recursively(
                child, op_name + '.', op_qcfg, model_qconfig)

    def _find_quantized_op_num(self, model, op_count=0):
        quantize_op_num = op_count
        for name_tmp, child_tmp in model.named_children():
            if type(child_tmp) in self.white_list \
                and not (isinstance(child_tmp, torch.quantization.QuantStub)
                         or isinstance(child_tmp, torch.quantization.DeQuantStub)):
                quantize_op_num += 1
            else:
                quantize_op_num = self._find_quantized_op_num(
                    child_tmp, quantize_op_num)
        return quantize_op_num

    def _fallback_quantizable_ops_recursively(
            self, model, prefix, fallback_ops):
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

        for name, child in model.named_children():
            op_name = prefix + name
            if op_name in fallback_ops:
                child.qconfig = None
                quantize_op_num = self._find_quantized_op_num(model)
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
                            child, observer=self._observer)
                else:
                    model._modules[name] = DequantQuantWrapper(
                        child, observer=self._observer)
            else:
                self._fallback_quantizable_ops_recursively(
                    child, op_name + '.', fallback_ops)

    def _get_quantizable_ops_recursively(self, model, prefix, quantizable_ops):
        for name, child in model.named_children():
            op_name = prefix + name
            if type(child) in self.white_list:
                quantizable_ops.append((op_name, type(child)))
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
