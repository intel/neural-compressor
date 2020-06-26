from .adaptor import adaptor_registry, Adaptor
from ..utils.utility import LazyImport, AverageMeter
import copy
from collections import OrderedDict

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
        self.q_mapping = torch.quantization.default_mappings.DEFAULT_MODULE_MAPPING

        self.capability = \
        {
          'activation':
            {
              'granularity': ['per_tensor'],
              'scheme': ['sym', 'asym'],
              'dtype':['uint8', 'fp32'],
              'algorithm': ['minmax', 'kl'],
            },
          'weight':
            {
              'granularity': ['per_channel'],
              'scheme': ['sym', 'asym'],
              'dtype':['int8', 'fp32'],
              'algorithm': ['minmax'],
            }
        }

    def quantize(self, tune_cfg, model, dataloader):
        assert isinstance(model, torch.nn.Module), "The model passed in is not the instance of torch.nn.Module"

        q_model = copy.deepcopy(model.eval())

        op_cfgs = self._cfg_to_qconfig(tune_cfg)
        self._propagate_qconfig(q_model, op_cfgs)
        # sanity check common API misusage
        if not any(hasattr(m, 'qconfig') and m.qconfig for m in q_model.modules()):
            print("None of the submodule got qconfig applied. Make sure you "
                      "passed correct configuration through `qconfig_dict` or "
                      "by assigning the `.qconfig` attribute directly on submodules")
        torch.quantization.add_observer_(q_model)

        iterations = tune_cfg.get('calib_iteration', 1)
        assert iterations >= 1
        with torch.no_grad():
            for _, (input, label) in enumerate(dataloader):
                if isinstance(input, dict):
                    output = q_model(**input)
                elif isinstance(input, list) or isinstance(input, tuple):
                    output = q_model(*input)
                else:
                    output = q_model(input)

                iterations -= 1
                if iterations == 0:
                    break

        q_model = torch.quantization.convert(q_model, inplace = True)

        return q_model

    def evaluate(self, model, dataloader, metric=None):
        assert isinstance(model, torch.nn.Module), "The model passed in is not the instance of torch.nn.Module"
        model.to('cpu')
        model.eval()
        with torch.no_grad():
            for _, (input, label) in enumerate(dataloader):
                if isinstance(input, dict):
                    output = model(**input)
                elif isinstance(input, list) or isinstance(input, tuple):
                    output = model(*input)
                else:
                    output = model(input)
                acc = metric.evaluate(output, label)
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
                weights_observer = self._observer(algorithm, scheme, granularity, dtype)

                scheme = activation['scheme']
                granularity = activation['granularity']
                algorithm = activation['algorithm']
                dtype = activation['dtype']
                activation_observer = self._observer(algorithm, scheme, granularity, dtype)

                qconfig = torch.quantization.QConfig(activation=activation_observer, weight=weights_observer)
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

    def _propagate_qconfig(self, model, op_qcfgs):
        fallback_ops = []
        for k, v in op_qcfgs.items():
            if v is None and k[1] != torch.quantization.QuantStub \
                and k[1] != torch.quantization.DeQuantStub and k[1] != torch.nn.quantized.FloatFunctional:
                fallback_ops.append(k[0])
            else:
                if v is None:
                    weights_observer = self._observer('minmax', 'asym', 'per_channel', 'int8')
                    activation_observer = self._observer('minmax', 'sym', 'per_tensor', 'uint8')
                    v = torch.quantization.QConfig(activation=activation_observer, weight=weights_observer)
                op_qcfg = {k[0]: v}
                self._propagate_qconfig_recursively(model, '', op_qcfg)

        if fallback_ops:
            self._fallback_quantizable_ops_recursively(model, '', fallback_ops)

    def _propagate_qconfig_recursively(self, model, prefix, op_qcfg, qconfig_parent=None):
        for name, child in model.named_children():
            model_qconfig = qconfig_parent
            op_name = prefix + name
            if op_name in op_qcfg:
                child.qconfig = op_qcfg[op_name]
                model_qconfig = op_qcfg[op_name]
            elif model_qconfig is not None:
                child.qconfig = model_qconfig
            self._propagate_qconfig_recursively(child, op_name + '.', op_qcfg, model_qconfig)

    def _fallback_quantizable_ops_recursively(self, model, prefix, fallback_ops):
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

            def __init__(self, module, observer = None):
                super(DequantQuantWrapper, self).__init__()
                if not module.qconfig and observer:
                    weights_observer = observer('minmax', 'asym', 'per_channel', 'int8')
                    activation_observer = observer('minmax', 'sym', 'per_tensor', 'uint8')
                    module.qconfig = torch.quantization.QConfig(activation=activation_observer, weight=weights_observer)
                self.add_module('quant', torch.quantization.QuantStub(module.qconfig))
                self.add_module('dequant', torch.quantization.DeQuantStub())
                self.add_module('module', module)
                module.qconfig = None
                self.train(module.training)

            def forward(self, X):
                X = self.dequant(X)
                X = self.module(X)
                return self.quant(X)

        for name, child in model.named_children():
            op_name = prefix + name
            if op_name in fallback_ops:
                child.qconfig = None
                quantize_op_num = 0
                for name_tmp, child_tmp in model.named_children():
                    if type(child_tmp) in self.q_mapping.keys() \
                        and not (isinstance(child_tmp, torch.quantization.QuantStub) or isinstance(child_tmp, torch.quantization.DeQuantStub)):
                        quantize_op_num += 1
                if quantize_op_num == 1:
                    found = False
                    for name_tmp, child_tmp in model.named_children():
                        if isinstance(child_tmp, torch.quantization.QuantStub) or isinstance(child_tmp, torch.quantization.DeQuantStub):
                            model._modules[name_tmp] = torch.nn.Identity()
                            found = True
                    if not found:
                        model._modules[name] = DequantQuantWrapper(child, observer=self._observer)
                else:
                    model._modules[name] = DequantQuantWrapper(child, observer=self._observer)
            else:
                self._fallback_quantizable_ops_recursively(child, op_name + '.', fallback_ops)

    def _get_quantizable_ops_recursively(self, model, prefix, quantizable_ops):
        for name, child in model.named_children():
            op_name = prefix + name
            if type(child) in self.q_mapping.keys() and not isinstance(child, torch.quantization.DeQuantStub):
                quantizable_ops.append((op_name, type(child)))
            else:
                self._get_quantizable_ops_recursively(child, op_name + '.', quantizable_ops)

    def query_fused_patterns(self, model):
        pass

    def query_fw_capability(self, model):
        quantizable_ops = []
        self._get_quantizable_ops_recursively(model, '', quantizable_ops)

        q_capability = {}
        q_capability['modelwise'] = self.capability
        q_capability['opwise'] = OrderedDict()

        for q_op in quantizable_ops:
            q_capability['opwise'][q_op] = self.capability

        return q_capability

    def inspect_tensor(self, model, dataloader, op_list=[], iteration_list=[]):
        pass

