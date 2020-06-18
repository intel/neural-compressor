from .adaptor import adaptor_registry, Adaptor
from ..utils import LazyImport, AverageMeter
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
              'mode': ['sym', 'asym'],
              'data_type':['uint8', 'fp32'],
              'algo': ['minmax', 'kl'],
            },
          'weight':
            {
              'granularity': ['per_channel'],
              'mode': ['sym', 'asym'],
              'data_type':['int8', 'fp32'],
              'algo': ['minmax'],
            }
        }

    def quantize(self, tune_cfg, model, dataloader):
        assert isinstance(model, torch.nn.Module), "The model passed in is not the instance of torch.nn.Module"

        q_model = copy.deepcopy(model)

        op_cfgs = self._cfg_to_qconfig(tune_cfg)
        self._propagate_qconfig(q_model, op_cfgs)
        # sanity check common API misusage
        if not any(hasattr(m, 'qconfig') and m.qconfig for m in q_model.modules()):
            print("None of the submodule got qconfig applied. Make sure you "
                      "passed correct configuration through `qconfig_dict` or "
                      "by assigning the `.qconfig` attribute directly on submodules")
        torch.quantization.add_observer_(q_model)
        q_model.eval()

        iterations = tune_cfg.get('calib_iteration', 1)
        assert iterations >= 1
        with torch.no_grad():
            for _, (input, label) in enumerate(dataloader):
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
                 'activation':  {'data_type': 'uint8', 'algo': 'minmax', 'mode':'sym', 'granularity': 'per_tensor'},
                 'weight': {'data_type': 'int8', 'algo': 'kl', 'mode':'asym', 'granularity': 'per_channel'}
               },
               ('op2', 'RELU): {
                 'activation': {'data_type': 'int8', 'mode': 'asym', 'granularity': 'per_tensor', 'algo': 'minmax'}
               },
               ('op3', 'CONV2D'): {
                 'activation':  {'data_type': 'fp32'},
                 'weight': {'data_type': 'fp32'}
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
            if value['activation']['data_type'] == 'fp32':
                assert value['weight']['data_type'] == 'fp32'
                op_qcfgs[key[0]] = None
            else:
                weight = value['weight']
                activation = value['activation']

                mode = weight['mode']
                gran = weight['granularity']
                algo = weight['algo']
                dtype = weight['data_type']
                weights_observer = self._observer(algo, mode, gran, dtype)

                mode = activation['mode']
                gran = activation['granularity']
                algo = activation['algo']
                dtype = activation['data_type']
                activation_observer = self._observer(algo, mode, gran, dtype)

                qconfig = torch.quantization.QConfig(activation=activation_observer, weight=weights_observer)
                op_qcfgs[key[0]] = qconfig

        return op_qcfgs

    def _observer(self, algo, mode, granularity, dtype):
        if algo == 'minmax':
            if granularity == 'per_channel':
                observer = torch.quantization.PerChannelMinMaxObserver
                if mode == 'sym':
                    qscheme = torch.per_channel_symmetric
                else:
                    assert mode == 'asym'
                    qscheme = torch.per_channel_affine
            else:
                assert granularity == 'per_tensor'
                observer = torch.quantization.MinMaxObserver
                if mode == 'sym':
                    qscheme = torch.per_tensor_symmetric
                else:
                    assert mode == 'asym'
                    qscheme = torch.per_tensor_affine
        else:
            assert algo == 'kl'
            observer = torch.quantization.HistogramObserver
            if granularity == 'per_channel':
                if mode == 'sym':
                    qscheme = torch.per_channel_symmetric
                else:
                    assert mode == 'asym'
                    qscheme = torch.per_channel_affine
            else:
                assert granularity == 'per_tensor'
                if mode == 'sym':
                    qscheme = torch.per_tensor_symmetric
                else:
                    assert mode == 'asym'
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
            if v is None:
                fallback_ops.append(k)
            else:
                op_qcfg = {k: v}
                self._propagate_qconfig_recursively(model, '', op_qcfg)

        if fallback_ops:
            self._fallback_quantizable_ops_recursively(model, '', fallback_ops)

    def _propagate_qconfig_recursively(self, model, prefix, op_qcfg, qconfig_parent=None):
        model_qconfig = qconfig_parent
        for name, child in model.named_children():
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

            def __init__(self, module, quant = False):
                super(DequantQuantWrapper, self).__init__()
                self.add_module('quant', torch.quantization.QuantStub(module.qconfig))
                if quant:
                    self.add_module('dequant', torch.nn.Identity())
                else:
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
                if isinstance(child, torch.quantization.QuantStub):
                    model._modules[name] = DequantQuantWrapper(child, True)
                quantize_op_num = 0
                for name_tmp, child_tmp in model.named_children():
                    if type(child_tmp) in self.q_mapping.keys() \
                        and not (isinstance(child_tmp, torch.quantization.QuantStub) or isinstance(child_tmp, torch.quantization.DeQuantStub)):
                        quantize_op_num += 1
                if quantize_op_num == 1:
                    for name_tmp, child_tmp in model.named_children():
                        if isinstance(child_tmp, torch.quantization.QuantStub) or isinstance(child_tmp, torch.quantization.DeQuantStub):
                            model._modules[name_tmp] = torch.nn.Identity()
                else:
                    for name_tmp, child_tmp in model.named_children():
                        if hasattr(child_tmp, 'qconfig') \
                        and not (isinstance(child_tmp, torch.quantization.QuantStub) or isinstance(child_tmp, torch.quantization.DeQuantStub)):
                            model._modules[name_tmp] = DequantQuantWrapper(child_tmp)
            else:
                self._fallback_quantizable_ops_recursively(child, op_name + '.', fallback_ops)

    def _get_quantizable_ops_recursively(self, model, prefix, quantizable_ops):
        for name, child in model.named_children():
            op_name = prefix + name
            if type(child) in self.q_mapping.keys():
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

    def inspect_tensor(self, model):
        pass

