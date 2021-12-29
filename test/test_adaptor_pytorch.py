import torch
import torch.nn as nn
import torch.nn.quantized as nnq
from torch.quantization import QuantStub, DeQuantStub
import torchvision
import unittest
import os
from neural_compressor.adaptor import FRAMEWORKS
from neural_compressor.model import MODELS
from neural_compressor.adaptor.pytorch import PyTorchVersionMode
import neural_compressor.adaptor.pytorch as nc_torch
from neural_compressor.experimental import Quantization, common
from neural_compressor.conf.config import Quantization_Conf
from neural_compressor.utils.pytorch import load
from neural_compressor.utils.utility import recover
import shutil
import copy
import numpy as np
import yaml

try:
    try:
        import intel_pytorch_extension as ipex
    except:
        import intel_extension_for_pytorch as ipex
    TEST_IPEX = True
except:
    TEST_IPEX = False

PT_VERSION = nc_torch.get_torch_version()
if PT_VERSION >= PyTorchVersionMode.PT18.value:
    FX_MODE = True
else:
    FX_MODE = False


fake_dyn_yaml = '''
    model:
      name: imagenet
      framework: pytorch

    quantization:
      approach: post_training_dynamic_quant
      op_wise: {
              'decoder': {
                'activation':  {'dtype': ['fp32']},
                'weight': {'dtype': ['fp32']}
              }
      }
    evaluation:
      accuracy:
        metric:
          topk: 1
      performance:
        warmup: 5
        iteration: 10

    tuning:
      accuracy_criterion:
        relative:  0.01
      exit_policy:
        timeout: 0
      random_seed: 9527
      workspace:
        path: saved
    '''


fake_ptq_yaml = '''
    model:
      name: imagenet
      framework: pytorch

    quantization:
      op_wise: {
              'quant': {
                'activation': {'dtype': ['fp32']},
                'weight': {'dtype': ['fp32']}
              },
              'layer1.0.conv1': {
                'activation': {'dtype': ['fp32']},
                'weight': {'dtype': ['fp32']}
              },
              'layer1.0.conv2': {
                'activation': {'dtype': ['fp32']},
                'weight': {'dtype': ['fp32']}
              },
              'layer2.0.conv1': {
                'activation': {'dtype': ['uint8'], 'algorithm': ['minmax'], 'granularity': ['per_tensor'], 'scheme':['sym']},
                'weight': {'dtype': ['int8'], 'algorithm': ['minmax'], 'granularity': ['per_channel'], 'scheme':['sym']}
              },
              'layer3.0.conv1': {
                'activation': {'dtype': ['uint8'], 'algorithm': ['kl'], 'granularity': ['per_tensor'], 'scheme':['sym']},
                'weight': {'dtype': ['int8'], 'algorithm': ['minmax'], 'granularity': ['per_channel'], 'scheme':['sym']}
              },
              'layer1.0.add_relu': {
                'activation': {'dtype': ['fp32']},
                'weight': {'dtype': ['fp32']}
              },
      }
    evaluation:
      accuracy:
        metric:
          topk: 1
      performance:
        warmup: 1
        iteration: 10

    tuning:
      accuracy_criterion:
        relative:  0.01
      exit_policy:
        timeout: 0
      random_seed: 9527
      workspace:
        path: saved
    '''

fake_ptq_yaml_for_fx = '''
    model:
      name: imagenet
      framework: pytorch_fx

    quantization:
      op_wise: {
              'quant': {
                'activation': {'dtype': ['fp32']},
                'weight': {'dtype': ['fp32']}
              },
              'layer1.0.conv1': {
                'activation': {'dtype': ['fp32']},
                'weight': {'dtype': ['fp32']}
              },
              'layer1.0.conv2': {
                'activation': {'dtype': ['fp32']},
                'weight': {'dtype': ['fp32']}
              },
              'layer2.0.conv1': {
                'activation': {'dtype': ['uint8'], 'algorithm': ['minmax'], 'granularity': ['per_tensor'], 'scheme':['sym']},
                'weight': {'dtype': ['int8'], 'algorithm': ['minmax'], 'granularity': ['per_channel'], 'scheme':['sym']}
              },
              'layer3.0.conv1': {
                'activation': {'dtype': ['uint8'], 'algorithm': ['kl'], 'granularity': ['per_tensor'], 'scheme':['sym']},
                'weight': {'dtype': ['int8'], 'algorithm': ['minmax'], 'granularity': ['per_channel'], 'scheme':['sym']}
              },
              'layer1.0.add_relu': {
                'activation': {'dtype': ['fp32']},
                'weight': {'dtype': ['fp32']}
              },
              'default_qconfig': {
                'activation':  {'dtype': ['fp32']},
                'weight': {'dtype': ['fp32']}
              }
      }
    evaluation:
      accuracy:
        metric:
          topk: 1
      performance:
        warmup: 5
        iteration: 10

    tuning:
      accuracy_criterion:
        relative:  0.01
      exit_policy:
        timeout: 0
      random_seed: 9527
      workspace:
        path: saved
    '''


fake_qat_yaml = '''
    model:
      name: imagenet
      framework: pytorch

    quantization:
      approach: quant_aware_training
      train:
        end_epoch: 1
        iteration: 1
        optimizer:
          SGD:
            learning_rate: 0.0001
        criterion:
          CrossEntropyLoss:
            reduction: mean
      op_wise: {
              'quant': {
                'activation': {'dtype': ['fp32']},
                'weight': {'dtype': ['fp32']}
              },
              'layer1.0.conv1': {
                'activation': {'dtype': ['fp32']},
                'weight': {'dtype': ['fp32']}
              },
              'layer1.0.conv2': {
                'activation': {'dtype': ['fp32']},
                'weight': {'dtype': ['fp32']}
              },
              'layer2.0.conv1': {
                'activation': {'dtype': ['uint8'], 'algorithm': ['minmax'], 'granularity': ['per_tensor'], 'scheme':['sym']},
                'weight': {'dtype': ['int8'], 'algorithm': ['minmax'], 'granularity': ['per_channel'], 'scheme':['sym']}
              },
              'layer3.0.conv1': {
                'activation': {'dtype': ['uint8'], 'algorithm': ['kl'], 'granularity': ['per_tensor'], 'scheme':['sym']},
                'weight': {'dtype': ['int8'], 'algorithm': ['minmax'], 'granularity': ['per_channel'], 'scheme':['sym']}
              },
              'layer1.0.add_relu': {
                'activation': {'dtype': ['fp32']},
                'weight': {'dtype': ['fp32']}
              }
      }
    evaluation:
      accuracy:
        metric:
          topk: 1

    tuning:
      accuracy_criterion:
        relative:  0.01
      exit_policy:
        timeout: 0
      random_seed: 9527
      workspace:
        path: saved
    '''


def build_pytorch_yaml():
    with open('ptq_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_ptq_yaml)

    with open('dynamic_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_dyn_yaml)

    with open('qat_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_qat_yaml)


def build_pytorch_fx_yaml():
    if PT_VERSION >= PyTorchVersionMode.PT19.value:
      fake_fx_ptq_yaml = fake_ptq_yaml_for_fx
    else:
      fake_fx_ptq_yaml = fake_ptq_yaml.replace('pytorch', 'pytorch_fx')
    with open('fx_ptq_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_fx_ptq_yaml)

    fake_fx_dyn_yaml = fake_dyn_yaml.replace('pytorch', 'pytorch_fx')
    with open('fx_dynamic_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_fx_dyn_yaml)

    fake_fx_qat_yaml = fake_qat_yaml.replace('pytorch', 'pytorch_fx')
    with open('fx_qat_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_fx_qat_yaml)


def build_ipex_yaml():
    fake_yaml = '''
        model:
          name: imagenet
          framework: pytorch_ipex

        evaluation:
          accuracy:
            metric:
              topk: 1
          performance:
            warmup: 5
            iteration: 10

        tuning:
          accuracy_criterion:
            relative:  0.01
          exit_policy:
            timeout: 0
          random_seed: 9527
          workspace:
            path: saved
        '''
    with open('ipex_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)


def build_dump_tensors_yaml():
    fake_yaml = '''
        model:
          name: imagenet
          framework: pytorch

        evaluation:
          accuracy:
            metric:
              topk: 1

        tuning:
          accuracy_criterion:
            relative:  0.01
          exit_policy:
            timeout: 0
          random_seed: 9527
          workspace:
            path: saved
          tensorboard: true
        '''
    with open('dump_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.conv = nn.Conv2d(3, 1, 1)
        self.linear = nn.Linear(224 * 224, 5)
        self.dequant = DeQuantStub()

    def forward(self, x):
        dim = x.size()
        x = self.quant(x)
        x = self.conv(x)
        x = x.view(1, -1)
        x = self.linear(x)
        x = self.dequant(x)
        return x


class FP32Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)
    def forward(self, x):
        x = self.conv(x)
        return x


class SubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.conv = nn.Conv2d(1, 1, 1)
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.fp32 = FP32Model()
        self.norm = nn.LayerNorm([1, 224, 224])
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.conv(x)
        x = self.quant(x)
        x = self.conv1(x)
        x = self.dequant(x)
        x = self.fp32(x)
        x = self.norm(x)
        return x


class PartialQuantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.conv = nn.Conv2d(3, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.bn1 = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(1, 1, 1)
        self.linear = nn.Linear(224 * 224, 1)
        self.dequant = DeQuantStub()
        self.sub = SubModel()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sub(x)
        x = self.quant(x)
        x = self.conv2(x)
        x = x.view(1, -1)
        x = self.linear(x)
        x = self.dequant(x)
        return x


def eval_func(model):
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        input = torch.randn(1, 3, 224, 224)
        # compute output
        output = model(input)
    return 0.0


def q_func(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    # switch to evaluate mode
    model.train()
    input = torch.randn(1, 3, 224, 224)
    # compute output
    output = model(input)
    loss = output.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return model


class TestPytorchAdaptor(unittest.TestCase):
    framework_specific_info = {"device": "cpu",
                               "approach": "post_training_static_quant",
                               "random_seed": 1234,
                               "q_dataloader": None,
                               "workspace_path": "./"}
    framework = "pytorch"
    adaptor = FRAMEWORKS[framework](framework_specific_info)
    model = torchvision.models.quantization.resnet18()
    nc_model = MODELS['pytorch'](model)

    @classmethod
    def setUpClass(self):
        build_pytorch_yaml()
        build_dump_tensors_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('ptq_yaml.yaml')
        os.remove('dynamic_yaml.yaml')
        os.remove('qat_yaml.yaml')
        os.remove('dump_yaml.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_get_all_weight_name(self):
        assert len(list(self.nc_model.get_all_weight_names())) == 62

    def test_get_weight(self):
        for name, param in self.model.named_parameters():
            if name == "layer4.1.conv2.weight":
                param.data.fill_(0.0)
            if name == "fc.bias":
                param.data.fill_(0.1)
        assert int(torch.sum(self.nc_model.get_weight("layer4.1.conv2.weight"))) == 0
        assert torch.allclose(
            torch.sum(
                self.nc_model.get_weight("fc.bias")),
            torch.tensor(100.))

    def test_get_input(self):
        model = MODELS['pytorch'](torchvision.models.quantization.resnet18())
        model.model.eval().fuse_model()
        model.register_forward_pre_hook()
        rand_input = torch.rand(100, 3, 224, 224).float()
        model.model(rand_input)
        assert torch.equal(model.get_inputs('x'), rand_input)
        model.remove_hooks()

    def test_update_weights(self):
        self.nc_model.update_weights('fc.bias', torch.zeros([1000]))
        assert int(torch.sum(self.nc_model.get_weight("fc.bias"))) == 0

    def test_get_gradient(self):
        with self.assertRaises(AssertionError):
            self.nc_model.get_gradient('fc.bias')

        for name, tensor in self.nc_model._model.named_parameters():
            if name == 'fc.bias':
                tensor.grad = torch.zeros_like(tensor)
                break
        assert torch.equal(torch.Tensor(self.nc_model.get_gradient('fc.bias')), torch.zeros_like(tensor))

        rand_input = torch.rand(100, 3, 224, 224).float()
        rand_input.grad = torch.ones_like(rand_input)
        assert torch.equal(torch.Tensor(self.nc_model.get_gradient(rand_input)),
                           torch.ones_like(rand_input))

    def test_report_sparsity(self):
        df, total_sparsity = self.nc_model.report_sparsity()
        self.assertTrue(total_sparsity > 0)
        self.assertTrue(len(df) == 22)

    def test_quantization_saved(self):
        for fake_yaml in ['dynamic_yaml.yaml', 'qat_yaml.yaml', 'ptq_yaml.yaml']:
            model = M()
            quantizer = Quantization(fake_yaml)
            quantizer.conf.usr_cfg.tuning.exit_policy['performance_only'] = True
            dataset = quantizer.dataset('dummy', (100, 3, 224, 224), label=True)
            quantizer.model = model
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            q_model = quantizer.fit()
            q_model.save('./saved')
            # Load configure and weights by neural_compressor.utils
            saved_model = load("./saved", model)
            # recover int8 model with only tune_cfg
            tune_cfg_file = './saved/best_configure.yaml'
            with open(tune_cfg_file, 'r') as f:
                history_cfg = yaml.safe_load(f)
            saved_model = load(model=model, \
                                history_cfg=history_cfg)
            eval_func(saved_model)
            shutil.rmtree('./saved', ignore_errors=True)
        from neural_compressor.experimental import Benchmark
        evaluator = Benchmark('ptq_yaml.yaml')
        # Load configure and weights by neural_compressor.model
        evaluator.model = model
        evaluator.b_dataloader = common.DataLoader(dataset)
        evaluator()
        evaluator.model = model
        evaluator()

        for fake_yaml in ['qat_yaml.yaml', 'ptq_yaml.yaml']:
            model = copy.deepcopy(self.model)
            if fake_yaml == 'ptq_yaml.yaml':
                model.eval().fuse_model()
            conf = Quantization_Conf(fake_yaml)
            quantizer = Quantization(conf)
            dataset = quantizer.dataset('dummy', (100, 3, 224, 224))
            quantizer.model = model
            if fake_yaml == 'qat_yaml.yaml':
                quantizer.q_func = q_func
            else:
                quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.eval_func = eval_func
            q_model = quantizer.fit()
            q_model.save('./saved')
            # Load configure and weights by neural_compressor.utils
            saved_model = load("./saved", model)
            # recover int8 model with only tune_cfg
            tune_cfg_file = './saved/best_configure.yaml'
            with open(tune_cfg_file, 'r') as f:
                history_cfg = yaml.safe_load(f)
            saved_model = load(model=model, \
                                history_cfg=history_cfg)
            eval_func(saved_model)
            shutil.rmtree('./saved', ignore_errors=True)

    def test_non_quant_module(self):
        for fake_yaml in ['qat_yaml.yaml', 'ptq_yaml.yaml']:
            model = PartialQuantModel()
            conf = Quantization_Conf(fake_yaml)
            quantizer = Quantization(conf)
            dataset = quantizer.dataset('dummy', (1, 3, 224, 224))
            non_quant_dict = {'non_quant_module_name': ['conv', 'conv1', 'sub.conv'], \
                              'non_quant_module_class': ['BatchNorm2d', 'FP32Model']}
            quantizer.model = common.Model(model, **non_quant_dict)
            if fake_yaml == 'qat_yaml.yaml':
                quantizer.q_func = q_func
            else:
                quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.eval_func = eval_func
            q_model = quantizer.fit()
            q_model.save('./saved')
            saved_model = load("./saved", model, **non_quant_dict)
            eval_func(saved_model)
            shutil.rmtree('./saved', ignore_errors=True)

    def test_tensorboard(self):
        model = copy.deepcopy(self.nc_model)
        model.model.eval().fuse_model()
        quantizer = Quantization('dump_yaml.yaml')
        dataset = quantizer.dataset('dummy', (100, 3, 224, 224), label=True)
        quantizer.model = model.model
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_func = eval_func
        quantizer.fit()
        self.assertTrue(True if os.path.exists('runs/eval/baseline_acc0.0') else False)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.eval_func = None
        quantizer.fit()
        self.assertTrue(True if os.path.exists('runs/eval/baseline_acc0.0') else False)

    def test_tensor_dump_and_set(self):
        model = copy.deepcopy(self.nc_model)
        model.model.eval().fuse_model()
        quantizer = Quantization('ptq_yaml.yaml')
        dataset = quantizer.dataset('dummy', (100, 3, 224, 224), label=True)
        dataloader = common.DataLoader(dataset)
        dataloader = common._generate_common_dataloader(dataloader, 'pytorch')
        quantizer.eval_dataloader = dataloader
        quantizer.calib_dataloader = dataloader
        quantizer.model = model.model
        q_model = quantizer.fit()
        quantizer.strategy.adaptor.inspect_tensor(
            model, dataloader, op_list=['conv1.0', 'layer1.0.conv1.0'],
            iteration_list=[1, 2], inspect_type='all', save_to_disk=True)
        load_array = lambda *a, **k: np.load(*a, allow_pickle=True, **k)
        a = load_array('saved/dump_tensor/activation_iter1.npz')
        w = load_array('saved/dump_tensor/weight.npz')
        if PT_VERSION >= PyTorchVersionMode.PT18.value:
          self.assertTrue(w['conv1.0'].item()['conv1.0.weight'].shape[0] ==
                          a['conv1.0'].item()['conv1.0.output0'].shape[1])
        else:
          self.assertTrue(w['conv1.0'].item()['conv1.0.weight'].shape[0] ==
                          a['conv1.0'].item()['conv1.1.output0'].shape[1])
        data = np.random.random(w['conv1.0'].item()['conv1.0.weight'].shape).astype(np.float32)
        quantizer.strategy.adaptor.set_tensor(q_model, {'conv1.0.weight': data})
        changed_tensor = q_model.get_weight('conv1.weight')
        scales = changed_tensor.q_per_channel_scales()
        changed_tensor_fp32 = torch.dequantize(changed_tensor)
        self.assertTrue(np.allclose(data, changed_tensor_fp32.numpy(), atol=2 / np.min(scales.numpy())))
        quantizer.strategy.adaptor.inspect_tensor(
            q_model, dataloader, op_list=['conv1.0', 'layer1.0.conv1.0'],
            iteration_list=[1, 2], inspect_type='all', save_to_disk=False)

    def test_get_graph_info(self):
        from neural_compressor.adaptor.pytorch import get_ops_recursively
        model = copy.deepcopy(self.model)
        op_map = {}
        get_ops_recursively(model, '', op_map)
        self.assertTrue(op_map['conv1'] == 'Conv2d')

    def test_forward_wrapper(self):
        vision_model = torchvision.models.resnet18()
        class dummymodel(torch.nn.Module):
            def __init__(self, model):
                super(dummymodel, self).__init__()
                self._model = model
            def forward(self,input=None):
                return self._model(input)

        data = [[{'input': torch.rand(3,224,224)}, torch.ones(1,1)], ]
        # dataloader.batch_size=100
        dataloader = common.DataLoader(data, batch_size=1)

        quantizer = Quantization('dynamic_yaml.yaml')
        model = dummymodel(vision_model)
        quantizer.model = model
        quantizer.calib_dataloader = dataloader
        quantizer.eval_dataloader = dataloader
        quantizer.fit()

    def test_floatfunctions_fallback(self):
        class ModelWithFunctionals(torch.nn.Module):
            def __init__(self):
                super(ModelWithFunctionals, self).__init__()
                self.mycat = nnq.FloatFunctional()
                self.myadd = nnq.FloatFunctional()
                self.myadd_relu = nnq.FloatFunctional()
                # Tracing doesnt work yet for c10 ops with scalar inputs
                # https://github.com/pytorch/pytorch/issues/27097
                self.my_scalar_add = nnq.FloatFunctional()
                self.mymul = nnq.FloatFunctional()
                self.my_scalar_mul = nnq.FloatFunctional()
                self.quant = QuantStub()
                self.dequant = DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                y = self.mycat.cat([x, x, x])
                z = self.myadd.add(y, y)
                w = self.myadd_relu.add_relu(z, z)
                # Tracing doesnt work yet for c10 ops with scalar inputs
                # https://github.com/pytorch/pytorch/issues/27097
                w = self.my_scalar_add.add_scalar(w, -0.5)
                w = self.mymul.mul(w, w)
                w = self.my_scalar_mul.mul_scalar(w, 0.5)
                w = self.dequant(w)
                return w

        model = ModelWithFunctionals()
        model = MODELS['pytorch'](model)
        x = torch.rand(10, 1, dtype=torch.float)
        y = model.model(x)
        fallback_ops = []
        q_capability = self.adaptor.query_fw_capability(model)
        for k, v in q_capability["opwise"].items():
            if k[0] != "quant" and k[0] != "dequant":
              fallback_ops.append(k[0])
        model.model.qconfig = torch.quantization.default_qconfig
        model.model.quant.qconfig = torch.quantization.default_qconfig
        if PT_VERSION >= PyTorchVersionMode.PT18.value:
            model.model.dequant.qconfig = torch.quantization.default_qconfig
        nc_torch._fallback_quantizable_ops_recursively(
            model.model, '', fallback_ops, op_qcfgs={})
        torch.quantization.add_observer_(model.model)
        model.model(x)
        torch.quantization.convert(model.model, self.adaptor.q_mapping, inplace=True)
        qy = model.model(x)
        tol = {'atol': 1e-01, 'rtol': 1e-03}
        self.assertTrue(np.allclose(y, qy, **tol))


@unittest.skipIf(not TEST_IPEX, "Unsupport Intel PyTorch Extension")
class TestPytorchIPEXAdaptor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_ipex_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('ipex_yaml.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_tuning_ipex(self):
        from neural_compressor.experimental import Quantization
        model = M()
        quantizer = Quantization('ipex_yaml.yaml')
        quantizer.conf.usr_cfg.tuning.exit_policy['performance_only'] = True
        dataset = quantizer.dataset('dummy', (100, 3, 224, 224), label=True)
        quantizer.model = model
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        nc_model = quantizer.fit()
        nc_model.save('./saved')
        try:
            script_model = torch.jit.script(model.to(ipex.DEVICE))
        except:
            script_model = torch.jit.trace(model.to(ipex.DEVICE), torch.randn(10, 3, 224, 224).to(ipex.DEVICE))
        from neural_compressor.experimental import Benchmark
        evaluator = Benchmark('ipex_yaml.yaml')
        evaluator.model = script_model
        evaluator.b_dataloader = common.DataLoader(dataset)
        results = evaluator()


@unittest.skipIf(not FX_MODE, "Unsupport Fx Mode with PyTorch Version Below 1.8")
class TestPytorchFXAdaptor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_pytorch_fx_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('fx_ptq_yaml.yaml')
        os.remove('fx_dynamic_yaml.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_fx_quant(self):
        for fake_yaml in ['fx_qat_yaml.yaml', 'fx_ptq_yaml.yaml']:
            model_origin = torchvision.models.resnet18()
            # run fx_quant in neural_compressor and save the quantized GraphModule
            quantizer = Quantization(fake_yaml)
            dataset = quantizer.dataset('dummy', (10, 3, 224, 224), label=True)
            quantizer.eval_func = eval_func
            if fake_yaml == 'fx_qat_yaml.yaml':
                quantizer.q_func = q_func
            else:
                quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = common.Model(model_origin,
                            **{'prepare_custom_config_dict': \
                                    {'non_traceable_module_name': ['a']},
                               'convert_custom_config_dict': \
                                    {'preserved_attributes': []}
                              })
            q_model = quantizer.fit()
            q_model.save('./saved')
            # Load configure and weights with neural_compressor.utils
            model_fx = load('./saved', model_origin,
                            **{'prepare_custom_config_dict': \
                                    {'non_traceable_module_name': ['a']},
                               'convert_custom_config_dict': \
                                    {'preserved_attributes': []}
                              })
            self.assertTrue(isinstance(model_fx, torch.fx.graph_module.GraphModule))

            # recover int8 model with only tune_cfg
            history_file = './saved/history.snapshot'
            model_fx_recover = recover(model_origin, history_file, 0,
                            **{'prepare_custom_config_dict': \
                                    {'non_traceable_module_name': ['a']},
                               'convert_custom_config_dict': \
                                    {'preserved_attributes': []}
                              })
            self.assertEqual(model_fx.code, model_fx_recover.code)
            shutil.rmtree('./saved', ignore_errors=True)

        for fake_yaml in ['fx_qat_yaml.yaml', 'fx_ptq_yaml.yaml']:
            model_origin = M()
            # run fx_quant in neural_compressor and save the quantized GraphModule
            quantizer = Quantization(fake_yaml)
            quantizer.conf.usr_cfg.tuning.exit_policy['performance_only'] = True
            dataset = quantizer.dataset('dummy', (10, 3, 224, 224), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.model = common.Model(model_origin,
                            **{'prepare_custom_config_dict': \
                                    {'non_traceable_module_name': ['a']},
                               'convert_custom_config_dict': \
                                    {'preserved_attributes': []}
                              })
            q_model = quantizer.fit()
            q_model.save('./saved')
            # Load configure and weights with neural_compressor.utils
            model_fx = load('./saved', model_origin,
                            **{'prepare_custom_config_dict': \
                                    {'non_traceable_module_name': ['a']},
                               'convert_custom_config_dict': \
                                    {'preserved_attributes': []}
                              })
            self.assertTrue(isinstance(model_fx, torch.fx.graph_module.GraphModule))

    @unittest.skipIf(PT_VERSION < PyTorchVersionMode.PT19.value,
      "Please use PyTroch 1.9 or higher version for dynamic quantization with pytorch_fx backend")
    def test_fx_dynamic_quant(self):
        # Model Definition
        class LSTMModel(nn.Module):
            '''Container module with an encoder, a recurrent module, and a decoder.'''

            def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
                super(LSTMModel, self).__init__()
                self.drop = nn.Dropout(dropout)
                self.encoder = nn.Embedding(ntoken, ninp)
                self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
                self.decoder = nn.Linear(nhid, ntoken)
                self.init_weights()
                self.nhid = nhid
                self.nlayers = nlayers

            def init_weights(self):
                initrange = 0.1
                self.encoder.weight.data.uniform_(-initrange, initrange)
                self.decoder.bias.data.zero_()
                self.decoder.weight.data.uniform_(-initrange, initrange)

            def forward(self, input, hidden):
                emb = self.drop(self.encoder(input))
                output, hidden = self.rnn(emb, hidden)
                output = self.drop(output)
                decoded = self.decoder(output)
                return decoded, hidden

        model = LSTMModel(
            ntoken = 10,
            ninp = 512,
            nhid = 256,
            nlayers = 5,
        )

        # run fx_quant in neural_compressor and save the quantized GraphModule
        model.eval()
        quantizer = Quantization('fx_dynamic_yaml.yaml')
        quantizer.model = common.Model(model,
                            **{'prepare_custom_config_dict': \
                                    {'non_traceable_module_name': ['a']},
                               'convert_custom_config_dict': \
                                    {'preserved_attributes': []}
                              })
        q_model = quantizer.fit()
        q_model.save('./saved')

        # Load configure and weights by neural_compressor.utils
        model_fx = load("./saved", model,
                            **{'prepare_custom_config_dict': \
                                    {'non_traceable_module_name': ['a']},
                               'convert_custom_config_dict': \
                                    {'preserved_attributes': []}
                              })
        self.assertTrue(isinstance(model_fx, torch.fx.graph_module.GraphModule))
        # recover int8 model with only tune_cfg
        history_file = './saved/history.snapshot'
        model_fx_recover = recover(model, history_file, 0,
                            **{'prepare_custom_config_dict': \
                                    {'non_traceable_module_name': ['a']},
                               'convert_custom_config_dict': \
                                    {'preserved_attributes': []}
                              })
        self.assertEqual(model_fx.code, model_fx_recover.code)
        shutil.rmtree('./saved', ignore_errors=True)



if __name__ == "__main__":
    unittest.main()
