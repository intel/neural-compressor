
import torch
import torch.nn.quantized as nnq
from torch.quantization import QuantStub, DeQuantStub
import torchvision
import unittest
import os
from lpot.adaptor import FRAMEWORKS
from lpot.model import MODELS
import lpot.adaptor.pytorch as lpot_torch
from lpot.experimental import Quantization, common
import shutil
import copy
import numpy as np

try:
    import intel_pytorch_extension as ipex
    TEST_IPEX = True
except:
    TEST_IPEX = False


def build_ptq_yaml():
    fake_yaml = '''
        model:
          name: imagenet
          framework: pytorch

        quantization:
          op_wise: {
                 'quant': {
                   'activation':  {'dtype': ['fp32']},
                   'weight': {'dtype': ['fp32']}
                 },
                 'layer1.0.conv1': {
                   'activation': {'dtype': ['uint8'], 'algorithm': ['minmax'], 'granularity': ['per_tensor'], 'scheme':['asym']},
                   'weight':  {'dtype': ['int8'], 'algorithm': ['minmax'], 'granularity': ['per_channel'], 'scheme':['asym']}
                 },
                 'layer2.0.conv1': {
                   'activation': {'dtype': ['uint8'], 'algorithm': ['minmax'], 'granularity': ['per_tensor'], 'scheme':['sym']},
                   'weight':  {'dtype': ['int8'], 'algorithm': ['minmax'], 'granularity': ['per_channel'], 'scheme':['sym']}
                 },
                 'layer3.0.conv1': {
                   'activation':  {'dtype': ['uint8'], 'algorithm': ['kl'], 'granularity': ['per_tensor'], 'scheme':['sym']},
                   'weight': {'dtype': ['int8'], 'algorithm': ['minmax'], 'granularity': ['per_channel'], 'scheme':['sym']}
                 },
                 'layer1.0.add_relu': {
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
    with open('ptq_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)


def build_dynamic_yaml():
    fake_yaml = '''
        model:
          name: imagenet
          framework: pytorch

        quantization:
          approach: post_training_dynamic_quant
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
    with open('dynamic_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)


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


def build_qat_yaml():
    fake_yaml = '''
        model:
          name: imagenet
          framework: pytorch

        quantization:
          approach: quant_aware_training
          op_wise: {
                 'quant': {
                   'activation':  {'dtype': ['fp32']},
                   'weight': {'dtype': ['fp32']}
                 },
                 'layer1.0.conv1': {
                   'activation': {'dtype': ['uint8'], 'algorithm': ['minmax'], 'granularity': ['per_tensor'], 'scheme':['asym']},
                   'weight':  {'dtype': ['int8'], 'algorithm': ['minmax'], 'granularity': ['per_channel'], 'scheme':['asym']}
                 },
                 'layer2.0.conv1': {
                   'activation': {'dtype': ['uint8'], 'algorithm': ['minmax'], 'granularity': ['per_tensor'], 'scheme':['sym']},
                   'weight':  {'dtype': ['int8'], 'algorithm': ['minmax'], 'granularity': ['per_channel'], 'scheme':['sym']}
                 },
                 'layer3.0.conv1': {
                   'activation':  {'dtype': ['uint8'], 'algorithm': ['kl'], 'granularity': ['per_tensor'], 'scheme':['sym']},
                   'weight': {'dtype': ['int8'], 'algorithm': ['minmax'], 'granularity': ['per_channel'], 'scheme':['sym']}
                 },
                 'layer1.0.add_relu': {
                   'activation':  {'dtype': ['fp32']},
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
    with open('qat_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)


def eval_func(model):
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        input = torch.randn(10, 3, 224, 224)
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

    return


@unittest.skipIf(TEST_IPEX, "TODO: Please wait to IPEX + PyTorch1.7 release")
class TestPytorchAdaptor(unittest.TestCase):
    framework_specific_info = {"device": "cpu",
                               "approach": "post_training_static_quant",
                               "random_seed": 1234,
                               "q_dataloader": None}
    framework = "pytorch"
    adaptor = FRAMEWORKS[framework](framework_specific_info)
    model = torchvision.models.quantization.resnet18()
    lpot_model = MODELS['pytorch'](model)

    @classmethod
    def setUpClass(self):
        build_ptq_yaml()
        build_dynamic_yaml()
        build_qat_yaml()
        build_dump_tensors_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('ptq_yaml.yaml')
        os.remove('qat_yaml.yaml')
        os.remove('dump_yaml.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_get_all_weight_name(self):
        assert len(list(self.lpot_model.get_all_weight_names())) == 62

    def test_get_weight(self):
        for name, param in self.model.named_parameters():
            if name == "layer4.1.conv2.weight":
                param.data.fill_(0.0)
            if name == "fc.bias":
                param.data.fill_(0.1)
        assert int(torch.sum(self.lpot_model.get_weight("layer4.1.conv2.weight"))) == 0
        assert torch.allclose(
            torch.sum(
                self.lpot_model.get_weight("fc.bias")),
            torch.tensor(100.))

    def test_update_weights(self):
        self.lpot_model.update_weights("fc.bias", torch.zeros([1000]))
        assert int(torch.sum(self.lpot_model.get_weight("fc.bias"))) == 0

    def test_report_sparsity(self):
        df, total_sparsity = self.lpot_model.report_sparsity()
        self.assertTrue(total_sparsity > 0)
        self.assertTrue(len(df) == 22)

    def test_quantization_saved(self):
        from lpot.utils.pytorch import load

        for fake_yaml in ['dynamic_yaml.yaml', 'qat_yaml.yaml', 'ptq_yaml.yaml']:
            if fake_yaml == 'dynamic_yaml.yaml':
                model = torchvision.models.resnet18()
            else:
                model = copy.deepcopy(self.model)
            if fake_yaml == 'ptq_yaml.yaml':
                model.eval().fuse_model()
            quantizer = Quantization(fake_yaml)
            dataset = quantizer.dataset('dummy', (100, 3, 256, 256), label=True)
            quantizer.model = common.Model(model)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            if fake_yaml == 'qat_yaml.yaml':
                quantizer.q_func = q_func
            q_model = quantizer()
            q_model.save('./saved')
            # Load configure and weights by lpot.utils
            saved_model = load("./saved", model)
            eval_func(saved_model)
        from lpot.experimental import Benchmark
        evaluator = Benchmark('ptq_yaml.yaml')
        # Load configure and weights by lpot.model
        evaluator.model = common.Model(model)
        evaluator.b_dataloader = common.DataLoader(dataset)
        results = evaluator()
        evaluator.model = common.Model(model)
        fp32_results = evaluator()
        self.assertTrue((fp32_results['accuracy'][0] - results['accuracy'][0]) < 0.01)

    def test_tensor_dump(self):
        model = copy.deepcopy(self.lpot_model)
        model.model.eval().fuse_model()
        quantizer = Quantization('dump_yaml.yaml')
        dataset = quantizer.dataset('dummy', (100, 3, 256, 256), label=True)
        quantizer.model = common.Model(model.model)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_func = eval_func
        quantizer()
        self.assertTrue(True if os.path.exists('runs/eval/baseline_acc0.0') else False)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.eval_func = None
        quantizer()
        self.assertTrue(True if os.path.exists('runs/eval/baseline_acc0.0') else False)

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
            if k[0] != "quant":
              fallback_ops.append(k[0])
        model.model.qconfig = torch.quantization.default_qconfig
        model.model.quant.qconfig = torch.quantization.default_qconfig
        lpot_torch._fallback_quantizable_ops_recursively(
            model.model, '', fallback_ops, white_list=self.adaptor.white_list)
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
        from lpot.experimental import Quantization
        model = torchvision.models.resnet18()
        quantizer = Quantization('ipex_yaml.yaml')
        dataset = quantizer.dataset('dummy', (100, 3, 256, 256), label=True)
        quantizer.model = common.Model(model)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        lpot_model = quantizer()
        lpot_model.save("./saved")
        try:
            script_model = torch.jit.script(model.to(ipex.DEVICE))
        except:
            script_model = torch.jit.trace(model.to(ipex.DEVICE), torch.randn(10, 3, 224, 224).to(ipex.DEVICE))
        from lpot.experimental import Benchmark
        evaluator = Benchmark('ipex_yaml.yaml')
        evaluator.model = common.Model(script_model)
        evaluator.b_dataloader = common.DataLoader(dataset)
        results = evaluator()


if __name__ == "__main__":
    unittest.main()
