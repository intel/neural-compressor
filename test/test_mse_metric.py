
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

torch.manual_seed(1)

def build_ptq_yaml():
    fake_yaml = '''
        model:
          name: imagenet
          framework: pytorch

        evaluation:
          accuracy:
            metric:
              MSE:
                compare_label: False
          performance:
            warmup: 5
            iteration: 10

        tuning:
          accuracy_criterion:
            absolute:  100.0
            higher_is_better: False
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
              MSE:
                compare_label: False
          performance:
            warmup: 5
            iteration: 10

        tuning:
          accuracy_criterion:
            absolute:  100.0
            higher_is_better: False
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
              MSE:
                compare_label: False
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


@unittest.skipIf(TEST_IPEX, "TODO: Please wait to IPEX + PyTorch1.7 release")
class TestPytorchAdaptor(unittest.TestCase):
    framework_specific_info = {"device": "cpu",
                               "approach": "post_training_static_quant",
                               "random_seed": 1234,
                               "q_dataloader": None,
                               "workspace_path": './'}
    framework = "pytorch"
    adaptor = FRAMEWORKS[framework](framework_specific_info)
    model = torchvision.models.quantization.resnet18()
    lpot_model = MODELS['pytorch'](model)

    @classmethod
    def setUpClass(self):
        build_ptq_yaml()
        build_dynamic_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('ptq_yaml.yaml')
        os.remove('dynamic_yaml.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_quantization_saved(self):
        from lpot.utils.pytorch import load

        for fake_yaml in ['dynamic_yaml.yaml', 'ptq_yaml.yaml']:
            if fake_yaml == 'dynamic_yaml.yaml':
                model = torchvision.models.quantization.resnet18()
            else:
                model = copy.deepcopy(self.model)
            if fake_yaml == 'ptq_yaml.yaml':
                model.eval().fuse_model()
            quantizer = Quantization(fake_yaml)
            dataset = quantizer.dataset('dummy', (100, 3, 256, 256), label=True)
            quantizer.model = common.Model(model)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            q_model = quantizer()
        self.assertTrue(bool(q_model))

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
