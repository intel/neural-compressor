
import numpy as np
import torch
import torchvision
import unittest
import os
from ilit.adaptor import FRAMEWORKS
import shutil


def build_ptq_yaml():
    fake_yaml = '''
        model:
          name: imagenet
          framework: pytorch

        quantization:
          op_wise: {
                 'layer1.0.conv1': {
                   'activation':  {'dtype': ['uint8'], 'algorithm': ['minmax'], 'granularity': ['per_tensor'], 'scheme':['sym']},
                   'weight': {'dtype': ['int8'], 'algorithm': ['kl'], 'granularity': ['per_channel'], 'scheme':['sym']}
                 },
                 'layer2.0.conv1': {
                   'activation':  {'dtype': ['uint8'], 'algorithm': ['minmax'], 'granularity': ['per_tensor'], 'scheme':['asym']},
                   'weight': {'dtype': ['int8'], 'algorithm': ['kl'], 'granularity': ['per_channel'], 'scheme':['asym']}
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
    with open('ptq_yaml.yaml', 'w', encoding="utf-8") as f:
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


class TestAdaptorPytorch(unittest.TestCase):
    framework_specific_info = {'device': "cpu",
                               'approach': "post_training_static_quant",
                               'random_seed': 1234,
                               'q_dataloader': None}
    framework = "pytorch"
    adaptor = FRAMEWORKS[framework](framework_specific_info)
    model = torchvision.models.quantization.resnet18()

    @classmethod
    def setUpClass(self):
        build_ptq_yaml()
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
        assert len(list(self.adaptor.get_all_weight_names(self.model))) == 62

    def test_get_weight(self):
        for name, param in self.model.named_parameters():
            if name == "layer4.1.conv2.weight":
                param.data.fill_(0.0)
            if name == "fc.bias":
                param.data.fill_(0.1)
        assert int(torch.sum(self.adaptor.get_weight(self.model, "layer4.1.conv2.weight"))) == 0
        assert torch.allclose(
            torch.sum(
                self.adaptor.get_weight(
                    self.model,
                    "fc.bias")),
            torch.tensor(100.))

    def test_update_weights(self):
        model = self.adaptor.update_weights(self.model, "fc.bias", torch.zeros([1000]))
        assert int(torch.sum(self.adaptor.get_weight(model, "fc.bias"))) == 0

    def test_quantization_saved(self):
        from ilit import Quantization
        from ilit.utils.pytorch import load
        self.model.eval()
        self.model.fuse_model()
        for fake_yaml in ['ptq_yaml.yaml', 'qat_yaml.yaml']:
            quantizer = Quantization(fake_yaml)
            dataset = quantizer.dataset('dummy', (100, 3, 256, 256), label=True)
            dataloader = quantizer.dataloader(dataset)
            q_model = quantizer(
                self.model,
                q_func=q_func if fake_yaml == 'qat_yaml.yaml' else None,
                q_dataloader=dataloader,
                eval_dataloader=dataloader
            )
            new_model = load(
                os.path.join('./saved', 'best_configure.yaml'),
                os.path.join('./saved', 'best_model_weights.pt'), self.model)
            eval_func(new_model)

    def test_tensor_dump(self):
        from ilit import Quantization
        quantizer = Quantization('dump_yaml.yaml')
        dataset = quantizer.dataset('dummy', (100, 3, 256, 256), label=True)
        dataloader = quantizer.dataloader(dataset)
        quantizer(
            self.model,
            eval_func=eval_func,
            q_dataloader=dataloader,
        )
        self.assertTrue(True if os.path.exists('runs/eval/baseline_acc0.0') else False)
        quantizer(
            self.model,
            eval_dataloader=dataloader,
            q_dataloader=dataloader,
        )
        self.assertTrue(True if os.path.exists('runs/eval/baseline_acc0.0') else False)


if __name__ == "__main__":
    unittest.main()
