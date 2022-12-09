import os
import copy
import shutil
import unittest

import torch
import torchvision
import torch.nn as nn
import neural_compressor.adaptor.pytorch as nc_torch

from neural_compressor.conf.config import DistillationConf, PruningConf
from neural_compressor.data import DATASETS
from neural_compressor.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor.experimental.scheduler import Scheduler
from neural_compressor.training import fit, prepare
from neural_compressor.utils.pytorch import load
from neural_compressor.utils import logger
from packaging.version import Version

PT_VERSION = nc_torch.get_torch_version()
if PT_VERSION >= Version("1.8.0-rc1"):
    FX_MODE = True
else:
    FX_MODE = False




fake2_yaml = """
model:
  name: imagenet_qat
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
"""

fake3_yaml = """
model:
  name: imagenet_distillation
  framework: pytorch

distillation:
  train:
    optimizer:
      SGD:
        learning_rate: 0.001
        momentum: 0.1
        nesterov: True
        weight_decay: 0.001
    criterion:
      KnowledgeDistillationLoss:
        temperature: 1.0
        loss_types: ['CE', 'KL']
        loss_weights: [0.5, 0.5]
    dataloader:
        batch_size: 1
        dataset:
            dummy:
                shape: [16, 3, 224, 224]
                label: True

evaluation:
  accuracy:
    metric:
      topk: 1
    dataloader:
        batch_size: 1
        dataset:
            dummy:
                shape: [16, 3, 224, 224]
                label: True
tuning:
  accuracy_criterion:
    relative:  0.01
  exit_policy:
    timeout: 0
  random_seed: 9527
"""


def build_fake_yaml2():
    with open('fake2.yaml', 'w', encoding="utf-8") as f:
        f.write(fake2_yaml)

def build_fake_yaml3():
    with open('fake3.yaml', 'w', encoding="utf-8") as f:
        f.write(fake3_yaml)



def build_fx_fake_yaml2():
    fx_fake2_yaml = fake2_yaml.replace('pytorch', 'pytorch_fx')
    with open('fx_fake2.yaml', 'w', encoding="utf-8") as f:
        f.write(fx_fake2_yaml)

def build_fx_fake_yaml3():
    fx_fake3_yaml = fake3_yaml.replace('pytorch', 'pytorch_fx')
    with open('fx_fake3.yaml', 'w', encoding="utf-8") as f:
        f.write(fx_fake3_yaml)


class DynamicControlModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.linear = nn.Linear(224 * 224, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if x.size(1) == 1:
            x = x.view(1, -1)
        x = self.linear(x)
        return x


class TestPruning(unittest.TestCase):

    model = torchvision.models.resnet18()
    q_model = torchvision.models.quantization.resnet18()
    q_model.fuse_model()

    @classmethod
    def setUpClass(cls):

        build_fake_yaml2()
        build_fake_yaml3()
        build_fx_fake_yaml2()
        build_fx_fake_yaml3()

    @classmethod
    def tearDownClass(cls):
        os.remove('fake.yaml')
        os.remove('fake2.yaml')
        os.remove('fake3.yaml')
        os.remove('fx_fake.yaml')
        os.remove('fx_fake2.yaml')
        os.remove('fx_fake3.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)


    def test_distillation_qat_oneshot(self):
        from neural_compressor.experimental import Distillation, Quantization
        datasets = DATASETS('pytorch')
        dummy_dataset = datasets['dummy'](shape=(16, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        model = copy.deepcopy(self.model)
        q_model = copy.deepcopy(self.q_model)
        distiller = Distillation('./fake3.yaml')
        quantizer = Quantization('./fake2.yaml')
        scheduler = Scheduler()
        distiller.teacher_model = model
        scheduler.model = q_model
        combination = scheduler.combine(distiller, quantizer)

        def train_func_for_nc(model):
            epochs = 3
            iters = 3
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
            model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            torch.quantization.prepare_qat(model, inplace=True)
            combination.on_train_begin()
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                combination.on_epoch_begin(nepoch)
                for image, target in dummy_dataloader:
                    combination.on_step_begin(cnt)
                    print('.', end='')
                    cnt += 1
                    output = model(image)
                    loss = criterion(output, target)
                    loss = combination.on_after_compute_loss(image, output, loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    combination.on_step_end()
                    if cnt >= iters:
                        break
                combination.on_epoch_end()
            combination.on_train_end()

        combination.train_func = train_func_for_nc
        combination.eval_dataloader = dummy_dataloader
        combination.train_dataloader = dummy_dataloader
        scheduler.append(combination)
        opt_model = scheduler()
        opt_model.save('./saved')
        logger.info(20*'=' + 'test_distillation_qat_oneshot' + 20*'=')

        self.assertEqual(combination.__repr__().lower(), 'combination of distillation,quantization')
        # reloading int8 model
        reloaded_model = load('./saved', self.q_model)


    @unittest.skipIf(PT_VERSION < Version("1.9.0-rc1"),
      "requires higher version of torch than 1.9.0")
    def test_distillation_qat_oneshot_fx(self):
        from neural_compressor.experimental import Distillation, Quantization
        datasets = DATASETS('pytorch_fx')
        dummy_dataset = datasets['dummy'](shape=(16, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        model = DynamicControlModel()
        distiller = Distillation('./fx_fake3.yaml')
        quantizer = Quantization('./fx_fake2.yaml')
        scheduler = Scheduler()
        distiller.teacher_model = copy.deepcopy(model)
        scheduler.model = model
        combination = scheduler.combine(distiller, quantizer)

        def train_func_for_nc(model):
            epochs = 3
            iters = 3
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
            combination.on_train_begin(dummy_dataloader)
            model = combination.model.model
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                combination.on_epoch_begin(nepoch)
                for image, target in dummy_dataloader:
                    combination.on_step_begin(cnt)
                    print('.', end='')
                    cnt += 1
                    output = model(image)
                    loss = criterion(output, target)
                    loss = combination.on_after_compute_loss(image, output, loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    combination.on_step_end()
                    if cnt >= iters:
                        break
                combination.on_epoch_end()
            combination.on_train_end()
            return model

        combination.train_func = train_func_for_nc
        combination.eval_dataloader = dummy_dataloader
        combination.train_dataloader = dummy_dataloader
        scheduler.append(combination)
        opt_model = scheduler()
        opt_model.save('./saved')
        logger.info(20*'=' + 'test_distillation_qat_oneshot_fx' + 20*'=')

        self.assertEqual(combination.__repr__().lower(), 'combination of distillation,quantization')
        # reloading int8 model
        model = DynamicControlModel()
        reloaded_model = load('./saved', model, dataloader=dummy_dataloader)


if __name__ == "__main__":
    unittest.main()
