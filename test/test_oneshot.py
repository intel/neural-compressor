import os
import copy
import shutil
import unittest

import torch
import torchvision
import torch.nn as nn
from torch.quantization.quantize_fx import convert_fx, prepare_qat_fx

from neural_compressor.data import DATASETS
from neural_compressor.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor.experimental.scheduler import Scheduler

fake_yaml = """
model:
  name: imagenet_prune
  framework: pytorch

pruning:
  approach:
    weight_compression:
      initial_sparsity: 0.0
      target_sparsity: 0.97
      start_epoch: 0
      end_epoch: 4
      pruners:
        - !Pruner
            start_epoch: 1
            end_epoch: 3
            prune_type: basic_magnitude
            names: ['layer1.0.conv1.weight']

        - !Pruner
            target_sparsity: 0.6
            prune_type: basic_magnitude
            update_frequency: 2
            names: ['layer1.0.conv2.weight']
evaluation:
  accuracy:
    metric:
      topk: 1
"""

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
        batch_size: 30
        dataset:
            dummy:
                shape: [128, 3, 224, 224]
                label: True

evaluation:
  accuracy:
    metric:
      topk: 1
    dataloader:
        batch_size: 30
        dataset:
            dummy:
                shape: [128, 3, 224, 224]
                label: True
tuning:
  accuracy_criterion:
    relative:  0.01
  exit_policy:
    timeout: 0
  random_seed: 9527
"""

def build_fake_yaml():
    with open('fake.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)

def build_fake_yaml2():
    with open('fake2.yaml', 'w', encoding="utf-8") as f:
        f.write(fake2_yaml)
        
def build_fake_yaml3():
    with open('fake3.yaml', 'w', encoding="utf-8") as f:
        f.write(fake3_yaml)

def build_fx_fake_yaml():
    fx_fake_yaml = fake_yaml.replace('pytorch', 'pytorch_fx')
    with open('fx_fake.yaml', 'w', encoding="utf-8") as f:
        f.write(fx_fake_yaml)

def build_fx_fake_yaml2():
    fx_fake2_yaml = fake2_yaml.replace('pytorch', 'pytorch_fx')
    with open('fx_fake2.yaml', 'w', encoding="utf-8") as f:
        f.write(fx_fake2_yaml)

def build_fx_fake_yaml3():
    fx_fake3_yaml = fake3_yaml.replace('pytorch', 'pytorch_fx')
    with open('fx_fake3.yaml', 'w', encoding="utf-8") as f:
        f.write(fx_fake3_yaml)

class TestPruning(unittest.TestCase):

    model = torchvision.models.resnet18()
    q_model = torchvision.models.quantization.resnet18()

    @classmethod
    def setUpClass(cls):
        build_fake_yaml()
        build_fake_yaml2()
        build_fake_yaml3()
        build_fx_fake_yaml()
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

    def test_prune_qat_oneshot(self):
        from neural_compressor.experimental import Pruning, common, Quantization
        datasets = DATASETS('pytorch')
        dummy_dataset = datasets['dummy'](shape=(100, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        q_model = copy.deepcopy(self.q_model)
        q_model.fuse_model()
        prune = Pruning('./fake.yaml')
        quantizer = Quantization('./fake2.yaml')
        scheduler = Scheduler()
        scheduler.model = q_model
        combination = scheduler.combine(prune, quantizer)

        def train_func_for_nc(model):
            epochs = 5
            iters = 3
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
            model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            torch.quantization.prepare_qat(model, inplace=True)
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                combination.on_epoch_begin(nepoch)
                for image, target in dummy_dataloader:
                    combination.on_batch_begin(cnt)
                    print('.', end='')
                    cnt += 1
                    output = model(image)
                    loss = criterion(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    combination.on_batch_end()
                    if cnt >= iters:
                        break
                combination.on_epoch_end()
            combination.post_epoch_end()

        combination.train_func = train_func_for_nc
        combination.eval_dataloader = dummy_dataloader
        combination.train_dataloader = dummy_dataloader
        scheduler.append(combination)
        opt_model = scheduler()
        print(20*'=' + 'test_prune_qat_oneshot' + 20*'=')
        print(opt_model.model)

        conv_weight = opt_model.model.layer1[0].conv1.weight().dequantize()
        self.assertAlmostEqual((conv_weight == 0).sum().item() / conv_weight.numel(),
                               0.97,
                               delta=0.01)
        self.assertEqual(combination.__repr__().lower(), 'combination of pruning,quantization')
    
    def test_distillation_qat_oneshot(self):
        from neural_compressor.experimental import Distillation, common, Quantization
        datasets = DATASETS('pytorch')
        dummy_dataset = datasets['dummy'](shape=(100, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        model = copy.deepcopy(self.model)
        q_model = copy.deepcopy(self.q_model)
        q_model.fuse_model()
        distiller = Distillation('./fake3.yaml')
        quantizer = Quantization('./fake2.yaml')
        scheduler = Scheduler()
        distiller.teacher_model = model
        scheduler.model = q_model
        combination = scheduler.combine(distiller, quantizer)

        def train_func_for_nc(model):
            epochs = 5
            iters = 3
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
            model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            torch.quantization.prepare_qat(model, inplace=True)
            combination.pre_epoch_begin()
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                combination.on_epoch_begin(nepoch)
                for image, target in dummy_dataloader:
                    combination.on_batch_begin(cnt)
                    print('.', end='')
                    cnt += 1
                    output = model(image)
                    combination.on_post_forward(image)
                    loss = criterion(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    combination.on_batch_end()
                    if cnt >= iters:
                        break
                combination.on_epoch_end()
            combination.post_epoch_end()

        combination.train_func = train_func_for_nc
        combination.eval_dataloader = dummy_dataloader
        combination.train_dataloader = dummy_dataloader
        scheduler.append(combination)
        opt_model = scheduler()
        print(20*'=' + 'test_distillation_qat_oneshot' + 20*'=')
        print(opt_model.model)

        self.assertEqual(combination.__repr__().lower(), 'combination of distillation,quantization')
    
    def test_distillation_prune_oneshot(self):
        from neural_compressor.experimental import Distillation, common, Pruning
        datasets = DATASETS('pytorch')
        dummy_dataset = datasets['dummy'](shape=(100, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        distiller = Distillation('./fake3.yaml')
        pruner = Pruning('./fake.yaml')
        scheduler = Scheduler()
        model = copy.deepcopy(self.model)
        distiller.teacher_model = copy.deepcopy(model)
        scheduler.model = model
        combination = scheduler.combine(distiller, pruner)

        def train_func_for_nc(model):
            epochs = 5
            iters = 3
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
            combination.pre_epoch_begin()
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                combination.on_epoch_begin(nepoch)
                for image, target in dummy_dataloader:
                    combination.on_batch_begin(cnt)
                    print('.', end='')
                    cnt += 1
                    output = model(image)
                    combination.on_post_forward(image)
                    loss = criterion(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    combination.on_batch_end()
                    if cnt >= iters:
                        break
                combination.on_epoch_end()
            combination.post_epoch_end()

        combination.train_func = train_func_for_nc
        combination.eval_dataloader = dummy_dataloader
        combination.train_dataloader = dummy_dataloader
        scheduler.append(combination)
        opt_model = scheduler()
        print(20*'=' + 'test_distillation_prune_oneshot' + 20*'=')
        print(opt_model.model)

        conv_weight = opt_model.model.layer1[0].conv1.weight
        self.assertAlmostEqual((conv_weight == 0).sum().item() / conv_weight.numel(),
                               0.97,
                               delta=0.01)
        self.assertEqual(combination.__repr__().lower(), 'combination of distillation,pruning')
    
    def test_prune_qat_distillation_oneshot(self):
        from neural_compressor.experimental import Pruning, common, Quantization, Distillation
        datasets = DATASETS('pytorch')
        dummy_dataset = datasets['dummy'](shape=(100, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        model = copy.deepcopy(self.model)
        q_model = copy.deepcopy(self.q_model)
        q_model.fuse_model()
        prune = Pruning('./fake.yaml')
        quantizer = Quantization('./fake2.yaml')
        distiller = Distillation('./fake3.yaml')
        scheduler = Scheduler()
        distiller.teacher_model = model
        scheduler.model = q_model
        combination = scheduler.combine(prune, quantizer, distiller)

        def train_func_for_nc(model):
            epochs = 5
            iters = 3
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
            model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            torch.quantization.prepare_qat(model, inplace=True)
            combination.pre_epoch_begin()
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                combination.on_epoch_begin(nepoch)
                for image, target in dummy_dataloader:
                    combination.on_batch_begin(cnt)
                    print('.', end='')
                    cnt += 1
                    output = model(image)
                    combination.on_post_forward(image)
                    loss = criterion(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    combination.on_batch_end()
                    if cnt >= iters:
                        break
                combination.on_epoch_end()
            combination.post_epoch_end()

        combination.train_func = train_func_for_nc
        combination.eval_dataloader = dummy_dataloader
        combination.train_dataloader = dummy_dataloader
        scheduler.append(combination)
        opt_model = scheduler()
        print(20*'=' + 'test_prune_qat_distillation_oneshot' + 20*'=')
        print(opt_model.model)

        conv_weight = opt_model.model.layer1[0].conv1.weight().dequantize()
        self.assertAlmostEqual((conv_weight == 0).sum().item() / conv_weight.numel(),
                               0.97,
                               delta=0.01)
        self.assertEqual(combination.__repr__().lower(), 'combination of pruning,quantization,distillation')

    def test_prune_qat_oneshot_fx(self):
        from neural_compressor.experimental import Pruning, common, Quantization
        datasets = DATASETS('pytorch_fx')
        dummy_dataset = datasets['dummy'](shape=(100, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        prune = Pruning('./fx_fake.yaml')
        quantizer = Quantization('./fx_fake2.yaml')
        scheduler = Scheduler()
        model = copy.deepcopy(self.model)
        scheduler.model = model
        combination = scheduler.combine(prune, quantizer)

        def train_func_for_nc(model):
            epochs = 5
            iters = 3
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
            qconfig = {"":torch.quantization.get_default_qat_qconfig('fbgemm')}
            model = prepare_qat_fx(model, qconfig)
            # TODO: For test here, needs to bypass this in fx q_function
            combination.model.model = model
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                combination.on_epoch_begin(nepoch)
                for image, target in dummy_dataloader:
                    combination.on_batch_begin(cnt)
                    print('.', end='')
                    cnt += 1
                    output = model(image)
                    loss = criterion(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    combination.on_batch_end()
                    if cnt >= iters:
                        break
                combination.on_epoch_end()
            model = convert_fx(model)
            return model

        combination.train_func = train_func_for_nc
        combination.eval_dataloader = dummy_dataloader
        combination.train_dataloader = dummy_dataloader
        scheduler.append(combination)
        opt_model = scheduler()
        print(20*'=' + 'test_prune_qat_oneshot_fx' + 20*'=')
        print(opt_model.model)

        conv_weight = dict(opt_model.model.layer1.named_modules())['0'].conv1.weight().dequantize()
        self.assertAlmostEqual((conv_weight == 0).sum().item() / conv_weight.numel(),
                               0.97,
                               delta=0.01)
        self.assertEqual(combination.__repr__().lower(), 'combination of pruning,quantization')

    def test_distillation_qat_oneshot_fx(self):
        from neural_compressor.experimental import Distillation, common, Quantization
        datasets = DATASETS('pytorch_fx')
        dummy_dataset = datasets['dummy'](shape=(100, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        model = copy.deepcopy(self.model)
        distiller = Distillation('./fx_fake3.yaml')
        quantizer = Quantization('./fx_fake2.yaml')
        scheduler = Scheduler()
        distiller.teacher_model = copy.deepcopy(model)
        scheduler.model = model
        combination = scheduler.combine(distiller, quantizer)

        def train_func_for_nc(model):
            epochs = 5
            iters = 3
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
            combination.pre_epoch_begin()
            model = combination.model.model
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                combination.on_epoch_begin(nepoch)
                for image, target in dummy_dataloader:
                    combination.on_batch_begin(cnt)
                    print('.', end='')
                    cnt += 1
                    output = model(image)
                    combination.on_post_forward(image)
                    loss = criterion(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    combination.on_batch_end()
                    if cnt >= iters:
                        break
                combination.on_epoch_end()
            combination.post_epoch_end()

        combination.train_func = train_func_for_nc
        combination.eval_dataloader = dummy_dataloader
        combination.train_dataloader = dummy_dataloader
        scheduler.append(combination)
        opt_model = scheduler()
        print(20*'=' + 'test_distillation_qat_oneshot_fx' + 20*'=')
        print(opt_model.model)

        self.assertEqual(combination.__repr__().lower(), 'combination of distillation,quantization')
    
    def test_distillation_prune_oneshot_fx(self):
        from neural_compressor.experimental import Distillation, common, Pruning
        datasets = DATASETS('pytorch_fx')
        dummy_dataset = datasets['dummy'](shape=(100, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        distiller = Distillation('./fx_fake3.yaml')
        pruner = Pruning('./fx_fake.yaml')
        scheduler = Scheduler()
        model = copy.deepcopy(self.model)
        distiller.teacher_model = copy.deepcopy(model)
        scheduler.model = model
        combination = scheduler.combine(distiller, pruner)

        def train_func_for_nc(model):
            epochs = 5
            iters = 3
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
            combination.pre_epoch_begin()
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                combination.on_epoch_begin(nepoch)
                for image, target in dummy_dataloader:
                    combination.on_batch_begin(cnt)
                    print('.', end='')
                    cnt += 1
                    output = model(image)
                    combination.on_post_forward(image)
                    loss = criterion(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    combination.on_batch_end()
                    if cnt >= iters:
                        break
                combination.on_epoch_end()
            combination.post_epoch_end()

        combination.train_func = train_func_for_nc
        combination.eval_dataloader = dummy_dataloader
        combination.train_dataloader = dummy_dataloader
        scheduler.append(combination)
        opt_model = scheduler()
        print(20*'=' + 'test_distillation_prune_oneshot_fx' + 20*'=')
        print(opt_model.model)

        conv_weight = dict(opt_model.model.layer1.named_modules())['0'].conv1.weight
        self.assertAlmostEqual((conv_weight == 0).sum().item() / conv_weight.numel(),
                               0.97,
                               delta=0.01)
        self.assertEqual(combination.__repr__().lower(), 'combination of distillation,pruning')
    
    def test_prune_qat_distillation_oneshot_fx(self):
        from neural_compressor.experimental import Pruning, common, Quantization, Distillation
        datasets = DATASETS('pytorch_fx')
        dummy_dataset = datasets['dummy'](shape=(100, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        model = copy.deepcopy(self.model)
        prune = Pruning('./fx_fake.yaml')
        quantizer = Quantization('./fx_fake2.yaml')
        distiller = Distillation('./fx_fake3.yaml')
        scheduler = Scheduler()
        distiller.teacher_model = copy.deepcopy(model)
        scheduler.model = model
        combination = scheduler.combine(prune, quantizer, distiller)

        def train_func_for_nc(model):
            epochs = 5
            iters = 3
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
            combination.pre_epoch_begin()
            model = combination.model.model
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                combination.on_epoch_begin(nepoch)
                for image, target in dummy_dataloader:
                    combination.on_batch_begin(cnt)
                    print('.', end='')
                    cnt += 1
                    output = model(image)
                    combination.on_post_forward(image)
                    loss = criterion(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    combination.on_batch_end()
                    if cnt >= iters:
                        break
                combination.on_epoch_end()
            combination.post_epoch_end()

        combination.train_func = train_func_for_nc
        combination.eval_dataloader = dummy_dataloader
        combination.train_dataloader = dummy_dataloader
        scheduler.append(combination)
        opt_model = scheduler()
        print(20*'=' + 'test_prune_qat_distillation_oneshot_fx' + 20*'=')
        print(opt_model.model)

        conv_weight = dict(opt_model.model.layer1.named_modules())['0'].conv1.weight().dequantize()
        self.assertAlmostEqual((conv_weight == 0).sum().item() / conv_weight.numel(),
                               0.97,
                               delta=0.01)
        self.assertEqual(combination.__repr__().lower(), 'combination of pruning,quantization,distillation')


if __name__ == "__main__":
    unittest.main()
