import os
import shutil
import unittest

import torch
import torchvision
import torch.nn as nn
import neural_compressor.adaptor.pytorch as nc_torch

from neural_compressor.data import DATASETS
from neural_compressor.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor.experimental.scheduler import Scheduler
from packaging.version import Version

PT_VERSION = nc_torch.get_torch_version()



def build_fake_yaml3():
    fake_yaml = """
    model:
      name: imagenet_qat
      framework: pytorch

    quantization:
      approach: quant_aware_training
      train:
        start_epoch: 0
        end_epoch: 3
        iteration: 10
        dataloader:
          batch_size: 1
          dataset:
            dummy:
              shape: [16, 3, 224, 224]
              label: True
        optimizer:
          SGD:
            learning_rate: 0.1
            momentum: 0.1
            nesterov: True
            weight_decay: 0.1
        criterion:
          CrossEntropyLoss:
            reduction: sum
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
    with open('fake3.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)


def build_fake_yaml5():
    fake_yaml = """
    model:
      name: imagenet_qat
      framework: pytorch_fx

    quantization:
      approach: quant_aware_training
      train:
        start_epoch: 0
        end_epoch: 3
        iteration: 10
        dataloader:
          batch_size: 1
          dataset:
            dummy:
              shape: [16, 3, 224, 224]
              label: True
        optimizer:
          SGD:
            learning_rate: 0.1
            momentum: 0.1
            nesterov: True
            weight_decay: 0.1
        criterion:
          CrossEntropyLoss:
            reduction: sum
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
    with open('fake5.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)

def build_fake_yaml6():
    fake_yaml = """
    model:
        name: imagenet_distillation
        framework: pytorch

    distillation:
        train:
            start_epoch: 0
            end_epoch: 3
            iteration: 10
            frequency: 1
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
    """
    with open('fake6.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)

class TestPruning(unittest.TestCase):

    model = torchvision.models.resnet18()
    q_model = torchvision.models.quantization.resnet18()
    q_model_teacher = torchvision.models.quantization.resnet50()

    @classmethod
    def setUpClass(cls):
        build_fake_yaml3()
        build_fake_yaml5()
        build_fake_yaml6()

    @classmethod
    def tearDownClass(cls):
        os.remove('fake3.yaml')
        os.remove('fake5.yaml')
        os.remove('fake6.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_scheduler_qat_distillation(self):
        from neural_compressor.experimental import Quantization, common, Distillation
        self.q_model = torchvision.models.quantization.resnet18()
        self.q_model.fuse_model()
        quantizer = Quantization('./fake3.yaml')
        distiller = Distillation('./fake6.yaml')
        scheduler = Scheduler()
        scheduler.model = self.q_model
        distiller.teacher_model = self.q_model_teacher
        scheduler.append(distiller)
        scheduler.append(quantizer)
        opt_model = scheduler.fit()
        opt_model.report_sparsity()
        try:
          conv_weight = opt_model.model.layer1[0].conv1.weight().dequantize()
        except:
          conv_weight = opt_model.model.layer1[0].conv1.weight
        self.assertAlmostEqual((conv_weight == 0).sum().item() / conv_weight.numel(),
                               0.01,
                               delta=0.01)



    def test_combine_qat_distillation(self):
        from neural_compressor.experimental import Quantization, common, Distillation
        self.q_model.fuse_model()
        quantizer = Quantization('./fake3.yaml')
        distiller = Distillation('./fake6.yaml')
        scheduler = Scheduler()
        scheduler.model = self.q_model
        distiller.teacher_model = self.q_model_teacher
        combination = scheduler.combine(distiller, quantizer)
        scheduler.append(combination)
        opt_model = scheduler.fit()
        opt_model.report_sparsity()
        try:
          conv_weight = opt_model.model.layer1[0].conv1.weight().dequantize()
        except:
          conv_weight = opt_model.model.layer1[0].conv1.weight
        self.assertAlmostEqual((conv_weight == 0).sum().item() / conv_weight.numel(),
                               0.01,
                               delta=0.01)
        self.assertEqual(combination.__repr__().lower(), 'combination of distillation,quantization')


if __name__ == "__main__":
    unittest.main()
