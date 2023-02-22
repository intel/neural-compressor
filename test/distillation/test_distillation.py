import copy
import os
import shutil
import unittest
import torch
import torchvision
import torch.nn as nn
import tensorflow as tf
from neural_compressor.adaptor.tf_utils.util import version1_lt_version2
from neural_compressor.config import DistillationConfig, \
    KnowledgeDistillationLossConfig, IntermediateLayersKnowledgeDistillationLossConfig
from neural_compressor.data import Datasets
from neural_compressor.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor.training import prepare_compression

class TestDistillation(unittest.TestCase):

    student_model = torchvision.models.resnet18()
    teacher_model = torchvision.models.resnet34()

    datasets = Datasets('pytorch')
    dummy_dataset = datasets['dummy'](shape=(100, 3, 224, 224), low=0., high=1., label=True)
    dummy_dataloader = PyTorchDataLoader(dummy_dataset)

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_distillation(self):
        criterion = nn.CrossEntropyLoss()
        distillation_criterion_conf = KnowledgeDistillationLossConfig(loss_types=['CE', 'KL'])

        optimizer = torch.optim.SGD(self.student_model.parameters(), lr=0.0001)
        conf = DistillationConfig(self.teacher_model, distillation_criterion_conf)
        compression_manager = prepare_compression(copy.deepcopy(self.student_model), conf)
        model = compression_manager.model

        epochs = 3
        iters = 10
        compression_manager.callbacks.on_train_begin()
        for nepoch in range(epochs):
            model.train()
            cnt = 0
            for image, target in self.dummy_dataloader:
                compression_manager.callbacks.on_step_begin(cnt)
                print('.', end='')
                cnt += 1
                output = model(image)
                loss = criterion(output, target)
                loss = compression_manager.callbacks.on_after_compute_loss(image, output, loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if cnt >= iters:
                    break
            compression_manager.callbacks.on_epoch_end()

        model.save('./saved')
        stat = torch.load('./saved/best_model.pt')
        opt_model = self.student_model.load_state_dict(stat)

    def test_distillation_intermediate_layers(self):
        criterion = nn.CrossEntropyLoss()
        distillation_criterion_conf = IntermediateLayersKnowledgeDistillationLossConfig(
            layer_mappings=[['layer1.0', ],
                            [['layer1.1.conv1', ''], ['layer1.1.conv1', '0']],],
            loss_types=['KL', 'MSE'],
            loss_weights=[0.5, 0.5])

        distillation_criterion_conf.config.IntermediateLayersKnowledgeDistillationLoss.layer_mappings[1][1][-1] = \
                lambda x: x[:, :2,...]
        optimizer = torch.optim.SGD(self.student_model.parameters(), lr=0.0001)
        conf = DistillationConfig(self.teacher_model, distillation_criterion_conf)
        compression_manager = prepare_compression(copy.deepcopy(self.student_model), conf)
        model = compression_manager.model

        epochs = 3
        iters = 10
        compression_manager.callbacks.on_train_begin()
        for nepoch in range(epochs):
            model.train()
            cnt = 0
            for image, target in self.dummy_dataloader:
                compression_manager.callbacks.on_step_begin(cnt)
                print('.', end='')
                cnt += 1
                output = model(image)
                loss = criterion(output, target)
                loss = compression_manager.callbacks.on_after_compute_loss(image, output, loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if cnt >= iters:
                    break
            compression_manager.callbacks.on_epoch_end()

        model.save('./saved')
        stat = torch.load('./saved/best_model.pt')
        opt_model = self.student_model.load_state_dict(stat)

if __name__ == "__main__":
    unittest.main()
