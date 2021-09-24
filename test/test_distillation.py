import os
import shutil
import unittest

import torch
import torchvision
import torch.nn as nn

from neural_compressor.data import DATASETS
from neural_compressor.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader

def build_fake_yaml():
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
    """
    with open('fake.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)

class TestDistillation(unittest.TestCase):

    student_model = torchvision.models.resnet18()
    teacher_model = torchvision.models.resnet34()

    @classmethod
    def setUpClass(cls):
        build_fake_yaml()

    @classmethod
    def tearDownClass(cls):
        os.remove('fake.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_distillation(self):
        from neural_compressor.experimental import Distillation, common
        from neural_compressor.conf.config import Distillation_Conf
        conf = Distillation_Conf('fake.yaml')
        distiller = Distillation(conf)
        distiller = Distillation('fake.yaml')
        distiller.student_model = common.Model(self.student_model)
        distiller.teacher_model = common.Model(self.teacher_model)
        print('student model: {}'.format(distiller.student_model))
        _ = distiller()

    def test_pruning_external(self):
        import tensorflow as tf
        from neural_compressor.experimental import Distillation
        from neural_compressor.utils.create_obj_from_config import create_train_func_for_distillation
        from neural_compressor.experimental.common.criterion import PyTorchKnowledgeDistillationLoss, \
            TensorflowKnowledgeDistillationLoss
        distiller = Distillation('fake.yaml')
        datasets = DATASETS('pytorch')
        dummy_dataset = datasets['dummy'](shape=(100, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)

        criterion = TensorflowKnowledgeDistillationLoss()
        criterion.teacher_model_forward(None)
        y_true = [[0, 1, 0]]
        y_pred = [[0.05, 0.95, 0]]
        criterion.teacher_student_loss_cal(y_true, y_pred)
        criterion.student_targets_loss_cal(y_true, y_pred)
        criterion = PyTorchKnowledgeDistillationLoss(loss_weights=[0, 1])
        criterion.teacher_model = self.teacher_model
        optimizer = torch.optim.SGD(self.student_model.parameters(), lr=0.0001)

        def training_func_for_nc(model):
            epochs = 3
            iters = 10
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                distiller.on_epoch_begin(nepoch)
                for image, target in dummy_dataloader:
                    distiller.on_batch_begin(cnt)
                    print('.', end='')
                    cnt += 1
                    output = model(image)
                    distiller.on_post_forward(image)
                    criterion.teacher_model_forward(image)
                    loss = criterion(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    distiller.on_batch_end()
                    if cnt >= iters:
                        break
                distiller.on_epoch_end()
        distiller.student_model = self.student_model
        distiller.teacher_model = self.teacher_model
        distiller.criterion = criterion
        distiller.optimizer = optimizer
        distiller.train_func = training_func_for_nc
        distiller.eval_dataloader = dummy_dataloader
        distiller.train_dataloader = dummy_dataloader
        _ = distiller()
        distiller.train_func = create_train_func_for_distillation(
                                                distiller.framework, \
                                                distiller.train_dataloader, \
                                                distiller.adaptor, \
                                                distiller.cfg.distillation.train, \
                                                hooks=distiller.hooks)

if __name__ == "__main__":
    unittest.main()
