import os
import shutil
import unittest

import torch
import torchvision
import torch.nn as nn
import tensorflow as tf

from neural_compressor.data import DATASETS
from neural_compressor.conf.config import DistillationConf
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

def build_fake_yaml_1():
    fake_yaml = """
    model:
        name: imagenet_distillation
        framework: tensorflow

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
                    loss_types: ['CE', 'CE']
                    loss_weights: [0.5, 0.5]
            dataloader:
                batch_size: 30
                dataset:
                    dummy:
                        shape: [128, 224, 224, 3]
                        label: True
    evaluation:
        accuracy:
            metric:
                topk: 1
            dataloader:
                batch_size: 30
                dataset:
                    dummy:
                        shape: [128, 224, 224, 3]
                        label: True
    """
    with open('fake_1.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)

def build_fake_yaml_2():
    fake_yaml = """
    model:
        name: imagenet_distillation
        framework: pytorch

    distillation:
        train:
            start_epoch: 0
            end_epoch: 3
            iteration: 10
            optimizer:
                SGD:
                    learning_rate: 0.001
                    momentum: 0.1
                    nesterov: True
                    weight_decay: 0.001
            criterion:
                IntermediateLayersKnowledgeDistillationLoss:
                    layer_mappings: [
                        ['layer1.0', 'layer1.0'],
                        ['layer1.1.conv1', '', 'layer1.1.conv1', '0'],
                                        ]
                    loss_types: ['KL', 'MSE']
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
    with open('fake_2.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)

class TestDistillation(unittest.TestCase):

    student_model = torchvision.models.resnet18()
    teacher_model = torchvision.models.resnet34()

    student_model_tf = tf.keras.applications.mobilenet.MobileNet()
    teacher_model_tf = tf.keras.applications.mobilenet_v2.MobileNetV2()

    @classmethod
    def setUpClass(cls):
        build_fake_yaml()
        build_fake_yaml_1()
        build_fake_yaml_2()

    @classmethod
    def tearDownClass(cls):
        os.remove('fake.yaml')
        os.remove('fake_1.yaml')
        os.remove('fake_2.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_distillation(self):
        from neural_compressor.experimental import Distillation, common
        from neural_compressor.conf.config import DistillationConf
        conf = DistillationConf('fake.yaml')
        distiller = Distillation(conf)
        distiller = Distillation()

        from neural_compressor import conf
        conf.model.framework = 'pytorch'
        conf.distillation.train.end_epoch = 3
        conf.distillation.train.iteration = 10
        conf.distillation.train.optimizer = {
            'SGD': {'learning_rate': 0.001, 'momentum': 0.1, 'nesterov': True, 'weight_decay': 0.001}}
        conf.distillation.train.dataloader.batch_size = 30
        conf.distillation.train.dataloader.dataset = {'dummy': {'shape': [128, 3, 224, 224], 'label': True}}
        conf.evaluation.accuracy.dataloader.batch_size = 30
        conf.evaluation.accuracy.dataloader.dataset = {'dummy': {'shape': [128, 3, 224, 224], 'label': True}}
        distiller = Distillation(conf)
        distiller.student_model = self.student_model
        distiller.teacher_model = self.teacher_model
        print('student model: {}'.format(distiller.student_model))
        distilled_model = distiller.fit()
        distilled_model.save('./saved')
        stat = torch.load('./saved/best_model.pt')
        self.student_model.load_state_dict(stat)

    def test_distillation_intermediate_layers(self):
        from neural_compressor.experimental import Distillation, common
        from neural_compressor.conf.config import DistillationConf
        conf = DistillationConf('fake_2.yaml')
        conf.usr_cfg.distillation.train.criterion.\
            IntermediateLayersKnowledgeDistillationLoss.layer_mappings[1][-1] = \
                lambda x: x[:, :2,...]
        distiller = Distillation(conf)
        distiller.student_model = self.student_model
        distiller.teacher_model = self.teacher_model
        print('student model: {}'.format(distiller.student_model))
        _ = distiller.fit()

    def test_distillation_external(self):
        from neural_compressor.experimental.common.criterion import \
            TensorflowKnowledgeDistillationLossExternal

        criterion = TensorflowKnowledgeDistillationLossExternal()
        criterion.teacher_model_forward(None)
        y_true = [[0, 1, 0]]
        y_pred = [[0.05, 0.95, 0]]
        criterion.teacher_student_loss_cal(y_pred, y_true)
        criterion.student_targets_loss_cal(y_pred, y_true)

    def test_distillation_external_new_API(self):
        from neural_compressor.training import prepare, fit
        datasets = DATASETS('pytorch')
        dummy_dataset = datasets['dummy'](shape=(100, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.student_model.parameters(), lr=0.0001)
        conf = DistillationConf('fake.yaml')
        callbacks, model = prepare(
            conf, model=self.student_model, teacher_model=self.teacher_model
        )

        def training_func_for_nc(model):
            epochs = 3
            iters = 10
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                callbacks.on_epoch_begin(nepoch)
                for image, target in dummy_dataloader:
                    callbacks.on_step_begin(cnt)
                    print('.', end='')
                    cnt += 1
                    output = model(image)
                    loss = criterion(output, target)
                    loss = callbacks.on_after_compute_loss(image, output, loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    callbacks.on_step_end()
                    if cnt >= iters:
                        break
                callbacks.on_epoch_end()
            return model

        def eval_func(model):
            for image, target in dummy_dataloader:
                model(image)
            return 1  # metric is 1 here, just for unit test

        model = fit(
            model, callbacks, train_func=training_func_for_nc,
            eval_func=eval_func
        )

    @unittest.skipIf(tf.version.VERSION < '2.3.0', " keras requires higher version than tf-2.3.0")
    def test_tf_distillation(self):
        from neural_compressor.experimental import Distillation
        from neural_compressor.conf.config import DistillationConf
        conf = DistillationConf('fake_1.yaml')
        distiller = Distillation(conf)
        distiller = Distillation('fake_1.yaml')
        distiller.student_model = self.student_model_tf
        distiller.teacher_model = self.teacher_model_tf
        print('student model: {}'.format(distiller.student_model))
        _ = distiller.fit()

if __name__ == "__main__":
    unittest.main()
