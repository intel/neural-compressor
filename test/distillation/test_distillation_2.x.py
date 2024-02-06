import copy
import datetime
import os
import shutil
import unittest

import tensorflow as tf
import torch
import torch.nn as nn
import torchvision

from neural_compressor.adaptor import FRAMEWORKS
from neural_compressor.adaptor.tf_utils.util import version1_lt_version2
from neural_compressor.conf.dotdict import DotDict
from neural_compressor.config import (
    DistillationConfig,
    IntermediateLayersKnowledgeDistillationLossConfig,
    KnowledgeDistillationLossConfig,
)
from neural_compressor.data import Datasets
from neural_compressor.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor.data.dataloaders.tensorflow_dataloader import TensorflowDataLoader
from neural_compressor.training import prepare_compression
from neural_compressor.utils import create_obj_from_config


class TestDistillation(unittest.TestCase):
    student_model = torchvision.models.resnet18()
    teacher_model = torchvision.models.resnet34()

    datasets = Datasets("pytorch")
    dummy_dataset = datasets["dummy"](shape=(100, 3, 224, 224), low=0.0, high=1.0, label=True)
    dummy_dataloader = PyTorchDataLoader(dummy_dataset)

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)
        shutil.rmtree("./nc_workspace", ignore_errors=True)
        shutil.rmtree("./distillation_model", ignore_errors=True)

    def test_distillation(self):
        criterion = nn.CrossEntropyLoss()
        distillation_criterion_conf = KnowledgeDistillationLossConfig(loss_types=["CE", "KL"])

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
                print(".", end="")
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

        model.save("./saved")
        stat = torch.load("./saved/best_model.pt")
        opt_model = self.student_model.load_state_dict(stat)

    def test_distillation_intermediate_layers(self):
        criterion = nn.CrossEntropyLoss()
        distillation_criterion_conf = IntermediateLayersKnowledgeDistillationLossConfig(
            layer_mappings=[
                [
                    "",
                ],
                [
                    "layer1.0",
                ],
                [["layer1.1.conv1", ""], ["layer1.1.conv1", "0"]],
            ],
            loss_types=["L1", "KL", "MSE"],
            loss_weights=[0.5, 0.2, 0.3],
        )

        distillation_criterion_conf.config.IntermediateLayersKnowledgeDistillationLoss.layer_mappings[2][1][-1] = (
            lambda x: x[:, :2, ...]
        )
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
                print(".", end="")
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

        model.save("./saved")
        stat = torch.load("./saved/best_model.pt")
        opt_model = self.student_model.load_state_dict(stat)

    def test_distillation_tf(self):
        tf_datasets = Datasets("tensorflow")
        dummy_dataset = tf_datasets["dummy"](shape=(100, 224, 224, 3), low=0.0, high=1.0, label=True)
        default_workspace = "./nc_workspace/{}/".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        train_dataloader = TensorflowDataLoader(dataset=dummy_dataset, batch_size=100)
        framework_specific_info = {
            "device": "cpu",
            "random_seed": 9527,
            "workspace_path": default_workspace,
            "format": "default",
            "backend": "default",
        }
        adaptor = FRAMEWORKS["tensorflow"](framework_specific_info)
        train_cfg = {
            "start_epoch": 0,
            "end_epoch": 2,
            "iteration": 10,
            "frequency": 1,
            "dataloader": train_dataloader,
            "criterion": {
                "KnowledgeDistillationLoss": {
                    "temperature": 1.0,
                    "loss_types": ["CE", "CE"],
                    "loss_weights": [0.5, 0.5],
                }
            },
            "optimizer": {"SGD": {"learning_rate": 0.001, "momentum": 0.1, "weight_decay": 0.001, "nesterov": True}},
        }
        train_cfg = DotDict(train_cfg)
        model = tf.keras.applications.MobileNet(weights="imagenet")
        teacher_model = tf.keras.applications.DenseNet201(weights="imagenet")
        distil_loss = KnowledgeDistillationLossConfig()
        conf = DistillationConfig(teacher_model=teacher_model, criterion=distil_loss)
        compression_manager = prepare_compression(model, conf)
        compression_manager.callbacks.on_train_begin()
        model = compression_manager.model

        train_func = create_obj_from_config.create_train_func(
            "tensorflow",
            train_dataloader,
            adaptor,
            train_cfg,
            hooks=compression_manager.callbacks.callbacks_list[0].hooks,
        )
        train_func(model)

        compression_manager.callbacks.on_train_end()


if __name__ == "__main__":
    unittest.main()
