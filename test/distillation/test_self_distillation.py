import os
import shutil
import unittest

import torch
import torch.nn as nn
import torchvision
from neural_compressor.conf.config import DistillationConf
from neural_compressor.data import DATASETS
from neural_compressor.experimental.data.dataloaders.pytorch_dataloader import \
    PyTorchDataLoader


def build_fake_yaml():
    fake_yaml = """
    model:
        name: self_distillation
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
                SelfKnowledgeDistillationLoss:
                    layer_mappings: [
                        [['resblock.1.feature.output', 'resblock.deepst.feature.output'],
                        ['resblock.2.feature.output','resblock.deepst.feature.output']],
                        [['resblock.2.fc','resblock.deepst.fc'],
                        ['resblock.3.fc','resblock.deepst.fc']],
                        [['resblock.1.fc','resblock.deepst.fc'],
                        ['resblock.2.fc','resblock.deepst.fc'],
                        ['resblock.3.fc','resblock.deepst.fc']]
                    ]
                    temperature: 3.0
                    loss_types: ['L2', 'KL', 'CE']
                    loss_weights: [0.5, 0.05, 0.02]
                    add_origin_loss: True
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
    with open("fake.yaml", "w", encoding="utf-8") as f:
        f.write(fake_yaml)


class TestSelfDistillation(unittest.TestCase):

    model = torchvision.models.resnet50()

    @classmethod
    def setUpClass(cls):
        build_fake_yaml()

    @classmethod
    def tearDownClass(cls):
        os.remove("fake.yaml")
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_self_distillation(self):
        from neural_compressor.training import fit, prepare

        datasets = DATASETS("pytorch")
        dummy_dataset = datasets["dummy"](
            shape=(100, 3, 224, 224), low=0.0, high=1.0, label=True
        )
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        conf = DistillationConf("fake.yaml")
        callbacks, model = prepare(conf, model=self.model, teacher_model=self.model)

        def training_func_for_nc(model):
            epochs = 3
            iters = 10
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                callbacks.on_epoch_begin(nepoch)
                for image, target in dummy_dataloader:
                    callbacks.on_step_begin(cnt)
                    print(".", end="")
                    cnt += 1
                    output = model(image)
                    loss = criterion(output, target)
                    outputs_features = dict()
                    outputs_features["resblock.deepst.feature.output"] = torch.randn(
                        128, 1024
                    )
                    outputs_features["resblock.2.feature.output"] = torch.randn(
                        128, 1024
                    )
                    outputs_features["resblock.1.feature.output"] = torch.randn(
                        128, 1024
                    )
                    outputs_features["resblock.deepst.fc"] = torch.randn(128, 100)
                    outputs_features["resblock.3.fc"] = torch.randn(128, 100)
                    outputs_features["resblock.2.fc"] = torch.randn(128, 100)
                    outputs_features["resblock.1.fc"] = torch.randn(128, 100)
                    loss = callbacks.on_after_compute_loss(
                        image, outputs_features, loss, teacher_output=outputs_features
                    )
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
            model, callbacks, train_func=training_func_for_nc, eval_func=eval_func
        )


if __name__ == "__main__":
    unittest.main()
