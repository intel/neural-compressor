import os
import shutil
import unittest

import torch
import torchvision
import torch.nn as nn

from neural_compressor.experimental.data.datasets.dummy_dataset import DummyDataset
from neural_compressor.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader

def build_fake_yaml():
    fake_yaml = """
    model:
      name: imagenet_prune
      framework: pytorch

    pruning:
      approach:
        weight_compression:
          start_epoch: 0
          pruners:
            - !Pruner
                prune_type: pattern_lock
                names: ['layer1.0.conv1.weight']
    evaluation:
      accuracy:
        metric:
          topk: 1
    """
    with open('fake.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)


class TestPatternLock(unittest.TestCase):
    model = torchvision.models.resnet18()

    @classmethod
    def setUpClass(cls):
        build_fake_yaml()

    @classmethod
    def tearDownClass(cls):
        os.remove('fake.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_pattern_lock(self):
        from neural_compressor.experimental import Pruning, common
        prune = Pruning('fake.yaml')

        weight = self.model.layer1[0].conv1.weight
        mask = torch.ones(weight.numel())
        mask[:round(weight.numel()*0.9)] = .0
        mask = mask[torch.randperm(mask.numel())].view(weight.shape)
        weight.data = weight * mask

        self.assertTrue(self.model.layer1[0].conv1.weight.ne(0).eq(mask).all())

        dummy_dataset = DummyDataset([tuple([100, 3, 256, 256])])
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)

        def training_func_for_nc(model):
            epochs = 2
            iters = 30
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                prune.on_epoch_begin(nepoch)
                for i, (image, target) in enumerate(dummy_dataloader):
                    prune.on_batch_begin(cnt)
                    print('.', end='')
                    cnt += 1
                    output = model(image)
                    loss = criterion(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    prune.on_batch_end()
                    if cnt >= iters:
                        break
                prune.on_epoch_end()
        dummy_dataset = DummyDataset(tuple([100, 3, 256, 256]), label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        prune.model = common.Model(self.model)
        prune.pruning_func = training_func_for_nc
        prune.eval_dataloader = dummy_dataloader
        prune.train_dataloader = dummy_dataloader
        _ = prune()
        self.assertTrue(self.model.layer1[0].conv1.weight.ne(0).eq(mask).all())


if __name__ == "__main__":
    unittest.main()
