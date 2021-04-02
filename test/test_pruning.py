import os
import shutil
import unittest

import torch
import torchvision
import torch.nn as nn

from lpot.experimental.data.datasets.dummy_dataset import PyTorchDummyDataset
from lpot.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader

def build_fake_yaml():
    fake_yaml = """
       model:
           name: imagenet_prune
           framework: pytorch                           # possible values are tensorflow, mxnet \
                                                              and pytorch
      
       pruning:
         magnitude:
           prune1:
             weights: ['layer1.0.conv1.weight',  'layer1.0.conv2.weight']
             target_sparsity: 0.3
             start_epoch: 1
         start_epoch: 0
         end_epoch: 4
         frequency: 1
         init_sparsity: 0.05
         target_sparsity: 0.25
       
       evaluation:
         accuracy:
           metric:
             topk: 1                                    # tuning metrics: accuracy
    """
    with open('fake.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)


class TestPruning(unittest.TestCase):

    model = torchvision.models.resnet18()

    @classmethod
    def setUpClass(cls):
        build_fake_yaml()

    @classmethod
    def tearDownClass(cls):
        os.remove('fake.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_pruning(self):
        from lpot.experimental import Pruning, common
        prune = Pruning('fake.yaml')
 
        dummy_dataset = PyTorchDummyDataset([tuple([100, 3, 256, 256])])
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
 
        def training_func_for_lpot(model):
            epochs = 16
            iters = 30
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                prune.on_epoch_begin(nepoch)
                for image, target in dummy_dataloader:
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
        dummy_dataset = PyTorchDummyDataset(tuple([100, 3, 256, 256]), label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        prune.model = common.Model(self.model)
        prune.q_func = training_func_for_lpot
        prune.eval_dataloader = dummy_dataloader
        _ = prune()


if __name__ == "__main__":
    unittest.main()
