import os
import shutil
import unittest
import numpy as np
import torch

from neural_compressor.data import DATASETS
from neural_compressor.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader

def build_fake_yaml(approach=None, search_algorithm=None, metrics=['acc']):
    fake_yaml = """
    model:
        name: imagenet_nas
        framework: pytorch

    nas:
        %s
        search:
            search_space: {'channels': [16, 32, 64], 'dimensions': [32, 64, 128]}
            %s
            %s
            max_trials: 3
    train:
        start_epoch: 0
        end_epoch: 1
        iteration: 10
        optimizer:
            SGD:
                learning_rate: 0.001
        criterion:
            CrossEntropyLoss:
                reduction: sum
        dataloader:
            batch_size: 8
            dataset:
                dummy:
                    shape: [32, 3, 64, 64]
                    label: True
    evaluation:
        accuracy:
            metric:
                topk: 1
            dataloader:
                batch_size: 8
                dataset:
                    dummy:
                        shape: [32, 3, 64, 64]
                        label: True
    """ % (
        'approach: \'{}\''.format(approach) if approach else '',
        'search_algorithm: \'{}\''.format(search_algorithm) if search_algorithm else '',
        'metrics: [{}]'.format(','.join(['\'{}\''.format(m) for m in metrics])) if metrics else ''
    )
    with open('fake.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)

def model_builder(model_arch_params):
    channels = model_arch_params['channels']
    dimensions = model_arch_params['dimensions']
    return ConvNet(channels, dimensions)


class ConvNet(torch.nn.Module):
    def __init__(self, channels, dimensions):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, channels, (3, 3), padding=1)
        self.avg_pooling = torch.nn.AvgPool2d((64, 64))
        self.dense = torch.nn.Linear(channels, dimensions)
        self.out = torch.nn.Linear(dimensions, 1)
        self.activation = torch.nn.Sigmoid()

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.avg_pooling(outputs).squeeze()
        outputs = self.dense(outputs)
        outputs = self.out(outputs)
        outputs = self.activation(outputs)
        return outputs


class TestNAS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        build_fake_yaml()

    @classmethod
    def tearDownClass(cls):
        os.remove('fake.yaml')
        shutil.rmtree(os.path.join(os.getcwd(), 'NASResults'), ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_basic_nas(self):
        from neural_compressor.experimental import common, NAS
        # Built-in train, evaluation
        nas_agent = NAS('fake.yaml')
        nas_agent.model_builder = \
            lambda model_arch_params:common.Model(model_builder(model_arch_params))
        best_model_archs = nas_agent()
        self.assertTrue(len(best_model_archs) > 0)

        # Customized train, evaluation
        datasets = DATASETS('pytorch')
        dummy_dataset = datasets['dummy'](shape=(32, 3, 64, 64), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        def train_func(model):
            epochs = 2
            iters = 10
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                for image, target in dummy_dataloader:
                    print('.', end='')
                    cnt += 1
                    output = model(image).unsqueeze(dim=0)
                    loss = criterion(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if cnt >= iters:
                        break
        def eval_func(model):
            model.eval()
            acc = 0
            for image, target in dummy_dataloader:
                output = model(image).cpu().detach().numpy()
                acc += np.sum(output==target)
            return {'acc': acc / len(dummy_dataset)}

        for search_algorithm in [None, 'grid', 'random', 'bo']:
            print('{fix}Search algorithm: {msg}{fix}'.format(msg=search_algorithm, fix='='*30))
            build_fake_yaml(approach='basic', search_algorithm=search_algorithm, metrics=None)
            search_space = {'channels': [16, 32], 'dimensions': [32]}
            nas_agent = NAS('fake.yaml', search_space=search_space)
            nas_agent.model_builder = model_builder
            nas_agent.train_func = train_func
            nas_agent.eval_func = eval_func
            best_model_archs = nas_agent()
            self.assertTrue(len(best_model_archs) > 0)

if __name__ == "__main__":
    unittest.main()