from multiprocessing.spawn import import_main_path
import os
import shutil
import unittest
import numpy as np
import torch

from neural_compressor.conf.config import NASConfig
from neural_compressor.data import DATASETS
from neural_compressor.experimental import common, NAS
from neural_compressor.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor.experimental.nas.dynas import DyNAS

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

def build_dynas_fake_yaml():
    fake_yaml = """
    model:
        name: imagenet_nas
        framework: pytorch

    nas:
        approach: dynas
        search:
            search_algorithm: nsga2
        dynas:
            supernet: ofa_resnet50
            metrics: ['acc', 'macs']
            results_csv_path: './search_results.csv'
    """
    with open('dynas_fake.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)

def build_dynas_results_csv():
    results_csv = """
Sub-network,Date,Latency (ms), MACs,Top-1 Acc (%)
"{'wid': None, 'ks': [7, 7, 3, 3, 5, 7, 7, 3, 5, 5, 3, 3, 7, 3, 5, 5, 5, 7, 5, 7], 'e': [3, 4, 4, 4, 4, 6, 6, 4, 4, 3, 4, 4, 3, 6, 4, 3, 4, 6, 3, 3], 'd': [2, 4, 4, 2, 3], 'r': [224]}",2022-07-07 03:13:06.306540,39,391813792,77.416
"{'wid': None, 'ks': [3, 5, 5, 7, 5, 5, 3, 3, 7, 7, 7, 5, 7, 3, 7, 5, 3, 5, 3, 3], 'e': [4, 6, 3, 4, 4, 4, 4, 6, 3, 6, 4, 3, 4, 3, 4, 3, 6, 4, 4, 6], 'd': [4, 3, 3, 2, 3], 'r': [224]}",2022-07-07 03:14:50.398553,41,412962768,77.234
"{'wid': None, 'ks': [5, 5, 5, 3, 7, 5, 7, 5, 7, 3, 3, 7, 7, 5, 7, 3, 5, 5, 7, 3], 'e': [6, 4, 3, 3, 3, 3, 4, 4, 3, 4, 3, 6, 4, 4, 3, 6, 4, 3, 4, 6], 'd': [4, 4, 4, 2, 4], 'r': [224]}",2022-07-07 03:16:53.105436,44,444295456,77.632
"{'wid': None, 'ks': [3, 5, 3, 7, 3, 5, 7, 5, 3, 3, 3, 7, 3, 5, 3, 5, 3, 3, 7, 3], 'e': [4, 6, 3, 3, 6, 3, 3, 6, 6, 4, 4, 6, 3, 4, 3, 6, 3, 6, 3, 4], 'd': [4, 4, 2, 2, 4], 'r': [224]}",2022-07-07 03:18:47.301137,41,410969240,76.79
"{'wid': None, 'ks': [3, 3, 3, 3, 7, 5, 3, 5, 3, 5, 5, 7, 7, 7, 3, 5, 7, 5, 3, 7], 'e': [3, 6, 6, 4, 6, 3, 3, 4, 3, 6, 3, 4, 4, 6, 3, 6, 4, 3, 6, 3], 'd': [2, 3, 4, 4, 2], 'r': [224]}",2022-07-07 03:20:35.391443,40,405868672,77.338
"{'wid': None, 'ks': [3, 3, 3, 7, 5, 7, 7, 3, 3, 3, 3, 5, 7, 3, 7, 5, 3, 7, 5, 5], 'e': [4, 6, 3, 6, 4, 3, 3, 6, 3, 6, 4, 6, 4, 4, 3, 6, 4, 3, 4, 4], 'd': [3, 4, 4, 2, 2], 'r': [224]}",2022-07-07 03:22:14.504855,37,370501152,76.448
"{'wid': None, 'ks': [7, 5, 3, 5, 7, 5, 3, 3, 5, 3, 3, 7, 7, 3, 5, 3, 3, 5, 5, 7], 'e': [3, 3, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 4, 3, 6, 3, 3, 3, 4], 'd': [4, 4, 3, 4, 2], 'r': [224]}",2022-07-07 03:24:12.500905,48,482299704,77.7
"{'wid': None, 'ks': [7, 3, 5, 7, 5, 5, 7, 5, 3, 3, 3, 5, 5, 3, 7, 5, 5, 7, 3, 7], 'e': [3, 6, 4, 6, 6, 3, 3, 3, 6, 3, 6, 4, 4, 6, 4, 4, 4, 4, 6, 6], 'd': [4, 4, 2, 2, 2], 'r': [224]}",2022-07-07 03:25:50.198665,42,423721952,76.506
"{'wid': None, 'ks': [7, 7, 3, 7, 5, 7, 5, 5, 5, 3, 5, 3, 3, 7, 3, 5, 3, 7, 7, 3], 'e': [3, 3, 3, 4, 4, 3, 4, 4, 4, 4, 4, 6, 6, 4, 3, 3, 3, 6, 3, 4], 'd': [4, 2, 2, 3, 3], 'r': [224]}",2022-07-07 03:27:26.901886,37,373770104,77.258
"{'wid': None, 'ks': [3, 7, 5, 5, 7, 3, 5, 3, 5, 5, 5, 3, 5, 5, 3, 5, 7, 3, 7, 5], 'e': [3, 4, 6, 6, 4, 3, 6, 6, 6, 3, 3, 3, 3, 6, 3, 6, 6, 3, 6, 3], 'd': [3, 2, 3, 2, 3], 'r': [224]}",2022-07-07 03:29:00.989578,36,369186480,77.096
"{'wid': None, 'ks': [7, 7, 5, 5, 7, 5, 3, 3, 3, 5, 7, 3, 7, 7, 5, 5, 3, 7, 3, 7], 'e': [6, 3, 6, 3, 4, 3, 3, 3, 4, 3, 6, 4, 3, 3, 6, 4, 4, 3, 4, 3], 'd': [4, 4, 3, 4, 4], 'r': [224]}",2022-07-07 03:31:07.608402,51,518341312,78.104
    """
    with open('search_results.csv', 'w', encoding="utf-8") as f:
        f.write(results_csv)

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
        build_dynas_fake_yaml()
        build_dynas_results_csv()

    @classmethod
    def tearDownClass(cls):
        os.remove('fake.yaml')
        os.remove('dynas_fake.yaml')
        os.remove('search_results.csv')
        shutil.rmtree(os.path.join(os.getcwd(), 'NASResults'), ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_basic_nas(self):
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

        for approach, search_algorithm in [(None, None), ('basic', 'grid'), ('basic', 'random'), ('basic', 'bo')]:
            print('{fix}Search algorithm: {msg}{fix}'.format(msg=search_algorithm, fix='='*30))
            search_space = {'channels': [16, 32], 'dimensions': [32]}
            nas_config = NASConfig(approach=approach, search_space=search_space, search_algorithm=search_algorithm)
            nas_config.usr_cfg.model.framework = 'pytorch'
            nas_agent = NAS(nas_config)
            nas_agent.model_builder = model_builder
            nas_agent.train_func = train_func
            nas_agent.eval_func = eval_func
            best_model_archs = nas_agent()
            self.assertTrue(len(best_model_archs) > 0)

    def test_dynas(self):
        nas_agent = NAS('dynas_fake.yaml')
        for search_algorithm, supernet in [('nsga2','ofa_mbv3_d234_e346_k357_w1.2'), ('age', 'ofa_mbv3_d234_e346_k357_w1.2')]:
            config = NASConfig(approach='dynas', search_algorithm=search_algorithm)
            config.dynas.supernet = supernet
            config.seed = 42
            config.dynas.metrics = ['acc', 'macs', 'lat']
            config.dynas.population = 10
            config.dynas.num_evals = 10
            config.dynas.results_csv_path = 'search_results.csv'
            config.dynas.batch_size = 64
            nas_agent = NAS(config)
            best_model_archs = nas_agent.search()
        nas_agent.acc_predictor.get_parameters()
        nas_agent.acc_predictor.save('tmp.pickle')
        nas_agent.acc_predictor.load('tmp.pickle')
        samples = nas_agent.supernet_manager.random_samples(10)
        subnet_cfg = nas_agent.supernet_manager.translate2param(samples[0])
        nas_agent.runner_validate.validate_macs(subnet_cfg)
        nas_agent.runner_validate.measure_latency(subnet_cfg)
        nas_agent.validation_interface.clear_csv()
        os.remove('tmp.pickle')
        from neural_compressor.experimental.nas.dynast.dynas_utils import TorchVisionReference
        reference = TorchVisionReference('ofa_resnet50_ofa_mbv3', dataset_path=None, batch_size=1)
        reference.validate_macs()
        reference.measure_latency()
        self.assertTrue(len(best_model_archs) > 0)


if __name__ == "__main__":
    unittest.main()
