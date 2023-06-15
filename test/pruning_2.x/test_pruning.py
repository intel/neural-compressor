import unittest

import torch
import torchvision
import torch.nn as nn
import sys
sys.path.insert(0, './')
from neural_compressor.data import Datasets
from neural_compressor.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor import WeightPruningConfig
from neural_compressor.training import prepare_compression
from neural_compressor.data import DataLoader
from neural_compressor.adaptor import FRAMEWORKS
from neural_compressor.conf.dotdict import DotDict
from neural_compressor.utils import create_obj_from_config
from neural_compressor.conf.config import default_workspace

class TestPruning(unittest.TestCase):
    model = torchvision.models.resnet18()

    def test_pruning_basic(self):
        local_configs = [
            {
                "op_names": ['layer1.*'],
                'target_sparsity': 0.5,
                "pattern": '8x2',
                "pruning_type": "magnitude_progressive",
                "false_key": "this is to test unsupport keys"
            },
            {
                "op_names": ['layer2.*'],
                'target_sparsity': 0.5,
                'pattern': '2:4'
            },
            {
                "op_names": ['layer3.*'],
                'target_sparsity': 0.7,
                'pattern': '5x1',
                "pruning_type": "snip_progressive"
            }
        ]
        config = WeightPruningConfig(
            local_configs,
            target_sparsity=0.8,
            start_step=1,
            end_step=10
        )
        compression_manager = prepare_compression(model=self.model, confs=config)
        compression_manager.callbacks.on_train_begin()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        datasets = Datasets('pytorch')
        dummy_dataset = datasets['dummy'](shape=(10, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)

        compression_manager.callbacks.on_train_begin()
        for epoch in range(2):
            self.model.train()
            compression_manager.callbacks.on_epoch_begin(epoch)
            local_step = 0
            for image, target in dummy_dataloader:
                compression_manager.callbacks.on_step_begin(local_step)
                output = self.model(image)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                compression_manager.callbacks.on_before_optimizer_step()
                optimizer.step()
                compression_manager.callbacks.on_after_optimizer_step()
                compression_manager.callbacks.on_step_end()
                local_step += 1

            compression_manager.callbacks.on_epoch_end()
        compression_manager.callbacks.on_train_end()
        compression_manager.callbacks.on_before_eval()
        compression_manager.callbacks.on_after_eval()

    def test_pruning_keras(self):
        import tensorflow as tf
        model = tf.keras.applications.ResNet50V2(weights='imagenet')
        def train(model, adaptor, compression_manager, train_dataloader):
            train_cfg = {
                'epoch': 1,
                'start_epoch': 0,
                'execution_mode': 'eager', 
                'criterion': {'SparseCategoricalCrossentropy': {'reduction': 'sum_over_batch_size'}}, 
                'optimizer': {'SGD': {'learning_rate': 1e-03, 'momentum': 0.9, 'nesterov': True}}, 
            }
            train_cfg = DotDict(train_cfg)
            train_func = create_obj_from_config.create_train_func(
                                    'tensorflow', \
                                    train_dataloader, \
                                    adaptor, \
                                    train_cfg, \
                                    hooks=compression_manager.callbacks.callbacks_list[0].hooks, \
                                    callbacks=compression_manager.callbacks.callbacks_list[0])
            train_func(model)

        tf_datasets = Datasets('tensorflow')
        dummy_dataset = tf_datasets['dummy'](shape=(100, 224, 224, 3), low=0., high=1., label=True)
        train_dataloader = DataLoader(dataset=dummy_dataset, batch_size=32, 
                            framework='tensorflow', distributed=False)

        framework_specific_info = {
            'device': 'cpu', 'random_seed': 9527, 
            'workspace_path': default_workspace, 
            'q_dataloader': None, 'format': 'default', 
            'backend': 'default', 'inputs': [], 'outputs': []
        }
        adaptor = FRAMEWORKS['keras'](framework_specific_info)

        configs = WeightPruningConfig(
            backend='itex',
            pruning_type='magnitude',
            pattern='3x1',
            target_sparsity=0.5,
            start_step=1,
            end_step=10,
            pruning_op_types=['Conv', 'Dense']
        )
        compression_manager = prepare_compression(model, confs=configs)
        compression_manager.callbacks.on_train_begin()
        model = compression_manager.model

        train(model, adaptor, compression_manager, train_dataloader)

        compression_manager.callbacks.on_train_end()
        stats, sparsity = model.report_sparsity()


if __name__ == "__main__":
    unittest.main()
