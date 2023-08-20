import unittest

from neural_compressor.data import Datasets
from neural_compressor import WeightPruningConfig
from neural_compressor.training import prepare_compression
from neural_compressor.data import DataLoader
from neural_compressor.adaptor import FRAMEWORKS
from neural_compressor.conf.dotdict import DotDict
from neural_compressor.utils import create_obj_from_config
from neural_compressor.utils import logger
from neural_compressor.conf.config import default_workspace

class TestPruning(unittest.TestCase):

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
        logger.info(stats)
        logger.info(sparsity)


if __name__ == "__main__":
    unittest.main()
