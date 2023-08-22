#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from neural_compressor.utils import logger
from neural_compressor.data import DataLoader
from neural_compressor.adaptor import FRAMEWORKS
from neural_compressor.conf.dotdict import DotDict
from neural_compressor.training import WeightPruningConfig
from neural_compressor.training import prepare_compression
from neural_compressor.utils import create_obj_from_config
from neural_compressor.conf.config import default_workspace

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    'output_model', None, 'The output pruned model.')

flags.DEFINE_integer(
    'start_step', 0, 'The start step of pruning process.')

flags.DEFINE_integer(
    'end_step', 9, 'The end step of pruning process.')

flags.DEFINE_bool(
    'train_distributed', False, 'Whether to perform distributed training.')

flags.DEFINE_bool(
    'evaluation_distributed', False, 'Whether to perform distributed evaluation.')

# Prepare dataset
def prepare_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    y_train = tf.keras.utils.to_categorical(y_train, 100)
    y_test = tf.keras.utils.to_categorical(y_test, 100)
    logger.info(f"Training set: x_shape-{x_train.shape}, y_shape-{y_train.shape}")
    logger.info(f"Test set: x_shape-{x_test.shape}, y_shape-{y_test.shape}")
    return TrainDataset(x_train, y_train), EvalDataset(x_test, y_test)

# Build TrainDataset and EvalDataset
class TrainDataset(object):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

class EvalDataset(object):
    def __init__(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

    def __len__(self):
        return len(self.x_test)

    def __getitem__(self, idx):
        return self.x_test[idx], self.y_test[idx]

def train(model, adaptor, compression_manager, train_dataloader):
    train_cfg = {
        'epoch': 15,
        'start_epoch': 0,
        'execution_mode': 'eager', 
        'criterion': {'CrossEntropyLoss': {'reduction': 'sum_over_batch_size', 'from_logits': True}}, 
        'optimizer': {'AdamW': {'learning_rate': 1e-03, 'weight_decay': 1e-04}}, 
    }
    train_cfg = DotDict(train_cfg)
    train_func = create_obj_from_config.create_train_func('tensorflow', \
                            train_dataloader, \
                            adaptor, \
                            train_cfg, \
                            hooks=compression_manager.callbacks.callbacks_list[0].hooks, \
                            callbacks=compression_manager.callbacks.callbacks_list[0])
    train_func(model)

def evaluate(model, adaptor, eval_dataloader):
    eval_cfg = {'accuracy': {'metric': {'topk': 1}, 
                             'iteration': -1, 
                             'multi_metrics': None}
                }
    eval_cfg = DotDict(eval_cfg)
    eval_func = create_obj_from_config.create_eval_func('tensorflow', \
                                                        eval_dataloader, \
                                                        adaptor, \
                                                        eval_cfg.accuracy.metric, \
                                                        eval_cfg.accuracy.postprocess, \
                                                        fp32_baseline = False)
    return eval_func(model)

if __name__ == '__main__':
    training_set, test_set = prepare_dataset()
    train_dataloader = DataLoader(framework='tensorflow', dataset=training_set,
                                        batch_size=128, distributed=FLAGS.train_distributed)
    eval_dataloader = DataLoader(framework='tensorflow', dataset=test_set,
                                        batch_size=256, distributed=FLAGS.evaluation_distributed)

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
        target_sparsity=0.7,
        start_step=FLAGS.start_step,
        end_step=FLAGS.end_step,
        pruning_op_types=['Conv', 'Dense']
    )
    compression_manager = prepare_compression(model='./ViT_Model', confs=configs)
    compression_manager.callbacks.on_train_begin()
    model = compression_manager.model

    train(model, adaptor, compression_manager, train_dataloader)
    print("Pruned model score is ",evaluate(model, adaptor, eval_dataloader))


    compression_manager.callbacks.on_train_end()
    compression_manager.save(FLAGS.output_model)
    stats, sparsity = model.report_sparsity()
    logger.info(stats)
    logger.info(sparsity)