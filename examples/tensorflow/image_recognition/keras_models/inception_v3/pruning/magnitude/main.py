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
import tensorflow as tf 
from neural_compressor.utils import logger
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

flags.DEFINE_string(
    'dataset_location', None, 'location of dataset.')

flags.DEFINE_integer(
    'start_epoch', 0, 'The start epoch of training process.')

flags.DEFINE_integer(
    'end_epoch', 19, 'The end epoch of training process.')

flags.DEFINE_integer(
    'batch_size', 128, 'The batch size of training process.')

flags.DEFINE_bool(
    'train_distributed', False, 'Whether to perform distributed training.')

flags.DEFINE_bool(
    'evaluation_distributed', False, 'Whether to perform distributed evaluation.')

def generate_model(model_name='InceptionV3'):
    model = getattr(tf.keras.applications, model_name)(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax"
    )
    return model

def get_vgg16_baseline(model_path):
    if FLAGS.train_distributed:
        import horovod.tensorflow as hvd
        hvd.init()
        if hvd.rank() == 0:
            model = generate_model('InceptionV3')
            model.summary()
            model.save(model_path)
    else:
        model = generate_model('InceptionV3')
        model.summary()
        model.save(model_path)
    return model_path

train_dataloader_args = {
    'batch_size': FLAGS.batch_size,
    'last_batch': 'rollover',
    'dataset': {"ImageRecord": {'root':FLAGS.dataset_location}},
    'transform': {'BilinearImagenet': {'height': 299, 'width': 299}},
    'filter': None, 'distributed': FLAGS.train_distributed, 'shuffle': False
}
train_dataloader = create_obj_from_config.create_dataloader('tensorflow', train_dataloader_args)

def train(model, adaptor, compression_manager):
    train_cfg = {
        'epoch': 40,
        'start_epoch': 0,
        'execution_mode': 'eager', 
        'dataloader': train_dataloader_args,
        'postprocess': {'transform': {'LabelShift': 1}},
        'criterion': {'SparseCategoricalCrossentropy': {'reduction': 'sum_over_batch_size', 
                                                        'from_logits': False}}, 
        'optimizer': {'Adam': {'learning_rate': 1e-06, 'beta_1': 0.9, 
                                'beta_2': 0.999, 'epsilon': 1e-07, 'amsgrad': False}}, 
    }
    train_cfg = DotDict(train_cfg)
    train_func = create_obj_from_config.create_train_func(
                            'tensorflow', \
                            train_dataloader, \
                            adaptor, \
                            train_cfg, \
                            hooks=compression_manager.callbacks.callbacks.hooks, \
                            callbacks=compression_manager.callbacks.callbacks.callbacks)
    train_func(model)

def evaluate(model, adaptor):
    eval_dataloader_args = {
        'batch_size': 512,
        'last_batch': 'rollover',
        'dataset': {"ImageRecord": {'root':FLAGS.dataset_location}},
        'transform': {'BilinearImagenet': {'height': 299, 'width': 299}},
        'filter': None, 'distributed': FLAGS.evaluation_distributed, 'shuffle': False
    }
    eval_dataloader = create_obj_from_config.create_dataloader('tensorflow', eval_dataloader_args)
    eval_cfg = {'accuracy': {'metric': {'topk': 1}, 
                             'dataloader': eval_dataloader_args, 
                             'postprocess': {'transform': {'LabelShift': 1}}, 
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
    framework_specific_info = {
        'device': 'cpu', 'random_seed': 9527, 
        'workspace_path': default_workspace, 
        'q_dataloader': None, 'format': 'default', 
        'backend': 'default', 'inputs': [], 'outputs': []
    }
    adaptor = FRAMEWORKS['tensorflow'](framework_specific_info)
    configs = WeightPruningConfig(
        pruning_type='magnitude',
        target_sparsity=0.54,
        start_step=FLAGS.start_epoch,
        end_step=FLAGS.end_epoch
    )
    model = get_vgg16_baseline('./Inception-V3_Model')
    compression_manager = prepare_compression(model=model, confs=configs)
    compression_manager.callbacks.on_train_begin()
    model = compression_manager.model

    train(model, adaptor, compression_manager)
    print("Pruned model score is ",evaluate(model, adaptor))


    compression_manager.callbacks.on_train_end()
    compression_manager.save(FLAGS.output_model)
    stats, sparsity = model.report_sparsity()
    logger.info(stats)
    logger.info(sparsity)
