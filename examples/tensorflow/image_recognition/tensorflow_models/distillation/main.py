#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import random
import datetime
import tensorflow as tf
from neural_compressor.conf.dotdict import DotDict
from neural_compressor.training import prepare_compression
from neural_compressor.config import DistillationConfig, KnowledgeDistillationLossConfig
from neural_compressor.utils import create_obj_from_config
from neural_compressor.adaptor import FRAMEWORKS

model_names = ['mobilenet','densenet201']

parser = argparse.ArgumentParser(description='Tensorflow ImageNet Training')
parser.add_argument('-t', '--topology', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('--teacher', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet50)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--distillation', dest='distillation', action='store_true',
                    help='distillation model on training dataset')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument("--output-model", default=None, help="output path", type=str)
parser.add_argument('--dataset_location', default=None, help='location of dataset', type=str)
args = parser.parse_args()

best_acc1 = 0

default_workspace = './nc_workspace/{}/'.format(
                        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


dataloader_args = {
    'batch_size': args.batch_size,
    'dataset': {"ImageFolder": {'root':args.dataset_location}},
    'transform': {'Resize': {'size': 224, 'interpolation': 'nearest'},
                  'KerasRescale': {'rescale': [127.5, 1]}},
    'filter': None
}
train_dataloader = create_obj_from_config.create_dataloader('tensorflow', dataloader_args)
eval_dataloader = create_obj_from_config.create_dataloader('tensorflow', dataloader_args)


def train(model, adaptor, compression_manager):
    train_cfg = {
        'start_epoch': 0, 
        'end_epoch': 90,
        'iteration': 1000,
        'frequency': 1,
        'dataloader': dataloader_args,
        'criterion': {'KnowledgeDistillationLoss': {'temperature': 1.0, 
                                                    'loss_types': ['CE', 'CE'],
                                                    'loss_weights': [0.5, 0.5]}}, 
        'optimizer': {'SGD': {'learning_rate': 0.001, 'momentum': 0.1, 
                                'weight_decay': 0.001, 'nesterov': True}}, 
    }
    train_cfg = DotDict(train_cfg)
    train_func = create_obj_from_config.create_train_func(
                            'tensorflow', \
                            train_dataloader, \
                            adaptor, \
                            train_cfg, \
                            hooks=compression_manager.callbacks.callbacks_list[0].hooks)
    train_func(model)


def evaluate(model, adaptor):
    eval_cfg = {'accuracy': {'metric': {'topk': 1},
                             'dataloader': dataloader_args}
                }
    eval_cfg = DotDict(eval_cfg)
    eval_func = create_obj_from_config.create_eval_func('tensorflow', \
                                                        eval_dataloader, \
                                                        adaptor, \
                                                        eval_cfg.accuracy.metric)
    return eval_func(model)

def main():
    if args.seed is not None:
        random.seed(args.seed)

    main_worker()


def main_worker():
    framework_specific_info = {
        'device': 'cpu', 'random_seed': 9527, 
        'workspace_path': default_workspace, 
        'q_dataloader': None, 'format': 'default', 
        'backend': 'default', 'inputs': [], 'outputs': []
    }
    adaptor = FRAMEWORKS['tensorflow'](framework_specific_info)

    global best_acc1

    print("=> using pre-trained model '{}'".format(args.topology))
    model = tf.keras.applications.MobileNet(weights='imagenet')

    print("=> using pre-trained teacher model '{}'".format(args.teacher))
    teacher_model = tf.keras.applications.DenseNet201(weights='imagenet')

    if args.distillation:
        distil_loss = KnowledgeDistillationLossConfig()
        conf = DistillationConfig(teacher_model=teacher_model, criterion=distil_loss)
        compression_manager = prepare_compression(model, conf)
        compression_manager.callbacks.on_train_begin()
        model = compression_manager.model

        train(model, adaptor, compression_manager)
        print("Accuracy of the distilled model is ",evaluate(model, adaptor))

        compression_manager.callbacks.on_train_end()
        compression_manager.save(args.output_model)
        return


if __name__ == '__main__':
    main()
