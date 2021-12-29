#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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
import os
import logging
import random
import shutil
import warnings
import tensorflow as tf
from neural_compressor.utils import logger
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
parser.add_argument("--config", default=None, help="tuning config")
parser.add_argument("--output-model", default=None, help="output path", type=str)

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    main_worker(args)


def main_worker(args):
    global best_acc1

    print("=> using pre-trained model '{}'".format(args.topology))
    model = tf.keras.applications.MobileNet(weights='imagenet')
    
    print("=> using pre-trained teacher model '{}'".format(args.teacher))
    teacher_model = tf.keras.applications.DenseNet201(weights='imagenet')

    if args.distillation:
        from neural_compressor.experimental import Distillation
        distiller = Distillation(args.config)

        distiller.model = model
        distiller.teacher_model = teacher_model
        model = distiller.fit()
        model.save(args.output_model)
        return




if __name__ == '__main__':
    main()
