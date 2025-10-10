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
from __future__ import print_function
import yaml
import numpy as np
import tensorflow

from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from  tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from  tensorflow.keras.callbacks import LearningRateScheduler
from  tensorflow.keras.callbacks import ReduceLROnPlateau
from  tensorflow.keras.regularizers import l2
from  tensorflow.keras.models import Model
from  tensorflow.keras.datasets import cifar10


from neural_compressor.utils import logger
from neural_compressor.data import DataLoader
from neural_compressor.adaptor import FRAMEWORKS
from neural_compressor.conf.dotdict import DotDict
from neural_compressor.training import WeightPruningConfig
from neural_compressor.training import prepare_compression
from neural_compressor.utils import create_obj_from_config
from neural_compressor.conf.config import default_workspace

flags = tensorflow.compat.v1.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_bool(
    'prune', False, 'Whether to perform distributed training.')

flags.DEFINE_bool(
    'benchmark', False, 'Whether to perform distributed evaluation.')

flags.DEFINE_string(
    'input_model', None, 'Run inference with specified model.')

flags.DEFINE_string(
    'output_model', None, 'The output pruned model.')

flags.DEFINE_integer(
    'start_step', 0, 'The start step of pruning process.')

flags.DEFINE_integer(
    'end_step', 7, 'The end step of pruning process.')

flags.DEFINE_integer(
    'iters', 100, 'The iteration of evaluating the performance.')

flags.DEFINE_bool(
    'train_distributed', False, 'Whether to perform distributed training.')

flags.DEFINE_bool(
    'evaluation_distributed', False, 'Whether to perform distributed evaluation.')


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  use_bias=True,
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tensorflow.keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 2
num_classes = 10

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 3
depth = n * 9 + 2

def generate_pretrained_model():
    # Load the CIFAR10 data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Input image dimensions.
    input_shape = x_train.shape[1:]
    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    # Convert class vectors to binary class matrices.
    y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
    y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

    model = resnet_v2(input_shape=input_shape, depth=depth)

    model.compile(loss='categorical_crossentropy',
                optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.01),
                metrics=['accuracy'])
    model.summary()

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                cooldown=0,
                                patience=5,
                                min_lr=0.5e-6)

    callbacks = [lr_reducer, lr_scheduler]

    # Run training, with or without data augmentation.
    model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                shuffle=True,
                callbacks=callbacks)


    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    model.save("baseline_model")

class EvalDataset(object):
    def __init__(self, batch_size=100):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # If subtract pixel mean is enabled
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        print('y_train shape:', y_train.shape)

        # Convert class vectors to binary class matrices.
        y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
        y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
        self.test_images = x_test
        self.test_labels = y_test

    def __len__(self):
        return len(self.test_images)

    def __getitem__(self, idx):
        return self.test_images[idx], self.test_labels[idx]

class TrainDataset(object):
    def __init__(self, batch_size=100):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # If subtract pixel mean is enabled
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        print('y_train shape:', y_train.shape)

        # Convert class vectors to binary class matrices.
        y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
        y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
        self.test_images = x_test
        self.test_labels = y_test
        self.train_images = x_train
        self.train_labels = y_train

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        return self.train_images[idx], self.train_labels[idx]

def train(model, adaptor, compression_manager, train_dataloader):
    train_cfg = {
        'epoch': 8,
        'start_epoch': 0,
        'execution_mode': 'eager', 
        'criterion': {'CrossEntropyLoss': {'reduction': 'sum_over_batch_size'}}, 
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
    train_dataloader = DataLoader(framework='tensorflow', dataset=TrainDataset(), 
                                    batch_size=32, distributed=FLAGS.train_distributed)
    eval_dataloader = DataLoader(framework='tensorflow', dataset=EvalDataset(), 
                                    batch_size=32, distributed=FLAGS.evaluation_distributed)

    if FLAGS.prune:
        generate_pretrained_model()
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
            target_sparsity=0.25,
            start_step=FLAGS.start_step,
            end_step=FLAGS.end_step,
            pruning_op_types=['Conv', 'Dense']
        )
        compression_manager = prepare_compression(model='./baseline_model', confs=configs)
        compression_manager.callbacks.on_train_begin()
        model = compression_manager.model

        train(model, adaptor, compression_manager, train_dataloader)
        print("Pruned model score is ",evaluate(model, adaptor, eval_dataloader))

        compression_manager.callbacks.on_train_end()
        compression_manager.save(FLAGS.output_model)
        stats, sparsity = model.report_sparsity()
        logger.info(stats)
        logger.info(sparsity)
        
    if FLAGS.benchmark:
        from neural_compressor.benchmark import fit
        from neural_compressor.config import BenchmarkConfig
        conf = BenchmarkConfig(cores_per_instance=4, num_of_instance=1, iteration=FLAGS.iters)
        fit(FLAGS.input_model, conf, b_dataloader=eval_dataloader)