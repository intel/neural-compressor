#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import \
    Input, GlobalAveragePooling2D, Add, Dense, Activation, \
    BatchNormalization, Conv2D, MaxPooling2D

"""Script to get keras saved model.

Source:https://github.com/yusugomori/deeplearning-tf2/blob/master/models/resnet50_fashion_mnist_beginner.py
"""

class ResNet50(object):
    '''
    Reference:
        "Deep Residual Learning for Image Recognition"
        https://arxiv.org/abs/1512.03385
    '''
    def __init__(self, input_shape, output_dim):
        x = Input(shape=input_shape)
        h = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(h)
        h = self._building_block(h, channel_out=256)
        h = self._building_block(h, channel_out=256)
        h = self._building_block(h, channel_out=256)
        h = Conv2D(512, kernel_size=(1, 1), strides=(2, 2))(h)
        h = self._building_block(h, channel_out=512)
        h = self._building_block(h, channel_out=512)
        h = self._building_block(h, channel_out=512)
        h = self._building_block(h, channel_out=512)
        h = Conv2D(1024, kernel_size=(1, 1), strides=(2, 2))(h)
        h = self._building_block(h, channel_out=1024)
        h = self._building_block(h, channel_out=1024)
        h = self._building_block(h, channel_out=1024)
        h = self._building_block(h, channel_out=1024)
        h = self._building_block(h, channel_out=1024)
        h = self._building_block(h, channel_out=1024)
        h = Conv2D(2048, kernel_size=(1, 1), strides=(2, 2))(h)
        h = self._building_block(h, channel_out=2048)
        h = self._building_block(h, channel_out=2048)
        h = self._building_block(h, channel_out=2048)
        h = GlobalAveragePooling2D()(h)
        h = Dense(1000, activation='relu')(h)
        y = Dense(output_dim, activation='softmax')(h)
        self.model = Model(x, y)

    def __call__(self):
        return self.model

    def _building_block(self, x, channel_out=256):
        channel = channel_out // 4
        h = Conv2D(channel, kernel_size=(1, 1), padding='same')(x)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Conv2D(channel, kernel_size=(3, 3), padding='same')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Conv2D(channel_out, kernel_size=(1, 1), padding='same')(h)
        h = BatchNormalization()(h)
        shortcut = self._shortcut(x, output_shape=h.get_shape().as_list())
        h = Add()([h, shortcut])
        return Activation('relu')(h)

    def _shortcut(self, x, output_shape):
        input_shape = x.get_shape().as_list()
        channel_in = input_shape[-1]
        channel_out = output_shape[-1]

        if channel_in != channel_out:
            return self._projection(x, channel_out)
        else:
            return x

    def _projection(self, x, channel_out):
        return Conv2D(channel_out, kernel_size=(1, 1), padding='same')(x)


if __name__ == '__main__':
    np.random.seed(1234)
    tf.random.set_seed(1234)

    '''
    Load data
    '''
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
    x_test = (x_test.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
    y_train = np.eye(10)[y_train].astype(np.float32)
    y_test = np.eye(10)[y_test].astype(np.float32)

    '''
    Build model
    '''
    resnet = ResNet50((28, 28, 1), 10)
    model = resnet()
    # model.summary()
    model.compile('adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    '''
    Train model
    '''
    model.fit(x_train, y_train, epochs=1)
    model.evaluate(x_test, y_test)
    model.save('./resnet50_fashion')

