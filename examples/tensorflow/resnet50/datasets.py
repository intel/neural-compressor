# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Benchmark dataset utilities.
"""

from abc import abstractmethod
import os

import tensorflow as tf


class Dataset(object):
    """Abstract class for cnn benchmarks dataset."""

    def __init__(self, name, data_dir=None):
        self.name = name
        if data_dir is None:
            raise ValueError('Data directory not specified')
        self.data_dir = data_dir

    def tf_record_pattern(self, subset):
        return os.path.join(self.data_dir, '%s-*-of-*' % subset)

    def reader(self):
        return tf.TFRecordReader()

    @abstractmethod
    def num_classes(self):
        pass

    @abstractmethod
    def num_examples_per_epoch(self, subset):
        pass

    def __str__(self):
        return self.name


class FlowersData(Dataset):

    def __init__(self, data_dir=None):
        super(FlowersData, self).__init__('Flowers', data_dir)

    def num_classes(self):
        return 5

    def num_examples_per_epoch(self, subset):
        if subset == 'train':
            return 3170
        elif subset == 'validation':
            return 500
        else:
            raise ValueError('Invalid data subset "%s"' % subset)


class ImagenetData(Dataset):

    def __init__(self, data_dir=None):
        super(ImagenetData, self).__init__('ImageNet', data_dir)

    def num_classes(self):
        return 1000

    def num_examples_per_epoch(self, subset='train'):
        if subset == 'train':
            return 1281167
        elif subset == 'validation':
            return 50000
        else:
            raise ValueError('Invalid data subset "%s"' % subset)
