#!/usr/bin/env python
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
import os
from PIL import Image
from lpot.utils.utility import LazyImport
from lpot.utils import logger
from lpot.experimental.data.datasets import dataset_registry, IterableDataset, Dataset
tf = LazyImport('tensorflow')

# BELOW API TO BE DEPRECATED!
@dataset_registry(dataset_type="Imagenet", framework="tensorflow", dataset_format='')
class TensorflowImagenetDataset(IterableDataset):
    """Configuration for Imagenet dataset."""

    def __new__(cls, root, subset='validation', num_cores=28, transform=None, filter=None):

        assert subset in ('validation', 'train'), \
            'only support subset (validation, train)'
        logger.warning('This api is going to be deprecated, '
                       'please use ImageRecord instead')

        from tensorflow.python.platform import gfile
        glob_pattern = os.path.join(root, '%s-*-of-*' % subset)
        file_names = gfile.Glob(glob_pattern)
        if not file_names:
            raise ValueError('Found no files in --root matching: {}'.format(glob_pattern))

        from tensorflow.python.data.experimental import parallel_interleave
        ds = tf.data.TFRecordDataset.list_files(file_names, shuffle=False)
        ds = ds.apply(
          parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=num_cores))

        ds = ds.map(transform, num_parallel_calls=None)

        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # this number can be tuned
        return ds

    def __iter__(self):
        ds_iterator = tf.compat.v1.data.make_one_shot_iterator(self)
        iter_tensors = ds_iterator.get_next()
        from tensorflow.python.framework.errors_impl import OutOfRangeError
        data_config = tf.compat.v1.ConfigProto()
        data_config.use_per_session_threads = 1
        data_config.intra_op_parallelism_threads = 1
        data_config.inter_op_parallelism_threads = 16
        with tf.compat.v1.Session(config=data_config) as sess:
            while True:
                try:
                    outputs = sess.run(iter_tensors)
                    yield outputs
                except OutOfRangeError:
                    return

@dataset_registry(dataset_type="Imagenet", framework="onnxrt_qlinearops, \
                   onnxrt_integerops", dataset_format='')
class ONNXRTImagenetDataset(Dataset):
    """Configuration for Imagenet dataset."""

    def __init__(self, root, subset='val', num_cores=28, transform=None, filter=None):
        self.val_dir = os.path.join(root, subset)
        assert os.path.exists(self.val_dir), "find no val dir in {}".format(root) + \
            "please make sure there are train/val subfolders"
        import glob
        logger.warning('This api is going to be deprecated, ' + \
                       'please use ImageRecord instead')

        self.transform = transform
        self.image_list = []
        files = glob.glob(os.path.join(self.val_dir, '*'))
        files.sort()
        for idx, file in enumerate(files):
            imgs = glob.glob(os.path.join(file, '*'))
            for img in imgs:
                self.image_list.append((img, idx))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        from PIL import Image
        sample = self.image_list[index]
        image = Image.open(sample[0])
        if self.transform is not None:
            image, label = self.transform((image, sample[1]))
            return (image, label)

