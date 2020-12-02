#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
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
from ilit.utils.utility import LazyImport
from .dataset import dataset_registry, IterableDataset
tf = LazyImport('tensorflow')

@dataset_registry(dataset_type="COCORecord", framework="tensorflow", dataset_format='')
class COCORecordDataset(IterableDataset):
    """Configuration for Coco dataset."""

    def __new__(cls, root, num_cores=28, transform=None):
        from tensorflow.python.data.experimental import parallel_interleave
        tfrecord_paths = [root]
        ds = tf.data.TFRecordDataset.list_files(tfrecord_paths)
        ds = ds.apply(
            parallel_interleave(tf.data.TFRecordDataset,
                                cycle_length=num_cores,
                                block_length=5,
                                sloppy=True,
                                buffer_output_elements=10000,
                                prefetch_input_elements=10000))
        ds = ds.map(transform, num_parallel_calls=None)
        ds = ds.prefetch(buffer_size=1000)
        ds.batch(1)
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
                    yield outputs[0]
                except OutOfRangeError:
                    return
