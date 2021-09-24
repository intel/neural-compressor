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

from neural_compressor.experimental.data.dataloaders import DATALOADERS

# THIS API IS TO BE DEPRECATED!
class DataLoader(object):
    """Entrance of all configured DataLoaders. Will dispatch the DataLoaders to framework
       specific one. Users will be not aware of the dispatching, and the Interface is unified.

    """

    def __new__(cls, framework, dataset, batch_size=1, collate_fn=None,
                 last_batch='rollover', sampler=None, batch_sampler=None,
                 num_workers=0, pin_memory=False, shuffle=False, distributed=False):

        assert framework in ('tensorflow', 'tensorflow_itex',
                             'pytorch', 'pytorch_ipex', 'pytorch_fx', \
                             'onnxrt_qlinearops', 'onnxrt_integerops', 'mxnet'), \
                             "framework support tensorflow pytorch mxnet onnxruntime"
        return DATALOADERS[framework](dataset=dataset,
                                      batch_size=batch_size,
                                      last_batch=last_batch,
                                      collate_fn=collate_fn,
                                      sampler=sampler,
                                      batch_sampler=batch_sampler,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory,
                                      shuffle=shuffle,
                                      distributed=distributed)
