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

from .utils import logger
from .data import DATALOADERS
from .experimental import Benchmark as ExpBenchmark

class Benchmark(object):
    """Benchmark class can be used to evaluate the model performance, with the objective
       setting, user can get the data of what they configured in yaml

    Args:
        conf_fname_or_obj (string or obj): The path to the YAML configuration file or 
            Benchmark_Conf class containing accuracy goal, tuning objective and preferred
            calibration & quantization tuning space etc.

    """

    def __init__(self, conf_fname_or_obj):
        self.exp_benchmarker = ExpBenchmark(conf_fname_or_obj)

    def __call__(self, model, b_dataloader=None, b_func=None):

        logger.warning("This API is going to be deprecated. Please import "
            "neural_compressor.experimental.Bencharmk, initialize an instance of `Benchmark`,"
            "set its dataloader and metric attributes, then invoke its __call__ method.")

        self.exp_benchmarker.model = model
        if b_dataloader is not None:
            self.exp_benchmarker.b_dataloader = b_dataloader
        elif b_func is not None:
            logger.warning("Benchmark will do nothing when you input b_func.")
        self.exp_benchmarker()
        return self.exp_benchmarker.results

    def dataloader(self, dataset, batch_size=1, collate_fn=None, last_batch='rollover',
                   sampler=None, batch_sampler=None, num_workers=0, pin_memory=False,
                   shuffle=False, distributed=False):
        return DATALOADERS[self.exp_benchmarker.framework](dataset=dataset,
                          batch_size=batch_size, collate_fn=collate_fn, last_batch=last_batch,
                          sampler=sampler, batch_sampler=batch_sampler, num_workers=num_workers,
                          pin_memory=pin_memory, shuffle=shuffle, distributed=distributed)

    def metric(self, name, metric_cls, **kwargs):
        from .experimental.common import Metric as NCMetric
        nc_metric = NCMetric(metric_cls, name, **kwargs)
        self.exp_benchmarker.metric = nc_metric

    def postprocess(self, name, postprocess_cls, **kwargs):
        from .experimental.common import Postprocess as NCPostprocess
        nc_postprocess = NCPostprocess(postprocess_cls, name, **kwargs)
        self.exp_benchmarker.postprocess = nc_postprocess
