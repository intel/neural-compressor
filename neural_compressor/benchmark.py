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

"""Benchmark is used for evaluating the model performance."""

from .utils import logger
from .data import DATALOADERS
from .experimental import Benchmark as ExpBenchmark
from .conf.pythonic_config import Config
from .config import BenchmarkConfig

class Benchmark(object):
    """Benchmark class can be used to evaluate the model performance.
    
    With the objective setting, user can get the data of what they configured in yaml.

    Args:
        conf_fname_or_obj (string or obj): The path to the YAML configuration file or 
            Benchmark_Conf class containing accuracy goal, tuning objective and preferred
            calibration & quantization tuning space etc.

    """
    def __init__(self, conf_fname_or_obj):
        """Init a Benchmark object."""
        self.exp_benchmarker = ExpBenchmark(conf_fname_or_obj)

    def __call__(self, model, b_dataloader=None, b_func=None):
        """Directly call a Benchmark object.

        Args:
            model: Get the model
            b_dataloader: Set dataloader for benchmarking
            b_func: Eval function for benchmark
        """
        logger.warning("This API is going to be deprecated. Please "
            "use benchmark(model, config, b_dataloader=None, b_func=None) instead.")

        self.exp_benchmarker.model = model
        if b_dataloader is not None:
            self.exp_benchmarker.b_dataloader = b_dataloader
        elif b_func is not None:
            logger.warning("Benchmark will do nothing when you input b_func.")
        self.exp_benchmarker()
        return self.exp_benchmarker.results

    fit = __call__

    def dataloader(self, dataset, batch_size=1, collate_fn=None, last_batch='rollover',
                   sampler=None, batch_sampler=None, num_workers=0, pin_memory=False,
                   shuffle=False, distributed=False):
        """Set dataloader for benchmarking."""
        return DATALOADERS[self.exp_benchmarker.framework](dataset=dataset,
                          batch_size=batch_size, collate_fn=collate_fn, last_batch=last_batch,
                          sampler=sampler, batch_sampler=batch_sampler, num_workers=num_workers,
                          pin_memory=pin_memory, shuffle=shuffle, distributed=distributed)

    def metric(self, name, metric_cls, **kwargs):
        """Set the metric class and Neural Compressor will initialize this class when evaluation.

        Neural Compressor has many built-in metrics, but users can set specific metrics through
        this api. The metric class should take the outputs of the model or
        postprocess (if have) as inputs. Neural Compressor built-in metrics always take
        (predictions, labels) as inputs for update,
        and user_metric.metric_cls should be a sub_class of neural_compressor.metric.metric
        or an user-defined metric object

        Args:
            metric_cls (cls): Should be a sub_class of neural_compressor.metric.BaseMetric, 
                which takes (predictions, labels) as inputs
            name (str, optional): Name for metric. Defaults to 'user_metric'.
        """
        from .experimental.common import Metric as NCMetric
        nc_metric = NCMetric(metric_cls, name, **kwargs)
        self.exp_benchmarker.metric = nc_metric

    def postprocess(self, name, postprocess_cls, **kwargs):
        """Set postprocess class and neural_compressor will initialize this class when evaluation.

        The postprocess function should take the outputs of the model as inputs, and
        outputs (predictions, labels) as inputs for metric updates.
        
        Args:
        name (str, optional): Name for postprocess.
        postprocess_cls (cls): Should be a sub_class of neural_compressor.data.transforms.postprocess.

        """
        from .experimental.common import Postprocess as NCPostprocess
        nc_postprocess = NCPostprocess(postprocess_cls, name, **kwargs)
        self.exp_benchmarker.postprocess = nc_postprocess


def fit(model, config=None, b_dataloader=None, b_func=None):
    """Benchmark the model performance with the configure.

    Args:
        model (object):           The model to be benchmarked.
        config (BenchmarkConfig): The configuration for benchmark containing accuracy goal,
                                  tuning objective and preferred calibration & quantization
                                  tuning space etc.
        b_dataloader:             The dataloader for frameworks.
        b_func:                   customized benchmark function. if user passes the dataloader,
                                  then b_func is not needed.
    """
    if isinstance(config, BenchmarkConfig):
        config = Config(benchmark=config)
    benchmarker = ExpBenchmark(config)
    benchmarker.model = model
    if b_func is not None:
        benchmarker.b_func = b_func
    if b_dataloader is not None:
        benchmarker.b_dataloader = b_dataloader
    benchmarker()
    return benchmarker.results
